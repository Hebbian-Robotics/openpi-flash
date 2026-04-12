use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use quinn::{Endpoint, Incoming, RecvStream, SendStream, ServerConfig, TransportConfig};
use rcgen::generate_simple_self_signed;
use rustls::crypto::ring::default_provider;
use rustls_pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;
use tracing::{info, warn};

const REQUEST_TYPE_METADATA: u8 = 0x01;
const REQUEST_TYPE_INFER: u8 = 0x02;

const RESPONSE_TYPE_METADATA: u8 = 0x11;
const RESPONSE_TYPE_INFER: u8 = 0x12;
const RESPONSE_TYPE_ERROR: u8 = 0x13;

const QUIC_MESSAGE_TYPE_DATA: u8 = 0x00;
const QUIC_MESSAGE_TYPE_ERROR: u8 = 0x01;

#[derive(Parser, Debug)]
#[command(
    name = "openpi-quic-sidecar",
    about = "Rust QUIC sidecar for openpi-hosting"
)]
struct Args {
    /// UDP port to listen on for direct QUIC clients.
    #[arg(long, default_value_t = 5555)]
    listen_port: u16,

    /// Unix socket path for the Python inference backend.
    #[arg(long)]
    backend_socket_path: PathBuf,

    /// Idle timeout for the QUIC connection.
    #[arg(long, default_value_t = 10)]
    max_idle_timeout_secs: u64,

    /// Keep-alive interval for the QUIC connection.
    #[arg(long, default_value_t = 2)]
    keep_alive_interval_secs: u64,

    /// Initial congestion window for the QUIC transport.
    #[arg(long, default_value_t = 1024 * 1024)]
    initial_window_bytes: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let endpoint = build_server_endpoint(&args)?;
    let listen_address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.listen_port));
    info!("Listening for QUIC connections on udp://{listen_address}");
    info!(
        "Forwarding requests to Python backend at {}",
        args.backend_socket_path.display()
    );

    loop {
        let Some(connecting_client) = endpoint.accept().await else {
            bail!("QUIC endpoint stopped accepting connections unexpectedly");
        };

        if let Err(error) =
            handle_client_connection(connecting_client, &args.backend_socket_path).await
        {
            warn!("QUIC client session ended with error: {error:#}");
        }
    }
}

fn build_server_endpoint(args: &Args) -> Result<Endpoint> {
    let server_config = build_server_config(args)?;
    let listen_address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, args.listen_port));
    Endpoint::server(server_config, listen_address)
        .with_context(|| format!("Failed to bind QUIC server to {listen_address}"))
}

fn build_server_config(args: &Args) -> Result<ServerConfig> {
    let generated_certificate = generate_simple_self_signed(vec!["localhost".to_string()])
        .context("Failed to generate self-signed certificate")?;
    let certificate_der = generated_certificate.cert.der().clone();
    let private_key_der = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(
        generated_certificate.signing_key.serialize_der(),
    ));

    let rustls_server_config =
        rustls::ServerConfig::builder_with_provider(default_provider().into())
            .with_safe_default_protocol_versions()
            .context("Failed to create rustls server config")?
            .with_no_client_auth()
            .with_single_cert(vec![certificate_der], private_key_der)
            .context("Failed to attach certificate to rustls config")?;

    let quic_server_config =
        quinn::crypto::rustls::QuicServerConfig::try_from(rustls_server_config)
            .context("Failed to convert rustls server config to QUIC config")?;

    let mut server_config = ServerConfig::with_crypto(Arc::new(quic_server_config));
    server_config.transport = Arc::new(build_transport_config(args));
    Ok(server_config)
}

fn build_transport_config(args: &Args) -> TransportConfig {
    let mut transport_config = TransportConfig::default();
    transport_config.max_idle_timeout(Some(
        Duration::from_secs(args.max_idle_timeout_secs)
            .try_into()
            .expect("valid timeout"),
    ));
    transport_config.keep_alive_interval(Some(Duration::from_secs(args.keep_alive_interval_secs)));
    transport_config.receive_window(10_000_000_u32.into());
    transport_config.send_window(10_000_000);
    transport_config.stream_receive_window(5_000_000_u32.into());
    transport_config.initial_rtt(Duration::from_millis(20));
    transport_config.max_concurrent_bidi_streams(100_u32.into());
    transport_config.max_concurrent_uni_streams(100_u32.into());
    transport_config.datagram_receive_buffer_size(Some(5_000_000));
    transport_config.datagram_send_buffer_size(5_000_000);
    transport_config.min_mtu(1200);
    transport_config.mtu_discovery_config(None);

    let mut cubic_config = quinn::congestion::CubicConfig::default();
    cubic_config.initial_window(args.initial_window_bytes);
    transport_config.congestion_controller_factory(Arc::new(cubic_config));
    transport_config
}

async fn handle_client_connection(
    incoming_client: Incoming,
    backend_socket_path: &Path,
) -> Result<()> {
    let quic_connection = incoming_client
        .await
        .context("Failed to establish QUIC connection")?;
    let remote_address = quic_connection.remote_address();
    info!("Client connected from {remote_address}");

    let (mut send_stream, mut recv_stream) = quic_connection
        .accept_bi()
        .await
        .context("Client did not open a bidirectional stream")?;
    let mut backend_stream = UnixStream::connect(backend_socket_path)
        .await
        .with_context(|| {
            format!(
                "Failed to connect to backend socket {}",
                backend_socket_path.display()
            )
        })?;

    handle_quic_message_handshake(&mut send_stream, &mut recv_stream, &mut backend_stream)
        .await
        .context("Failed during QUIC handshake")?;

    loop {
        let Some(quic_message_payload) = read_length_prefixed_message(&mut recv_stream).await?
        else {
            info!("Client {remote_address} disconnected");
            break;
        };

        let (quic_message_type, quic_message_body) =
            split_message_type_and_body(&quic_message_payload).context("Invalid QUIC message")?;
        if quic_message_type == QUIC_MESSAGE_TYPE_ERROR {
            bail!(
                "Client sent QUIC error message: {}",
                String::from_utf8_lossy(quic_message_body)
            );
        }
        if quic_message_type != QUIC_MESSAGE_TYPE_DATA {
            bail!("Unexpected QUIC message type: {quic_message_type:#x}");
        }

        write_local_backend_request(&mut backend_stream, REQUEST_TYPE_INFER, quic_message_body)
            .await
            .context("Failed to forward inference request to backend")?;
        let backend_response = read_local_backend_response(&mut backend_stream)
            .await
            .context("Failed to read backend inference response")?;
        forward_backend_response_to_quic(&mut send_stream, &backend_response)
            .await
            .context("Failed to forward backend response to QUIC client")?;
    }

    send_stream
        .finish()
        .context("Failed to finish QUIC send stream")?;
    quic_connection.close(0_u32.into(), b"client disconnected");
    Ok(())
}

async fn handle_quic_message_handshake(
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
    backend_stream: &mut UnixStream,
) -> Result<()> {
    let client_handshake_payload = read_length_prefixed_message(recv_stream)
        .await?
        .ok_or_else(|| anyhow!("Client disconnected before sending handshake"))?;
    let (quic_message_type, _quic_message_body) =
        split_message_type_and_body(&client_handshake_payload)
            .context("Invalid handshake payload")?;
    if quic_message_type != QUIC_MESSAGE_TYPE_DATA {
        bail!("Expected handshake DATA message, got type {quic_message_type:#x}");
    }

    write_local_backend_request(backend_stream, REQUEST_TYPE_METADATA, &[])
        .await
        .context("Failed to request metadata from backend")?;
    let backend_response = read_local_backend_response(backend_stream)
        .await
        .context("Failed to read backend metadata response")?;
    forward_backend_response_to_quic(send_stream, &backend_response)
        .await
        .context("Failed to send metadata to QUIC client")?;
    Ok(())
}

async fn forward_backend_response_to_quic(
    send_stream: &mut SendStream,
    backend_response: &BackendResponse,
) -> Result<()> {
    match backend_response.response_type {
        RESPONSE_TYPE_METADATA | RESPONSE_TYPE_INFER => {
            let mut quic_payload = Vec::with_capacity(1 + backend_response.response_body.len());
            quic_payload.push(QUIC_MESSAGE_TYPE_DATA);
            quic_payload.extend_from_slice(&backend_response.response_body);
            write_length_prefixed_message(send_stream, &quic_payload).await
        }
        RESPONSE_TYPE_ERROR => {
            let mut quic_payload = Vec::with_capacity(1 + backend_response.response_body.len());
            quic_payload.push(QUIC_MESSAGE_TYPE_ERROR);
            quic_payload.extend_from_slice(&backend_response.response_body);
            write_length_prefixed_message(send_stream, &quic_payload).await
        }
        unexpected_response_type => {
            bail!("Unexpected local backend response type: {unexpected_response_type:#x}")
        }
    }
}

struct BackendResponse {
    response_type: u8,
    response_body: Vec<u8>,
}

async fn read_local_backend_response(backend_stream: &mut UnixStream) -> Result<BackendResponse> {
    let response_payload = read_length_prefixed_message(backend_stream)
        .await?
        .ok_or_else(|| anyhow!("Backend disconnected unexpectedly"))?;
    let (response_type, response_body) =
        split_message_type_and_body(&response_payload).context("Invalid backend response")?;
    Ok(BackendResponse {
        response_type,
        response_body: response_body.to_vec(),
    })
}

async fn write_local_backend_request(
    backend_stream: &mut UnixStream,
    request_type: u8,
    request_body: &[u8],
) -> Result<()> {
    let mut framed_request = Vec::with_capacity(1 + request_body.len());
    framed_request.push(request_type);
    framed_request.extend_from_slice(request_body);
    write_length_prefixed_message(backend_stream, &framed_request).await
}

fn split_message_type_and_body(message_payload: &[u8]) -> Result<(u8, &[u8])> {
    let Some((&message_type, message_body)) = message_payload.split_first() else {
        bail!("Received empty message payload");
    };
    Ok((message_type, message_body))
}

async fn read_length_prefixed_message<StreamType>(
    stream: &mut StreamType,
) -> Result<Option<Vec<u8>>>
where
    StreamType: AsyncRead + Unpin,
{
    let mut raw_length_prefix = [0_u8; 4];
    match stream.read_exact(&mut raw_length_prefix).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error).context("Failed to read message length prefix"),
    }

    let payload_length = u32::from_be_bytes(raw_length_prefix) as usize;
    let mut payload = vec![0_u8; payload_length];
    if payload_length > 0 {
        stream
            .read_exact(&mut payload)
            .await
            .context("Failed to read framed message payload")?;
    }
    Ok(Some(payload))
}

async fn write_length_prefixed_message<StreamType>(
    stream: &mut StreamType,
    payload: &[u8],
) -> Result<()>
where
    StreamType: AsyncWrite + Unpin,
{
    let payload_length = u32::try_from(payload.len()).context("Payload too large to frame")?;
    stream
        .write_all(&payload_length.to_be_bytes())
        .await
        .context("Failed to write message length prefix")?;
    if !payload.is_empty() {
        stream
            .write_all(payload)
            .await
            .context("Failed to write message payload")?;
    }
    stream
        .flush()
        .await
        .context("Failed to flush framed message")?;
    Ok(())
}
