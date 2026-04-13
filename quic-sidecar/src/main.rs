use std::fs;
use std::net::{Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use quinn::crypto::rustls::QuicClientConfig;
use quinn::{
    ClientConfig, Endpoint, Incoming, RecvStream, SendStream, ServerConfig, TransportConfig,
};
use rcgen::generate_simple_self_signed;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::ring::default_provider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tracing::{info, warn};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalRequestType {
    Metadata = 0x01,
    Infer = 0x02,
    Reset = 0x03,
}

impl LocalRequestType {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::Metadata),
            0x02 => Ok(Self::Infer),
            0x03 => Ok(Self::Reset),
            _ => bail!("Unknown local request type: {value:#x}"),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalResponseType {
    Metadata = 0x11,
    Infer = 0x12,
    Error = 0x13,
    Reset = 0x14,
}

impl LocalResponseType {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x11 => Ok(Self::Metadata),
            0x12 => Ok(Self::Infer),
            0x13 => Ok(Self::Error),
            0x14 => Ok(Self::Reset),
            _ => bail!("Unknown local response type: {value:#x}"),
        }
    }
}

#[repr(u8)]
#[derive(PartialEq, Eq)]
enum QuicMessageType {
    Data = 0x00,
    Error = 0x01,
}

impl QuicMessageType {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x00 => Ok(Self::Data),
            0x01 => Ok(Self::Error),
            _ => bail!("Unknown QUIC message type: {value:#x}"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "openpi-quic-sidecar",
    about = "Rust QUIC transport sidecar for openpi-flash"
)]
struct Cli {
    #[command(subcommand)]
    command: SidecarCommand,
}

#[derive(Subcommand, Debug)]
enum SidecarCommand {
    /// Run the server-side QUIC listener used by the Docker/EC2 deployment.
    Server(ServerArgs),
    /// Run a local client-side QUIC sidecar that Python can talk to over a Unix socket.
    Client(ClientArgs),
}

#[derive(Args, Debug)]
struct ServerArgs {
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

#[derive(Args, Debug)]
struct ClientArgs {
    /// Hostname or IP address of the remote QUIC server.
    #[arg(long)]
    server_host: String,

    /// UDP port of the remote QUIC server.
    #[arg(long, default_value_t = 5555)]
    server_port: u16,

    /// Local UDP port used by the QUIC client socket.
    #[arg(long, default_value_t = 5556)]
    local_port: u16,

    /// Unix socket path exposed to the local Python wrapper.
    #[arg(long)]
    local_socket_path: PathBuf,

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
    let _ = default_provider().install_default();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        SidecarCommand::Server(server_args) => run_server_mode(&server_args).await,
        SidecarCommand::Client(client_args) => run_client_mode(&client_args).await,
    }
}

async fn run_server_mode(server_args: &ServerArgs) -> Result<()> {
    let endpoint = build_server_endpoint(server_args)?;
    let listen_address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, server_args.listen_port));
    info!("Listening for QUIC connections on udp://{listen_address}");
    info!(
        "Forwarding requests to Python backend at {}",
        server_args.backend_socket_path.display()
    );

    loop {
        let Some(connecting_client) = endpoint.accept().await else {
            bail!("QUIC endpoint stopped accepting connections unexpectedly");
        };

        if let Err(error) =
            handle_server_connection(connecting_client, &server_args.backend_socket_path).await
        {
            warn!("QUIC client session ended with error: {error:#}");
        }
    }
}

async fn run_client_mode(client_args: &ClientArgs) -> Result<()> {
    let remote_server_address =
        resolve_remote_server_address(&client_args.server_host, client_args.server_port)?;
    let endpoint = build_client_endpoint(client_args)?;
    info!(
        "Connecting local QUIC sidecar to remote server at udp://{}",
        remote_server_address
    );

    let quic_connection = endpoint
        .connect(remote_server_address, "localhost")
        .context("Failed to start QUIC connect attempt")?
        .await
        .context("Failed to establish direct QUIC connection")?;
    info!("Connected to remote QUIC server at {remote_server_address}");

    let (mut send_stream, mut recv_stream) = quic_connection
        .open_bi()
        .await
        .context("Failed to open QUIC bidirectional stream")?;
    let server_metadata = perform_client_handshake(&mut send_stream, &mut recv_stream)
        .await
        .context("Failed during direct QUIC client handshake")?;

    serve_local_client_socket(
        &client_args.local_socket_path,
        &mut send_stream,
        &mut recv_stream,
        &server_metadata,
    )
    .await?;

    let _ = send_stream.finish();
    quic_connection.close(0_u32.into(), b"local sidecar shutting down");
    endpoint.wait_idle().await;
    Ok(())
}

fn build_server_endpoint(server_args: &ServerArgs) -> Result<Endpoint> {
    let server_config = build_server_config(
        server_args.max_idle_timeout_secs,
        server_args.keep_alive_interval_secs,
        server_args.initial_window_bytes,
    )?;
    let listen_address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, server_args.listen_port));
    Endpoint::server(server_config, listen_address)
        .with_context(|| format!("Failed to bind QUIC server to {listen_address}"))
}

fn build_server_config(
    max_idle_timeout_secs: u64,
    keep_alive_interval_secs: u64,
    initial_window_bytes: u64,
) -> Result<ServerConfig> {
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
    server_config.transport = Arc::new(build_transport_config(
        max_idle_timeout_secs,
        keep_alive_interval_secs,
        initial_window_bytes,
    ));
    Ok(server_config)
}

fn build_client_endpoint(client_args: &ClientArgs) -> Result<Endpoint> {
    let bind_address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, client_args.local_port));
    let mut endpoint = Endpoint::client(bind_address)
        .with_context(|| format!("Failed to bind UDP {bind_address}"))?;
    endpoint.set_default_client_config(build_client_config(
        client_args.max_idle_timeout_secs,
        client_args.keep_alive_interval_secs,
        client_args.initial_window_bytes,
    )?);
    Ok(endpoint)
}

fn build_client_config(
    max_idle_timeout_secs: u64,
    keep_alive_interval_secs: u64,
    initial_window_bytes: u64,
) -> Result<ClientConfig> {
    let rustls_client_config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();
    let mut client_config =
        ClientConfig::new(Arc::new(QuicClientConfig::try_from(rustls_client_config)?));
    client_config.transport_config(Arc::new(build_transport_config(
        max_idle_timeout_secs,
        keep_alive_interval_secs,
        initial_window_bytes,
    )));
    Ok(client_config)
}

fn build_transport_config(
    max_idle_timeout_secs: u64,
    keep_alive_interval_secs: u64,
    initial_window_bytes: u64,
) -> TransportConfig {
    let mut transport_config = TransportConfig::default();
    transport_config.max_idle_timeout(Some(
        Duration::from_secs(max_idle_timeout_secs)
            .try_into()
            .expect("valid timeout"),
    ));
    transport_config.keep_alive_interval(Some(Duration::from_secs(keep_alive_interval_secs)));
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
    cubic_config.initial_window(initial_window_bytes);
    transport_config.congestion_controller_factory(Arc::new(cubic_config));
    transport_config
}

async fn handle_server_connection(
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

    handle_server_handshake(&mut send_stream, &mut recv_stream, &mut backend_stream)
        .await
        .context("Failed during QUIC handshake")?;

    loop {
        let quic_message_payload = match read_length_prefixed_message(&mut recv_stream).await {
            Ok(Some(quic_message_payload)) => quic_message_payload,
            Ok(None) => {
                info!("Client {remote_address} disconnected");
                break;
            }
            Err(error) if is_expected_peer_disconnect_error(&error) => {
                info!("Client {remote_address} disconnected");
                break;
            }
            Err(error) => {
                return Err(error).context("Failed while reading QUIC client message");
            }
        };

        let (raw_quic_message_type, quic_message_body) =
            split_message_type_and_body(&quic_message_payload).context("Invalid QUIC message")?;
        match QuicMessageType::from_u8(raw_quic_message_type)? {
            QuicMessageType::Error => {
                bail!(
                    "Client sent QUIC error message: {}",
                    String::from_utf8_lossy(quic_message_body)
                );
            }
            QuicMessageType::Data => {}
        }

        write_local_backend_request(
            &mut backend_stream,
            LocalRequestType::Infer,
            quic_message_body,
        )
        .await
        .context("Failed to forward inference request to backend")?;
        let backend_response = read_local_backend_response(&mut backend_stream)
            .await
            .context("Failed to read backend inference response")?;
        forward_backend_response_to_quic(&mut send_stream, &backend_response)
            .await
            .context("Failed to forward backend response to QUIC client")?;
    }

    let _ = send_stream.finish();
    quic_connection.close(0_u32.into(), b"client disconnected");
    Ok(())
}

async fn handle_server_handshake(
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
    backend_stream: &mut UnixStream,
) -> Result<()> {
    let client_handshake_payload = read_length_prefixed_message(recv_stream)
        .await?
        .ok_or_else(|| anyhow!("Client disconnected before sending handshake"))?;
    let (raw_quic_message_type, _quic_message_body) =
        split_message_type_and_body(&client_handshake_payload)
            .context("Invalid handshake payload")?;
    match QuicMessageType::from_u8(raw_quic_message_type).context("Invalid handshake payload")? {
        QuicMessageType::Data => {}
        QuicMessageType::Error => bail!("Client sent error during handshake"),
    }

    write_local_backend_request(backend_stream, LocalRequestType::Metadata, &[])
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

async fn perform_client_handshake(
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
) -> Result<Vec<u8>> {
    write_length_prefixed_message(send_stream, &[QuicMessageType::Data as u8])
        .await
        .context("Failed to send direct QUIC client handshake")?;
    let metadata_payload = read_length_prefixed_message(recv_stream)
        .await?
        .ok_or_else(|| anyhow!("Remote server disconnected before sending metadata"))?;
    let (raw_quic_message_type, metadata_body) =
        split_message_type_and_body(&metadata_payload).context("Invalid metadata payload")?;
    match QuicMessageType::from_u8(raw_quic_message_type)? {
        QuicMessageType::Data => Ok(metadata_body.to_vec()),
        QuicMessageType::Error => bail!(
            "Remote server returned error during handshake: {}",
            String::from_utf8_lossy(metadata_body)
        ),
    }
}

async fn serve_local_client_socket(
    local_socket_path: &Path,
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
    server_metadata: &[u8],
) -> Result<()> {
    remove_stale_local_socket_file(local_socket_path)?;
    let local_listener = UnixListener::bind(local_socket_path).with_context(|| {
        format!(
            "Failed to bind local Unix socket for Python client at {}",
            local_socket_path.display()
        )
    })?;
    info!(
        "Local client sidecar socket ready at {}",
        local_socket_path.display()
    );

    loop {
        let (mut local_client_stream, _) = local_listener
            .accept()
            .await
            .context("Failed to accept local Python client connection")?;
        info!("Local Python client connected");

        if let Err(error) = serve_local_client_connection(
            &mut local_client_stream,
            send_stream,
            recv_stream,
            server_metadata,
        )
        .await
        {
            warn!("Local Python client session ended with error: {error:#}");
        } else {
            info!("Local Python client disconnected");
        }
    }
}

async fn serve_local_client_connection(
    local_client_stream: &mut UnixStream,
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
    server_metadata: &[u8],
) -> Result<()> {
    loop {
        let Some(local_request_payload) = read_length_prefixed_message(local_client_stream).await?
        else {
            return Ok(());
        };

        let (raw_local_request_type, local_request_body) =
            split_message_type_and_body(&local_request_payload)
                .context("Invalid local sidecar request payload")?;
        let local_request_type = LocalRequestType::from_u8(raw_local_request_type)?;

        match local_request_type {
            LocalRequestType::Metadata => {
                write_local_response(
                    local_client_stream,
                    LocalResponseType::Metadata,
                    server_metadata,
                )
                .await?;
            }
            LocalRequestType::Reset => {
                write_local_response(local_client_stream, LocalResponseType::Reset, &[]).await?;
            }
            LocalRequestType::Infer => {
                let remote_response = forward_inference_request_to_remote_quic(
                    send_stream,
                    recv_stream,
                    local_request_body,
                )
                .await?;
                write_framed_message(local_client_stream, &remote_response).await?;
            }
        }
    }
}

async fn forward_inference_request_to_remote_quic(
    send_stream: &mut SendStream,
    recv_stream: &mut RecvStream,
    request_body: &[u8],
) -> Result<Vec<u8>> {
    let mut quic_payload = Vec::with_capacity(1 + request_body.len());
    quic_payload.push(QuicMessageType::Data as u8);
    quic_payload.extend_from_slice(request_body);
    write_length_prefixed_message(send_stream, &quic_payload)
        .await
        .context("Failed to forward local inference request to remote QUIC server")?;

    let remote_response_payload = read_length_prefixed_message(recv_stream)
        .await?
        .ok_or_else(|| anyhow!("Remote QUIC server disconnected during inference"))?;
    let (raw_quic_message_type, remote_response_body) =
        split_message_type_and_body(&remote_response_payload)
            .context("Invalid remote QUIC response payload")?;

    match QuicMessageType::from_u8(raw_quic_message_type)? {
        QuicMessageType::Data => Ok(make_local_response(
            LocalResponseType::Infer,
            remote_response_body,
        )),
        QuicMessageType::Error => Ok(make_local_response(
            LocalResponseType::Error,
            remote_response_body,
        )),
    }
}

async fn forward_backend_response_to_quic(
    send_stream: &mut SendStream,
    backend_response: &LocalBackendResponse,
) -> Result<()> {
    match backend_response.response_type {
        LocalResponseType::Metadata | LocalResponseType::Infer => {
            let mut quic_payload = Vec::with_capacity(1 + backend_response.response_body.len());
            quic_payload.push(QuicMessageType::Data as u8);
            quic_payload.extend_from_slice(&backend_response.response_body);
            write_length_prefixed_message(send_stream, &quic_payload).await
        }
        LocalResponseType::Error => {
            let mut quic_payload = Vec::with_capacity(1 + backend_response.response_body.len());
            quic_payload.push(QuicMessageType::Error as u8);
            quic_payload.extend_from_slice(&backend_response.response_body);
            write_length_prefixed_message(send_stream, &quic_payload).await
        }
        LocalResponseType::Reset => {
            bail!("Server backend returned unexpected RESET response")
        }
    }
}

struct LocalBackendResponse {
    response_type: LocalResponseType,
    response_body: Vec<u8>,
}

async fn read_local_backend_response(
    backend_stream: &mut UnixStream,
) -> Result<LocalBackendResponse> {
    let response_payload = read_length_prefixed_message(backend_stream)
        .await?
        .ok_or_else(|| anyhow!("Backend disconnected unexpectedly"))?;
    let (raw_response_type, response_body) =
        split_message_type_and_body(&response_payload).context("Invalid backend response")?;
    let response_type =
        LocalResponseType::from_u8(raw_response_type).context("Invalid backend response type")?;
    Ok(LocalBackendResponse {
        response_type,
        response_body: response_body.to_vec(),
    })
}

async fn write_local_backend_request(
    backend_stream: &mut UnixStream,
    request_type: LocalRequestType,
    request_body: &[u8],
) -> Result<()> {
    let mut framed_request = Vec::with_capacity(1 + request_body.len());
    framed_request.push(request_type as u8);
    framed_request.extend_from_slice(request_body);
    write_length_prefixed_message(backend_stream, &framed_request).await
}

async fn write_local_response<StreamType>(
    stream: &mut StreamType,
    response_type: LocalResponseType,
    response_body: &[u8],
) -> Result<()>
where
    StreamType: AsyncWrite + Unpin,
{
    let response_payload = make_local_response(response_type, response_body);
    write_framed_message(stream, &response_payload).await
}

fn make_local_response(response_type: LocalResponseType, response_body: &[u8]) -> Vec<u8> {
    let mut response_payload = Vec::with_capacity(1 + response_body.len());
    response_payload.push(response_type as u8);
    response_payload.extend_from_slice(response_body);
    response_payload
}

fn resolve_remote_server_address(server_host: &str, server_port: u16) -> Result<SocketAddr> {
    (server_host, server_port)
        .to_socket_addrs()
        .with_context(|| {
            format!("Failed to resolve remote QUIC server {server_host}:{server_port}")
        })?
        .find(SocketAddr::is_ipv4)
        .ok_or_else(|| anyhow!("Could not resolve an IPv4 address for {server_host}:{server_port}"))
}

fn remove_stale_local_socket_file(local_socket_path: &Path) -> Result<()> {
    if local_socket_path.exists() {
        fs::remove_file(local_socket_path).with_context(|| {
            format!(
                "Failed to remove existing local socket file {}",
                local_socket_path.display()
            )
        })?;
    }

    if let Some(parent_directory) = local_socket_path.parent() {
        fs::create_dir_all(parent_directory).with_context(|| {
            format!(
                "Failed to create parent directory for local socket {}",
                local_socket_path.display()
            )
        })?;
    }

    Ok(())
}

fn split_message_type_and_body(message_payload: &[u8]) -> Result<(u8, &[u8])> {
    let Some((&message_type, message_body)) = message_payload.split_first() else {
        bail!("Received empty message payload");
    };
    Ok((message_type, message_body))
}

fn is_expected_peer_disconnect_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io_error| {
                matches!(
                    io_error.kind(),
                    std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::BrokenPipe
                        | std::io::ErrorKind::NotConnected
                )
            })
    })
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

async fn write_framed_message<StreamType>(stream: &mut StreamType, payload: &[u8]) -> Result<()>
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

async fn write_length_prefixed_message<StreamType>(
    stream: &mut StreamType,
    payload: &[u8],
) -> Result<()>
where
    StreamType: AsyncWrite + Unpin,
{
    write_framed_message(stream, payload).await
}

#[derive(Debug)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(default_provider())))
    }
}

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        digitally_signed_struct: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            digitally_signed_struct,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        digitally_signed_struct: &rustls::DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            digitally_signed_struct,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
