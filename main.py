"""openpi-flash CLI: unified entrypoint for serve and test commands.

Usage:
    uv run python main.py serve [--config config.json]
    uv run python main.py test ws <url>
    uv run python main.py test quic <host> [--quic-port 5555] [--ws-port 8000]
    uv run python main.py test modal-tunnel
    uv run python main.py test modal-quic
"""

import os
from typing import Annotated

import typer

app = typer.Typer(
    name="openpi-flash",
    help="Real-time inference engine for openpi. Serves policy models over QUIC and WebSocket.",
    no_args_is_help=True,
)
test_app = typer.Typer(help="Run smoke tests against a running server.", no_args_is_help=True)
app.add_typer(test_app, name="test")


@app.command()
def serve(
    config: Annotated[str | None, typer.Option(help="Path to config JSON file.")] = None,
) -> None:
    """Start the local inference server (WebSocket + QUIC).

    Uses --config if provided, otherwise falls back to INFERENCE_CONFIG_PATH
    env var, then config.json.
    """
    if config is not None:
        os.environ["INFERENCE_CONFIG_PATH"] = config
    elif "INFERENCE_CONFIG_PATH" not in os.environ:
        os.environ["INFERENCE_CONFIG_PATH"] = "config.json"

    from hosting.serve import main as serve_main

    serve_main()


@test_app.command(name="ws")
def test_websocket(
    url: Annotated[str, typer.Argument(help="WebSocket URL (ws://host:port or wss://host).")],
) -> None:
    """Smoke test against a WebSocket server (EC2, Docker, or Modal ASGI)."""
    from tests.test_ws import run

    run(url)


@test_app.command(name="quic")
def test_quic(
    host: Annotated[str, typer.Argument(help="Server host (IP or hostname).")],
    quic_port: Annotated[int, typer.Option(help="Server QUIC/UDP port.")] = 5555,
    ws_port: Annotated[
        int, typer.Option(help="Server WebSocket/TCP port (for health check).")
    ] = 8000,
) -> None:
    """Smoke test against a direct QUIC server (EC2 or Docker)."""
    from tests.test_quic import run

    run(host, quic_port=quic_port, ws_port=ws_port)


@test_app.command(name="modal-tunnel")
def test_modal_tunnel() -> None:
    """Smoke test against a Modal tunnel deployment."""
    from tests.test_tunnel import run

    run()


@test_app.command(name="modal-quic")
def test_modal_quic() -> None:
    """Smoke test against a Modal QUIC portal deployment."""
    from tests.test_modal_quic import run

    run()


if __name__ == "__main__":
    app()
