"""openpi-flash CLI: unified entrypoint for serve and test commands.

Usage:
    uv run python main.py prepare-checkpoint
    uv run python main.py prepare-planner-checkpoint
    uv run python main.py serve [--config config.json]
    uv run python main.py test ws <url>
    uv run python main.py test quic <host> [--quic-port 5555] [--ws-port 8000]
    uv run python main.py test modal-tunnel
    uv run python main.py test modal-quic
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import typer

app = typer.Typer(
    name="openpi-flash",
    help="Real-time inference engine for openpi. Serves policy models over QUIC and WebSocket.",
    no_args_is_help=True,
)
test_app = typer.Typer(help="Run smoke tests against a running server.", no_args_is_help=True)
app.add_typer(test_app, name="test")


@app.command(name="prepare-checkpoint")
def prepare_checkpoint(
    model_id: Annotated[
        str,
        typer.Option(help="Hugging Face model ID for the upstream LeRobot checkpoint."),
    ] = "lerobot/pi05_base",
    openpi_assets_uri: Annotated[
        str,
        typer.Option(help="OpenPI assets URI containing normalization statistics."),
    ] = "gs://openpi-assets/checkpoints/pi05_base/assets",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Local directory where the assembled checkpoint should be written."),
    ] = None,
    force_download: Annotated[
        bool,
        typer.Option(help="Re-download upstream files and rebuild the prepared checkpoint."),
    ] = False,
) -> None:
    """Prepare a local OpenPI-compatible checkpoint from upstream sources."""
    from hosting.prepare_checkpoint import prepare_openpi_compatible_checkpoint

    prepare_openpi_compatible_checkpoint(
        model_id=model_id,
        openpi_assets_uri=openpi_assets_uri,
        output_dir=output_dir,
        force_download=force_download,
    )


@app.command(name="prepare-planner-checkpoint")
def prepare_planner_checkpoint(
    hf_repo: Annotated[
        str,
        typer.Option(help="Hugging Face repo id that holds the planner checkpoint tar."),
    ] = "Hebbian-Robotics/pi05_subtask",
    tar_path_in_repo: Annotated[
        str,
        typer.Option(help="Path inside the HF repo to the Orbax checkpoint tar."),
    ] = "jax/pi05_subtask.tar",
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Local directory where the extracted checkpoint should be written."),
    ] = None,
    force_download: Annotated[
        bool,
        typer.Option(
            help="Re-download and re-extract even if the output directory already exists."
        ),
    ] = False,
) -> None:
    """Download + extract a JAX subtask planner checkpoint from Hugging Face."""
    from hosting.prepare_planner_checkpoint import prepare_openpi_compatible_planner_checkpoint

    prepare_openpi_compatible_planner_checkpoint(
        hf_repo=hf_repo,
        tar_path_in_repo=tar_path_in_repo,
        output_dir=output_dir,
        force_download=force_download,
    )


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

    # Limit JAX GPU memory allocation when the JAX planner is loaded, so it
    # doesn't greedily pre-allocate the whole GPU and leave nothing for the
    # PyTorch action slot in combined mode. Must be set before any JAX import
    # (triggered transitively by openpi). Harmless when JAX isn't used.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

    from hosting.serve import main as serve_main

    serve_main()


InferenceModeChoice = Literal["default", "action_only", "subtask_only"]

_MODE_OPTION_HELP = (
    "Inference mode sent in the observation. Lets benchmarks target one phase "
    "of a combined-mode server: 'action_only' skips the planner, 'subtask_only' "
    "skips the action policy. Omit for the server default."
)


@test_app.command(name="ws")
def test_websocket(
    url: Annotated[str, typer.Argument(help="WebSocket URL (ws://host:port or wss://host).")],
    mode: Annotated[InferenceModeChoice | None, typer.Option(help=_MODE_OPTION_HELP)] = None,
) -> None:
    """Smoke test against a WebSocket server (EC2, Docker, or Modal ASGI)."""
    from tests.test_ws import run

    run(url, mode=mode)


@test_app.command(name="quic")
def test_quic(
    host: Annotated[str, typer.Argument(help="Server host (IP or hostname).")],
    quic_port: Annotated[int, typer.Option(help="Server QUIC/UDP port.")] = 5555,
    ws_port: Annotated[
        int, typer.Option(help="Server WebSocket/TCP port (for health check).")
    ] = 8000,
    mode: Annotated[InferenceModeChoice | None, typer.Option(help=_MODE_OPTION_HELP)] = None,
) -> None:
    """Smoke test against a direct QUIC server (EC2 or Docker)."""
    from tests.test_quic import run

    run(host, quic_port=quic_port, ws_port=ws_port, mode=mode)


@test_app.command(name="modal-tunnel")
def test_modal_tunnel(
    mode: Annotated[InferenceModeChoice | None, typer.Option(help=_MODE_OPTION_HELP)] = None,
) -> None:
    """Smoke test against a Modal tunnel deployment."""
    from tests.test_tunnel import run

    run(mode=mode)


@test_app.command(name="modal-quic")
def test_modal_quic(
    mode: Annotated[InferenceModeChoice | None, typer.Option(help=_MODE_OPTION_HELP)] = None,
) -> None:
    """Smoke test against a Modal QUIC portal deployment."""
    from tests.test_modal_quic import run

    run(mode=mode)


if __name__ == "__main__":
    app()
