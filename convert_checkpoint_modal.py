"""Convert a JAX checkpoint to PyTorch format on Modal.

Downloads the JAX checkpoint from GCS, converts it to PyTorch (safetensors),
and saves the result to the openpi-model-weights Modal Volume.

Usage:
    # Convert the default pi05_base checkpoint
    uv run modal run convert_checkpoint_modal.py

    # Convert a different checkpoint
    uv run modal run convert_checkpoint_modal.py \
        --checkpoint-dir gs://openpi-assets/checkpoints/pi05_droid \
        --config-name pi05_droid \
        --output-name pi05_droid_pytorch
"""

import modal

app = modal.App("openpi-convert-checkpoint")

convert_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "clang",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
    )
    .pip_install("uv")
    # Layer 1: Copy only dependency metadata (changes rarely).
    .add_local_file("../openpi/pyproject.toml", "/build/openpi/pyproject.toml", copy=True)
    .add_local_file("../openpi/uv.lock", "/build/openpi/uv.lock", copy=True)
    .add_local_file(
        "../openpi/packages/openpi-client/pyproject.toml",
        "/build/openpi/packages/openpi-client/pyproject.toml",
        copy=True,
    )
    # Layer 2: Install dependencies only (cached unless pyproject.toml/uv.lock change).
    .run_commands(
        "cd /build/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-install-workspace",
        "cd /build/openpi && uv pip install gsutil",
    )
    # Layer 3: Copy transformers_replace patch source and apply it.
    .add_local_dir(
        "../openpi/src/openpi/models_pytorch/transformers_replace",
        "/build/transformers_replace",
        copy=True,
    )
    .run_commands(
        'TRANSFORMERS_DIR=$(/build/openpi/.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname) && '
        "cp -r /build/transformers_replace/* $TRANSFORMERS_DIR/"
    )
    .env(
        {
            "OPENPI_DATA_HOME": "/model-cache",
            "PYTHONPATH": "/app/openpi-src:/app/openpi-client-src",
            "VIRTUAL_ENV": "/build/openpi/.venv",
            "PATH": "/build/openpi/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .add_local_dir("../openpi/src", "/app/openpi-src")
    .add_local_dir("../openpi/packages/openpi-client/src", "/app/openpi-client-src")
    .add_local_file(
        "../openpi/examples/convert_jax_model_to_pytorch.py",
        "/app/convert_jax_model_to_pytorch.py",
    )
)

model_weights_volume = modal.Volume.from_name("openpi-model-weights", create_if_missing=True)


@app.function(
    image=convert_image,
    gpu="L4",
    volumes={"/model-cache": model_weights_volume},
    timeout=3600,
)
def convert_checkpoint(
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_base",
    config_name: str = "pi05_aloha",
    output_name: str = "pi05_base_pytorch",
    precision: str = "bfloat16",
) -> None:
    import importlib.util
    import logging
    import pathlib
    import shutil
    import sys

    # Apply transformers patches before any model imports.
    # The build-time cp may be cached by Modal, so re-apply at runtime.
    import transformers

    transformers_dir = str(transformers.__path__[0])
    patch_source = "/app/openpi-src/openpi/models_pytorch/transformers_replace"
    shutil.copytree(patch_source, transformers_dir, dirs_exist_ok=True)

    # Delete stale .pyc bytecode caches so Python recompiles from the patched .py files.
    for pycache_dir in pathlib.Path(transformers_dir, "models").glob("*/__pycache__"):
        shutil.rmtree(pycache_dir)

    # Evict cached transformers submodules so Python reloads from the patched files.
    patched_modules = [key for key in sys.modules if key.startswith("transformers.models.")]
    for mod_name in patched_modules:
        del sys.modules[mod_name]

    import openpi.models.pi0_config
    import openpi.shared.download as download
    from openpi.training import config as _config

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Resolve the training config to get model config.
    train_config = _config.get_config(config_name)
    model_config = train_config.model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    # Download the JAX checkpoint from GCS.
    logger.info("Downloading JAX checkpoint from %s", checkpoint_dir)
    local_checkpoint_dir = download.maybe_download(checkpoint_dir)
    logger.info("Checkpoint downloaded to %s", local_checkpoint_dir)

    # Import the conversion function from the bundled script.
    spec = importlib.util.spec_from_file_location(
        "convert_jax_model_to_pytorch", "/app/convert_jax_model_to_pytorch.py"
    )
    assert spec is not None and spec.loader is not None
    convert_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert_module)
    convert_pi0_checkpoint = convert_module.convert_pi0_checkpoint

    # Run the conversion.
    output_path = pathlib.Path("/model-cache") / output_name
    logger.info("Converting checkpoint to PyTorch format at %s", output_path)
    convert_pi0_checkpoint(
        checkpoint_dir=str(local_checkpoint_dir),
        precision=precision,
        output_path=str(output_path),
        model_config=model_config,
    )

    # The conversion script copies assets from checkpoint_dir/../assets, which is
    # wrong for checkpoints downloaded via maybe_download. Copy assets from the
    # actual checkpoint directory into the output.
    assets_source = local_checkpoint_dir / "assets"
    assets_dest = output_path / "assets"
    if assets_source.exists() and not assets_dest.exists():
        logger.info("Copying assets from %s to %s", assets_source, assets_dest)
        shutil.copytree(assets_source, assets_dest)
    elif assets_dest.exists():
        logger.info("Assets already present at %s", assets_dest)
    else:
        logger.warning("No assets directory found at %s", assets_source)

    # Commit the volume so the data persists.
    model_weights_volume.commit()

    logger.info("Conversion complete. Output at %s", output_path)
    logger.info("Files: %s", list(output_path.iterdir()))
