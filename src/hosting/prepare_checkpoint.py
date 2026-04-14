"""Assemble a local OpenPI-compatible checkpoint from upstream sources.

This command downloads the authoritative upstream weights from Hugging Face
(`lerobot/pi05_base`) and the normalization statistics from the original
OpenPI checkpoint assets on GCS, then assembles a local checkpoint directory
that the hosting runtime can load directly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from openpi.shared import download as openpi_download

DEFAULT_LEROBOT_MODEL_ID = "lerobot/pi05_base"
DEFAULT_OPENPI_ASSETS_URI = "gs://openpi-assets/checkpoints/pi05_base/assets"
DEFAULT_REQUIRED_ASSET_ID = "trossen"
_REQUIRED_CHECKPOINT_FILENAMES = ("config.json", "model.safetensors")


def _make_huggingface_cache_directory_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def get_default_output_dir() -> Path:
    return openpi_download.get_cache_dir() / "pi05_base_openpi"


def _assert_prepared_checkpoint_directory_is_complete(output_dir: Path) -> None:
    missing_paths = [
        str(output_dir / checkpoint_filename)
        for checkpoint_filename in _REQUIRED_CHECKPOINT_FILENAMES
        if not (output_dir / checkpoint_filename).exists()
    ]
    required_norm_stats_path = output_dir / "assets" / DEFAULT_REQUIRED_ASSET_ID / "norm_stats.json"
    if not required_norm_stats_path.exists():
        missing_paths.append(str(required_norm_stats_path))

    if missing_paths:
        missing_paths_string = ", ".join(missing_paths)
        raise FileNotFoundError(
            "Prepared checkpoint directory is incomplete. "
            f"Missing required paths: {missing_paths_string}"
        )


def prepare_openpi_compatible_checkpoint(
    *,
    model_id: str = DEFAULT_LEROBOT_MODEL_ID,
    openpi_assets_uri: str = DEFAULT_OPENPI_ASSETS_URI,
    output_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    resolved_output_dir = (output_dir or get_default_output_dir()).resolve()
    if resolved_output_dir.exists() and not force_download:
        _assert_prepared_checkpoint_directory_is_complete(resolved_output_dir)
        print(f"Prepared checkpoint already exists at {resolved_output_dir}")
        return resolved_output_dir

    huggingface_cache_root = resolved_output_dir.parent / "huggingface"
    raw_model_download_dir = huggingface_cache_root / _make_huggingface_cache_directory_name(
        model_id
    )

    print(f"Downloading weights from Hugging Face model {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(raw_model_download_dir),
        allow_patterns=list(_REQUIRED_CHECKPOINT_FILENAMES),
    )

    print(f"Downloading normalization assets from {openpi_assets_uri}")
    local_openpi_assets_dir = openpi_download.maybe_download(
        openpi_assets_uri, force_download=force_download
    )

    temporary_output_dir = resolved_output_dir.with_suffix(".partial")
    if temporary_output_dir.exists():
        shutil.rmtree(temporary_output_dir)
    temporary_output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_filename in _REQUIRED_CHECKPOINT_FILENAMES:
        shutil.copy2(
            raw_model_download_dir / checkpoint_filename, temporary_output_dir / checkpoint_filename
        )

    shutil.copytree(local_openpi_assets_dir, temporary_output_dir / "assets")

    source_manifest = {
        "model_id": model_id,
        "openpi_assets_uri": openpi_assets_uri,
    }
    (temporary_output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2),
        encoding="utf-8",
    )

    _assert_prepared_checkpoint_directory_is_complete(temporary_output_dir)

    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)
    temporary_output_dir.rename(resolved_output_dir)
    print(f"Prepared checkpoint written to {resolved_output_dir}")
    return resolved_output_dir
