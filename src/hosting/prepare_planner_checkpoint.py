"""Download and unpack a JAX subtask planner checkpoint from Hugging Face.

Mirror of ``prepare_checkpoint`` but for the planner slot. The planner's
Orbax checkpoint ships as a single multi-GB ``.tar`` blob on Hugging Face
(e.g. ``Hebbian-Robotics/pi05_subtask/jax/pi05_subtask.tar``). Most tars wrap the
Orbax layout inside a ``<step>/`` directory — we strip that wrapper so the
prepared output directory is the exact directory that
``SubtaskGenerator`` expects as ``checkpoint_dir`` (i.e. one that contains
``params/_METADATA`` directly).
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from openpi.shared import download as openpi_download

DEFAULT_HF_REPO = "Hebbian-Robotics/pi05_subtask"
DEFAULT_TAR_PATH_IN_REPO = "jax/pi05_subtask.tar"

_PARAMS_METADATA_RELATIVE_PATH = Path("params") / "_METADATA"


def get_default_output_dir() -> Path:
    return openpi_download.get_cache_dir() / "pi05_subtask"


def _assert_planner_checkpoint_layout_is_complete(output_dir: Path) -> None:
    """Raise if the prepared directory doesn't look like an Orbax checkpoint."""
    required_path = output_dir / _PARAMS_METADATA_RELATIVE_PATH
    if not required_path.exists():
        raise FileNotFoundError(
            f"Prepared planner checkpoint is incomplete: {required_path} is missing. "
            "Expected an Orbax layout with params/_METADATA at the root."
        )


def _iter_tar_top_level_names(tar_path: Path) -> set[str]:
    """Return the set of top-level directory/file names inside a tar archive."""
    top_level_names: set[str] = set()
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            first_segment = Path(member.name).parts[0]
            if first_segment in {"", "."}:
                continue
            top_level_names.add(first_segment)
    return top_level_names


def _strip_single_top_level_dir(extracted_root: Path) -> Path:
    """If the extraction produced a single wrapper dir, promote its contents up.

    Many Orbax tars (including ``pi05_subtask.tar``) wrap the checkpoint
    inside a numeric step directory (e.g. ``99/``). We want the caller to
    be able to point ``SubtaskGenerator`` at the extraction root, not
    ``root/99/``, so we move the inner contents up one level when a
    single wrapper directory is detected.
    """
    children = [child for child in extracted_root.iterdir() if child.name != ".DS_Store"]
    if len(children) != 1 or not children[0].is_dir():
        return extracted_root

    wrapper = children[0]
    for item in wrapper.iterdir():
        item.rename(extracted_root / item.name)
    wrapper.rmdir()
    return extracted_root


def prepare_openpi_compatible_planner_checkpoint(
    *,
    hf_repo: str = DEFAULT_HF_REPO,
    tar_path_in_repo: str = DEFAULT_TAR_PATH_IN_REPO,
    output_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download + extract an HF-hosted JAX subtask planner checkpoint tar.

    Args:
        hf_repo: Hugging Face model repo holding the tar blob.
        tar_path_in_repo: Path inside the repo to the ``.tar`` file.
        output_dir: Where to write the prepared checkpoint. When omitted,
            defaults to ``$OPENPI_DATA_HOME/pi05_subtask``.
        force_download: Re-download and re-extract even if the output
            directory already looks complete.

    Returns:
        Absolute path to the prepared checkpoint directory (the value a
        caller should pass as ``planner.checkpoint_dir`` in the service
        config).
    """
    resolved_output_dir = (output_dir or get_default_output_dir()).resolve()

    if resolved_output_dir.exists() and not force_download:
        try:
            _assert_planner_checkpoint_layout_is_complete(resolved_output_dir)
        except FileNotFoundError:
            print(
                f"Output directory {resolved_output_dir} exists but is missing "
                f"params/_METADATA; re-extracting."
            )
        else:
            print(f"Prepared planner checkpoint already exists at {resolved_output_dir}")
            return resolved_output_dir

    print(f"Downloading {tar_path_in_repo} from Hugging Face repo {hf_repo}")
    local_tar_path = Path(
        hf_hub_download(
            repo_id=hf_repo,
            filename=tar_path_in_repo,
            force_download=force_download,
        )
    )
    print(
        f"Downloaded tar to {local_tar_path} ({local_tar_path.stat().st_size / (1024**3):.2f} GiB)"
    )

    temporary_output_dir = resolved_output_dir.with_suffix(".partial")
    if temporary_output_dir.exists():
        shutil.rmtree(temporary_output_dir)
    temporary_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting into {temporary_output_dir}")
    with tarfile.open(local_tar_path, "r:*") as archive:
        archive.extractall(temporary_output_dir)

    _strip_single_top_level_dir(temporary_output_dir)
    _assert_planner_checkpoint_layout_is_complete(temporary_output_dir)

    source_manifest = {
        "hf_repo": hf_repo,
        "tar_path_in_repo": tar_path_in_repo,
    }
    (temporary_output_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2),
        encoding="utf-8",
    )

    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)
    temporary_output_dir.rename(resolved_output_dir)
    print(f"Prepared planner checkpoint written to {resolved_output_dir}")
    return resolved_output_dir
