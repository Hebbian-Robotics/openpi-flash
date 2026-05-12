"""Shared helpers for the foreact_eval package.

Currently imported by the nano-banana generator + visualizer. The ForeAct
generators (generate_foresight.py, generate_foresight_lerobot.py) run in a
separate conda env on the remote GPU box and don't import these helpers,
but nothing here prevents them from doing so later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple


def foresight_path(output_dir: Path, episode_index: int, frame_idx: int) -> Path:
    """Where a foresight generator writes its PNG for one frame.

    Single owner of the on-disk ``episode_{:06d}/frame_{:05d}.png`` layout.
    If we ever change it (e.g. to include model name in the path), this is
    the only place to touch.
    """
    return output_dir / f"episode_{episode_index:06d}" / f"frame_{frame_idx:05d}.png"


class SourceFrame(NamedTuple):
    """One input frame for foresight generation or visualization.

    Produced by ``iter_source_frames`` at the filesystem boundary so
    downstream code treats (episode_index, frame_idx) as one logical ID
    and doesn't re-parse filenames.
    """

    episode_index: int
    frame_idx: int
    actual_path: Path


def iter_source_frames(source_root: Path) -> list[SourceFrame]:
    """Every frame under ``source_root/episode_*/actual/frame_*.png``.

    Sorted by (episode_index, frame_idx) so callers render or generate in
    temporal order.
    """
    frames: list[SourceFrame] = []
    for episode_dir in sorted(source_root.glob("episode_*")):
        actual_dir = episode_dir / "actual"
        if not actual_dir.exists():
            continue
        episode_index = int(episode_dir.name.removeprefix("episode_"))
        for path in sorted(actual_dir.glob("frame_*.png")):
            frame_idx = int(path.stem.removeprefix("frame_"))
            frames.append(SourceFrame(episode_index, frame_idx, path))
    return frames


# Chain-mode manifest records one of these per frame. Typing as a Literal
# (not free-form str) means a typo in any write site fails at type-check
# instead of silent log-grep later.
ForesightStatus = Literal["cached", "generated", "failed", "refused"]
