"""Make the parent directory and shared module importable so tests can use bare imports."""

from __future__ import annotations

import pathlib
import sys

# Add yor/ parent so ``from yor_client import ...`` works
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
# Add examples/ so ``from shared import ...`` works
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
