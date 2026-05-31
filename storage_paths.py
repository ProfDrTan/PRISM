"""
PRISM — storage_paths.py
------------------------
Resolves runtime data directories based on deployment mode.

DEMO_MODE=true  → writes to data/demo_guests/ (safe for public HF Space)
DEMO_MODE=false → writes to runtime_data/ (local development)
IRIS_DATA_DIR   → override with a custom path

Author: Professor Dr. Teik Kheong Tan
"""

from pathlib import Path
import os


def resolve_data_dir(project_root: Path, demo_mode: bool) -> Path:
    """
    Return the active data directory for this runtime.

    Priority:
      1. PRISM_DATA_DIR environment variable (explicit override)
      2. demo_mode=True  → <project_root>/data/demo_guests
      3. demo_mode=False → <project_root>/runtime_data
    """
    env_override = os.environ.get("PRISM_DATA_DIR", "").strip()
    if env_override:
        return Path(env_override)

    if demo_mode:
        return project_root / "data" / "demo_guests"

    return project_root / "runtime_data"
