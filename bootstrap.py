from __future__ import annotations
import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def _prepend(p: Path):
    p = p.expanduser().resolve()
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

def configure(matching_models_root=None):
    # avl_data.py / avl_utils.py supplied with the release
    _prepend(REPO_ROOT / "external" / "AnyVisLoc")

    raw = matching_models_root or os.environ.get("MATCHING_MODELS_ROOT")
    if raw:
        p = Path(raw).expanduser().resolve()
        # avl_utils imports "Matching_Models...."; accept either its parent
        # or the Matching_Models directory itself.
        if p.name == "Matching_Models":
            _prepend(p.parent)
        elif (p / "Matching_Models").is_dir():
            _prepend(p)
        else:
            # Also support a user-provided tree whose directory itself is the
            # import root; useful for custom matcher installations.
            _prepend(p)
    return REPO_ROOT
