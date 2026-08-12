"""Load concept prompt sets from disk.

Activation extraction lives in `taps.py` (`TapForward`) -- the single, truncated-forward
listener the facade uses. This module is now just the tiny concept-file loader.
"""
from __future__ import annotations

import json


def load_concept(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
