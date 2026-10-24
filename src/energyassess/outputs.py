"""Writing results as strict JSON."""
from __future__ import annotations

import json
import math

import numpy as np


def clean(o):
    """Plain Python types, with NaN and infinity as None, so the JSON is valid for any parser."""
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if o is None or isinstance(o, str):
        return o
    return str(o)


def to_json(obj, indent: int | None = None, compact: bool = False) -> str:
    seps = (",", ":") if compact else None
    return json.dumps(clean(obj), indent=indent, separators=seps, allow_nan=False)
