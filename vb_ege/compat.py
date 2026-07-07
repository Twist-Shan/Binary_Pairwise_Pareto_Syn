"""Compatibility shims for optional binary dependencies."""

from __future__ import annotations

import contextlib
import io


def import_pandas_quietly():
    """Import pandas while suppressing noisy optional-dependency ABI traces."""

    with contextlib.redirect_stderr(io.StringIO()):
        import pandas as pd

    return pd
