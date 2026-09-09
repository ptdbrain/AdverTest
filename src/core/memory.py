"""Low-overhead process memory trimming utilities for constrained environments."""

from __future__ import annotations

import gc

try:
    import ctypes

    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except Exception:
    _malloc_trim = None


def trim_memory() -> None:
    """Collect Python garbage and release unused glibc heap arenas to the OS."""
    gc.collect()
    if _malloc_trim is not None:
        try:
            _malloc_trim(0)
        except Exception:
            pass
