"""Native MACaQ kernels for oMLX (W4A16 decode)."""

from .fast import has_native, import_error, is_native_available, qmv

__all__ = ["has_native", "import_error", "is_native_available", "qmv"]
