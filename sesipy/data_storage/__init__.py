"""Data storage module for Sesipy."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database, DatabaseAoA, DatabasePS, decode, encode

__all__ = ["Database", "DatabasePS", "DatabaseAoA", "encode", "decode"]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".database", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
