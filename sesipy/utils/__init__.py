"""Utility functions and classes for Sesipy."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .array_factory import ArrayFactory
    from .formatting import Symbols, suppress_c_output
    from .normal_factory import NormalFactory

__all__ = [
    "Symbols",
    "suppress_c_output",
    "ArrayFactory",
    "NormalFactory",
]


_MODULES = {
    "Symbols": ".formatting",
    "suppress_c_output": ".formatting",
    "ArrayFactory": ".array_factory",
    "NormalFactory": ".normal_factory",
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__[:]
