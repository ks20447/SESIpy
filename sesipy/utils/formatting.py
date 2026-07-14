import os
from dataclasses import dataclass
from contextlib import contextmanager


@contextmanager
def suppress_c_output():
    """Suppress C/C++ printf output by redirecting file descriptors."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


@dataclass(frozen=True)
class Symbols:
    DEG = "°"

    PLUS_MINUS = "±"
    MULTIPLY = "×"
    DIVIDE = "÷"
    APPROX = "≈"
    NOT_EQUAL = "≠"
    LESS_EQUAL = "≤"
    GREATER_EQUAL = "≥"
    INFINITY = "∞"

    SUP_0 = "⁰"
    SUP_1 = "¹"
    SUP_2 = "²"
    SUP_3 = "³"
    SUP_4 = "⁴"
    SUP_5 = "⁵"
    SUP_6 = "⁶"
    SUP_7 = "⁷"
    SUP_8 = "⁸"
    SUP_9 = "⁹"

    SUB_0 = "₀"
    SUB_1 = "₁"
    SUB_2 = "₂"
    SUB_3 = "₃"
    SUB_4 = "₄"
    SUB_5 = "₅"
    SUB_6 = "₆"
    SUB_7 = "₇"
    SUB_8 = "₈"
    SUB_9 = "₉"

    ALPHA = "α"
    BETA = "β"
    GAMMA = "γ"
    DELTA = "δ"
    EPSILON = "ε"
    THETA = "θ"
    LAMBDA = "λ"
    MU = "μ"
    PI = "π"
    SIGMA = "σ"
    PHI = "φ"
    OMEGA = "ω"

    NABLA = "∇"
    PARTIAL = "∂"

    OHM = "Ω"

    M = "m"
    M2 = "m²"
    M3 = "m³"

    CM = "cm"
    MM = "mm"
    KM = "km"

    MPS = "m/s"
    MPS2 = "m/s²"

    RAD = "rad"
    RADPS = "rad/s"
    RADPS2 = "rad/s²"

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

    W = "W"
    MW = "mW"
    DB = "dB"
    DBM = "dBm"
    DBI = "dBi"
    WM2 = ""

    V = "V"
    MV = "mV"
    A = "A"
    MA = "mA"

    F = "F"
    PF = "pF"
    NF = "nF"
    UF = "μF"

    H = "H"
    NH = "nH"
    UH = "μH"

    TESLA = "T"
    WEBER = "Wb"

    ETA = "η"

    _SUBSCRIPT_MAP = str.maketrans(
        "0123456789+-=()",
        "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎",
    )

    _SUPERSCRIPT_MAP = str.maketrans(
        "0123456789+-=()",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾",
    )

    @classmethod
    def sub(cls, value) -> str:
        return str(value).translate(cls._SUBSCRIPT_MAP)

    @classmethod
    def sup(cls, value) -> str:
        return str(value).translate(cls._SUPERSCRIPT_MAP)