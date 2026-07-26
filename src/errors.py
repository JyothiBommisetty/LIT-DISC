# LIT-DISC Error Taxonomy
# Reference: Architecture Specification §12


class LITDISCError(Exception):
    """Base exception for all LIT-DISC errors."""


class InputError(LITDISCError):
    """File not found, encoding error, invalid format."""


class ParseError(LITDISCError):
    """Tokenization, sentence splitting, hierarchy detection failure."""


class FeatureError(LITDISCError):
    """Feature extraction failure; missing or invalid features."""


class NumericalError(LITDISCError):
    """Numerical instability, divergence, out-of-bounds values."""


class MemoryError(LITDISCError):
    """Memory capacity exceeded or store corruption."""


class ConfigError(LITDISCError):
    """Invalid configuration parameter or missing required field."""


class PipelineError(LITDISCError):
    """Pipeline orchestration failure or timeout."""
