"""
src/models/__init__.py

GLiNEREncoder is NOT imported here eagerly to avoid requiring torch/gliner
at import time (e.g. during unit tests that only use the prototype classifiers).
Import it directly: from src.models.gliner_encoder import GLiNEREncoder
"""
from .point_prototype import PointPrototype
from .gaussian_prototype import GaussianPrototype

__all__ = ["PointPrototype", "GaussianPrototype"]


def get_encoder(*args, **kwargs):
    """Lazy factory for GLiNEREncoder — imports torch only when called."""
    from .gliner_encoder import GLiNEREncoder  # noqa: PLC0415
    return GLiNEREncoder(*args, **kwargs)
