"""
Precompute system for text embeddings.

Provides tools to precompute embeddings before training,
avoiding the need to load large language models during training.
"""

from .base import EmbeddingPrecomputer
from .sentence_transformer import SentenceTransformerPrecomputer

__all__ = [
    "EmbeddingPrecomputer",
    "SentenceTransformerPrecomputer",
]
