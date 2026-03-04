"""
Base class for embedding precomputers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class EmbeddingPrecomputer(ABC):
    """
    Abstract base class for text embedding precomputers.

    Implementations should handle:
    - Loading the embedding model
    - Encoding text to fixed-dimension vectors
    - Batch processing for efficiency
    """

    @abstractmethod
    def compute(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array of shape (N, embed_dim)
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name for identification in filenames."""
        pass
