"""
SentenceTransformer-based embedding precomputer.

Supports models like:
- Qwen3-Embedding
- all-MiniLM-L6-v2
- multilingual-e5-large
"""

from typing import List, Optional
import numpy as np

from .base import EmbeddingPrecomputer


class SentenceTransformerPrecomputer(EmbeddingPrecomputer):
    """
    Precomputer using SentenceTransformer models.

    Supports dimension truncation for models that support it (e.g., Qwen3-Embedding).

    Args:
        model_path: Path or name of SentenceTransformer model
        embed_dim: Target embedding dimension (uses truncate_dim if supported)
        device: Device to run on (e.g., "cuda:0", "cpu")
        model_name: Name for identification (defaults to last part of path)
    """

    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        device: str = "cuda:0",
        model_name: Optional[str] = None,
    ):
        self.model_path = model_path
        self._embed_dim = embed_dim
        self.device = device
        self._model_name = model_name or model_path.split("/")[-1]

        # Lazy load model
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading SentenceTransformer model: {self.model_path}")
            print(f"Target embedding dimension: {self._embed_dim}")

            # Try with truncate_dim first (for models that support it)
            try:
                self._model = SentenceTransformer(
                    self.model_path,
                    truncate_dim=self._embed_dim,
                    device=self.device,
                )
            except TypeError:
                # Fallback for models that don't support truncate_dim
                self._model = SentenceTransformer(
                    self.model_path,
                    device=self.device,
                )
                print(f"Warning: Model doesn't support truncate_dim, will use full dimension")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def compute(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Compute embeddings for texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array of shape (N, embed_dim)
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Truncate if needed (for models that don't support truncate_dim)
        if embeddings.shape[1] > self._embed_dim:
            embeddings = embeddings[:, :self._embed_dim]

        return embeddings


def get_precomputer(
    model_type: str,
    model_path: str,
    embed_dim: int = 1024,
    device: str = "cuda:0",
) -> EmbeddingPrecomputer:
    """
    Factory function to get appropriate precomputer.

    Args:
        model_type: Type of model ("qwen3", "sentence-transformer", etc.)
        model_path: Path to model
        embed_dim: Target embedding dimension
        device: Device to use

    Returns:
        EmbeddingPrecomputer instance
    """
    model_type = model_type.lower()

    if model_type in ("qwen3", "qwen", "sentence-transformer", "st"):
        return SentenceTransformerPrecomputer(
            model_path=model_path,
            embed_dim=embed_dim,
            device=device,
            model_name=f"qwen3-embedding" if "qwen" in model_type else None,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
