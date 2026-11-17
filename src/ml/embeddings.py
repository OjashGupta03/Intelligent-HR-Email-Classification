"""
Executive AI Chief of Staff - Embedding Manager
Local sentence-embedding generation and similarity search using
sentence-transformers and FAISS.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.helpers import setup_logging, batch_iterator, timer_decorator

logger = setup_logging(__name__)


class EmbeddingManager:
    """
    Manages local sentence embeddings using ``sentence-transformers``.

    Provides encoding, similarity computation, FAISS-based nearest-neighbour
    search, and persistence of embedding matrices.

    Attributes:
        model_name: HuggingFace model identifier.
        model: Loaded ``SentenceTransformer`` instance.
        device: ``'cuda'`` or ``'cpu'``.
        embedding_dim: Dimensionality of the model's output vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialise the EmbeddingManager.

        Args:
            model_name: Name of the sentence-transformers model to load.
                        Defaults to ``all-MiniLM-L6-v2`` (384-dim, ~80 MB).
        """
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model '%s' on %s", model_name, self.device)

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded — dim=%d, device=%s",
            self.embedding_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    @timer_decorator
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense embeddings.

        Args:
            texts: Strings to encode.
            batch_size: Encoding batch size (tune for GPU memory).
            show_progress: Show a tqdm progress bar.
            normalize: L2-normalise embeddings (required for cosine via dot-product).

        Returns:
            ``np.ndarray`` of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            logger.warning("encode() called with empty text list")
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Sanitise inputs — model expects strings
        clean_texts = [str(t) if t else "" for t in texts]

        embeddings = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        logger.debug("Encoded %d texts → shape %s", len(texts), embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Cosine similarity in [-1, 1].
        """
        embs = self.encode([text1, text2], normalize=True)
        similarity = float(np.dot(embs[0], embs[1]))
        logger.debug("Similarity: %.4f", similarity)
        return similarity

    def find_similar(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        corpus_texts: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the most similar texts from a corpus using FAISS.

        Args:
            query: Query text.
            corpus_embeddings: Pre-computed corpus embeddings (normalised).
            corpus_texts: Corresponding corpus texts.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys ``text``, ``score``, ``index``.
        """
        if corpus_embeddings.shape[0] == 0:
            logger.warning("find_similar() called with empty corpus")
            return []

        top_k = min(top_k, corpus_embeddings.shape[0])

        index = self.build_faiss_index(corpus_embeddings)
        query_embedding = self.encode([query], normalize=True)
        scores, indices = self.search_faiss_index(index, query_embedding, top_k=top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for unfilled slots
            results.append(
                {
                    "text": corpus_texts[idx],
                    "score": float(score),
                    "index": int(idx),
                }
            )
        return results

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def build_faiss_index(self, embeddings: np.ndarray) -> "faiss.IndexFlatIP":
        """
        Build a FAISS inner-product (cosine) index from embeddings.

        If the embeddings are L2-normalised, inner-product equals cosine similarity.

        Args:
            embeddings: ``np.ndarray`` of shape ``(n, dim)`` (should be normalised).

        Returns:
            Populated ``faiss.IndexFlatIP``.
        """
        import faiss

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.debug("Built FAISS IndexFlatIP with %d vectors, dim=%d", index.ntotal, dim)
        return index

    def search_faiss_index(
        self,
        index: "faiss.IndexFlatIP",
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search a FAISS index.

        Args:
            index: Pre-built FAISS index.
            query_embedding: Query vector(s) of shape ``(n_queries, dim)``.
            top_k: Number of nearest neighbours to return.

        Returns:
            Tuple of ``(scores, indices)`` arrays, each of shape ``(n_queries, top_k)``.
        """
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, index.ntotal)
        scores, indices = index.search(query_embedding, top_k)
        return scores, indices

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """
        Save an embedding matrix to disk as a ``.npy`` file.

        Args:
            embeddings: Array to save.
            path: Destination file path (will be created / overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), embeddings)
        logger.info(
            "Saved embeddings %s → %s (%.2f MB)",
            embeddings.shape,
            path,
            embeddings.nbytes / (1024 * 1024),
        )

    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load an embedding matrix from a ``.npy`` file.

        Args:
            path: Source file path.

        Returns:
            Loaded ``np.ndarray``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        embeddings = np.load(str(path)).astype(np.float32)
        logger.info("Loaded embeddings %s from %s", embeddings.shape, path)
        return embeddings

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EmbeddingManager(model='{self.model_name}', "
            f"dim={self.embedding_dim}, device='{self.device}')"
        )
