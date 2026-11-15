"""
Polished versions of the BM25 and dense retrievers built in 2-bm25.py and
3-embed.py. The later files import from here so each one can focus on the new
idea it introduces (fusion, reranking, evaluation) instead of re-loading state.

This is the same pattern Dave uses in agents/agentic-search: numbered files
teach the construction, then the cleaned-up version graduates to a helper.
"""

import hashlib
import os
import sqlite3
from contextlib import closing
from pathlib import Path
import bm25s
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
BM25_DIR = ROOT / "indexes" / "bm25"
DENSE_DIR = ROOT / "indexes" / "dense"
DENSE_QUERY_CACHE = DENSE_DIR / "query_embeddings.sqlite3"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self) -> None:
        corpus = load_corpus()
        self._doc_ids = corpus["_id"].tolist()
        raw = np.load(DENSE_DIR / "embeddings.npy")
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _embed_query(self, query: str) -> np.ndarray:
        cache_key = hashlib.sha256(
            f"{EMBEDDING_MODEL}\0{query}".encode("utf-8")
        ).hexdigest()

        with closing(sqlite3.connect(DENSE_QUERY_CACHE)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """
            )
            cached = connection.execute(
                "SELECT embedding FROM query_embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if cached is not None:
            vec = np.frombuffer(cached[0], dtype=np.float32).copy()
            return vec / np.linalg.norm(vec)

        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        normalized_vec = vec / np.linalg.norm(vec)

        with closing(sqlite3.connect(DENSE_QUERY_CACHE)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO query_embeddings (cache_key, embedding)
                    VALUES (?, ?)
                    """,
                    (cache_key, normalized_vec.tobytes()),
                )

        return normalized_vec

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
