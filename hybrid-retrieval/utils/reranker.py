"""
Polished version of the Cohere cross-encoder rerank built in 5-rerank.py.
6-evaluate.py imports from here so the eval script can focus on the metric.
"""

import os
import time
import cohere
import pandas as pd
from cohere.errors import TooManyRequestsError
from dotenv import load_dotenv
from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever

load_dotenv()
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"
MIN_REQUEST_INTERVAL_SECONDS = 6.5
RATE_LIMIT_RETRY_SECONDS = 62
MAX_RATE_LIMIT_RETRIES = 2
_last_request_at = 0.0


def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    global _last_request_at

    documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

        _last_request_at = time.monotonic()
        try:
            response = co.rerank(
                model=RERANK_MODEL, query=query, documents=documents, top_n=k
            )
            break
        except TooManyRequestsError:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            time.sleep(RATE_LIMIT_RETRY_SECONDS)

    return [(candidate_ids[r.index], r.relevance_score) for r in response.results]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_cohere(query, candidate_ids, corpus_by_id, k=k)
