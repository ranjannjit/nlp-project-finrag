#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

if ! command -v poetry >/dev/null 2>&1; then
    printf 'Error: Poetry is required but was not found on PATH.\n' >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
    printf 'Error: %s is missing.\n' "${PROJECT_DIR}/.env" >&2
    printf 'Copy .env.example to .env and set OPENAI_API_KEY.\n' >&2
    exit 1
fi

cd "${PROJECT_DIR}"
poetry install --no-interaction

if ! poetry run python -c '
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key or api_key.lower() in {
    "your-openai-api-key",
    "replace-me",
}:
    print("OPENAI_API_KEY is missing or still a placeholder.", file=sys.stderr)
    sys.exit(1)
'; then
    printf 'Error: configure OPENAI_API_KEY in %s/.env.\n' "${PROJECT_DIR}" >&2
    exit 1
fi

if [[ ! -f data/fiqa/corpus.parquet \
   || ! -f data/fiqa/queries.parquet \
   || ! -f data/fiqa/qrels.parquet ]]; then
    printf '\n== Download FiQA dataset ==\n'
    poetry run python data/download_fiqa.py
else
    printf 'Using existing FiQA dataset cache.\n'
fi

if [[ ! -f indexes/bm25/doc_ids.txt \
   || ! -f indexes/bm25/data.csc.index.npy ]]; then
    printf '\n== Build BM25 index ==\n'
    poetry run python 2-bm25.py
else
    printf 'Using existing BM25 index.\n'
fi

if [[ ! -f indexes/dense/embeddings.npy ]]; then
    printf '\n== Build dense embedding index ==\n'
    poetry run python 3-embed.py
else
    printf 'Using existing dense embedding cache; no corpus re-embedding needed.\n'
fi

if [[ "${RUN_EVALUATION:-0}" == "1" ]]; then
    if ! poetry run python -c '
import os
import sys
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COHERE_API_KEY", "").strip()
if not api_key or api_key.lower() in {
    "your-cohere-api-key",
    "replace-me",
}:
    print("COHERE_API_KEY is missing or still a placeholder.", file=sys.stderr)
    sys.exit(1)
'; then
        printf 'Error: RUN_EVALUATION=1 requires COHERE_API_KEY in %s/.env.\n' "${PROJECT_DIR}" >&2
        exit 1
    fi

    printf '\n== Evaluate retrieval methods (uses OpenAI and Cohere API calls) ==\n'
    poetry run python 6-evaluate.py
fi

printf '\n== Launch FiQA Streamlit app ==\n'
poetry run streamlit run 8-streamlit-app.py "$@"
