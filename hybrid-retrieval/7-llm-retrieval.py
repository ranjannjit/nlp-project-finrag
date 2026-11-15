"""Answer a question using hybrid retrieval and a low-cost GPT model."""

import argparse
import os

from utils.qa import DEFAULT_CHAT_MODEL, answer_question
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answer a question using retrieved FiQA documents."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to answer. If omitted, the script prompts for one.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hybrid search results to use as context (default: 5).",
    )
    args = parser.parse_args()

    query = " ".join(args.question).strip() or input("Enter your question: ").strip()
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    corpus_by_id = load_corpus().set_index("_id")

    model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    print(f"Answering with {model} using retrieved sources...\n")
    answer, sources = answer_question(query, bm25, dense, corpus_by_id, args.top_k)
    print(answer)

    if sources:
        print("\nSources:")
        for doc_id, text, score in sources:
            print(f"\n[{doc_id}] (RRF score: {score:.4f})")
            print(text[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
