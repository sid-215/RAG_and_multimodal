"""
Main CLI for semantic search.
"""
import sys
import os
import argparse
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from retriever import dense_retrieve, sparse_retrieve, hybrid_retrieve
from reranker import rerank
from normalize import normalize_query
from datetime import datetime

# -----------------------------
# Helper: date filtering
# -----------------------------
def filter_by_date(results, start_date=None, end_date=None):
    if not start_date or not end_date:
        return results
    
    # The below code block is representational only and it works only if we provide ""published_date" in "YYYY-MM-DD" format in the corpus
    # We do not have published_date in that format in our current corpus, so this function is commented out for now.
    # This is written just to illustrate how one might implement date filtering if the data supported it.
    
    """
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    filtered = []
    for r in results:
        pub = r.get("published_date")
        if pub:
            try:
                d = datetime.fromisoformat(pub).date()
                if start <= d <= end:
                    filtered.append(r)
            except:
                pass
    return filtered
    """
    return results  # No filtering applied due to lack of date data

# -----------------------------
# Load data & indexes
# -----------------------------
def load_corpus(corpus_file):
    docs = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--corpus", type=str, default="data/semantic_search/corpus.jsonl")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--filter_dates", action="store_true")
    args = parser.parse_args()
    

    # Normalize query
    norm_query, start_date, end_date = normalize_query(args.query)
    print("Normalized Query:", norm_query)

    # Load docs
    docs = load_corpus(args.corpus)

    # Sparse model
    bm25 = BM25Okapi([d["title"].split() + d["abstract"].split() for d in docs])

    # Dense model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embed_model.encode([d["title"] + " " + d["abstract"] for d in docs], normalize_embeddings=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Retrieve
    if args.mode == "dense":
        results = dense_retrieve(norm_query, index, embed_model, docs, top_k=args.top_k)
    elif args.mode == "sparse":
        results = sparse_retrieve(norm_query, bm25, docs, top_k=args.top_k)
    else:
        results = hybrid_retrieve(norm_query, index, embed_model, bm25, docs, top_k=args.top_k)

    # Optional date filtering
    if args.filter_dates and start_date and end_date:
        results = filter_by_date(results, start_date, end_date)

    # Optional reranking
    if args.rerank:
        results = rerank(norm_query, results, top_k=args.top_k)

    # Show
    for r in results:
        print("=" * 60)
        print("ID:", r.get("paper_id"))
        print("Title:", r.get("title"))
        print("Authors:", ", ".join(r.get("authors", [])))
        #print("Date:", r.get("published_date"))
        print("Abstract:", r.get("abstract")[:300], "...")
        print("Score:", r.get("score", r.get("rerank_score")))

if __name__ == "__main__":
    main()
