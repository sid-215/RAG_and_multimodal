import json
from tqdm import tqdm

from retriever import dense_retrieve, sparse_retrieve, hybrid_retrieve
from reranker import rerank
from normalize import normalize_query

# -----------------------------
# Load corpus + indexes
# -----------------------------
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

with open("data/semantic_search/corpus.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

# Dense
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("data/semantic_search/index/faiss.index")

# Sparse
bm25 = BM25Okapi([
    (d["title"] + " " + d["abstract"] + " " + " ".join(d.get("keywords", []))).split()
    for d in docs
])

# Map corpus to ID → doc
doc_map = {d["paper_id"]: d for d in docs}


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(eval_file="rag_eval_dataset.jsonl", retriever="hybrid", top_k=5, use_rerank=True):
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f if line.strip()]

    total = len(eval_data)
    correct_at1 = 0
    reciprocal_ranks = []
    recall_hits = 0

    candidate_k = max(50, top_k)  

    for ex in tqdm(eval_data, desc="Evaluating"):
        query = ex["question"]
        gold_source = ex["source"].replace(".pdf", "")  # strip extension

        # Normalize query
        q_norm, _, _ = normalize_query(query)
        if not q_norm:
            q_norm = query

        # Retrieve
        if retriever == "dense":
            results = dense_retrieve(q_norm, index, embed_model, docs, top_k=candidate_k)
        elif retriever == "sparse":
            results = sparse_retrieve(q_norm, bm25, docs, top_k=candidate_k)
        else:
            results = hybrid_retrieve(q_norm, index, embed_model, bm25, docs, top_k=candidate_k)

        # Rerank if enabled
        if use_rerank:
            results = rerank(q_norm, results, top_k=candidate_k)

        # Top-k subset
        retrieved_ids_topk = [r["paper_id"] for r in results[:top_k]]
        retrieved_ids_all = [r["paper_id"] for r in results]

        # Accuracy@1/Precision@1
        if retrieved_ids_topk and retrieved_ids_topk[0] == gold_source:
            correct_at1 += 1

        # Hitrate@k
        if gold_source in retrieved_ids_topk:
            recall_hits += 1

        # MRR (on the full reranked list)
        if gold_source in retrieved_ids_all:
            rank = retrieved_ids_all.index(gold_source) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    # Final metrics
    accuracy_at1 = correct_at1 / total if total > 0 else 0.0
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
    recall_at_k = recall_hits / total if total > 0 else 0.0

    print(f"Retriever: {retriever}, Rerank: {use_rerank}, Top-k: {top_k}")
    print(f"Accuracy@1: {accuracy_at1:.2%} ({correct_at1}/{total})")
    print(f"Hit Rate@{top_k}: {recall_at_k:.2%} ({recall_hits}/{total})")
    print(f"MRR: {mrr:.3f}")


# -----------------------------
# Run eval
# -----------------------------
if __name__ == "__main__":
    evaluate(eval_file="data/semantic_search/rag_eval_dataset.jsonl",
             retriever="hybrid", top_k=5, use_rerank=True)
