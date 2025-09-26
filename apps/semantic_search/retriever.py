"""
Retriever functions: dense, sparse, hybrid.
"""
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------
# Dense Retriever
# -----------------------------
def dense_retrieve(query, index, embed_model: SentenceTransformer, docs, top_k=5):
    query_vec = embed_model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(query_vec, top_k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        d = docs[i].copy()
        d["score"] = float(score)
        results.append(d)
    return results

# -----------------------------
# Sparse Retriever
# -----------------------------
def sparse_retrieve(query, bm25: BM25Okapi, docs, top_k=5):
    scores = bm25.get_scores(query.split())
    idxs = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in idxs:
        d = docs[i].copy()
        d["score"] = float(scores[i])
        results.append(d)
    return results

# -----------------------------
# Hybrid Retriever
# -----------------------------
def hybrid_retrieve(query, index, embed_model, bm25, docs, alpha=0.8, top_k=5):
    # Dense scores
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    d_scores, d_idxs = index.search(q_vec, len(docs))
    dense_scores = {i: float(s) for i, s in zip(d_idxs[0], d_scores[0])}

    # Sparse scores
    s_scores = bm25.get_scores(query.split())
    sparse_scores = {i: float(s) for i, s in enumerate(s_scores)}

    # Normalize & fuse
    fused = {}
    for i in range(len(docs)):
        ds = dense_scores.get(i, 0.0)
        ss = sparse_scores.get(i, 0.0)
        fused[i] = alpha * ds + (1 - alpha) * ss

    # Rank
    sorted_idxs = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for i, score in sorted_idxs:
        d = docs[i].copy()
        d["score"] = float(score)
        results.append(d)
    return results
