"""
Cross-encoder reranking for candidate documents.
"""
from sentence_transformers import CrossEncoder

def rerank(query, candidates, top_k=5, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    reranker = CrossEncoder(model_name)
    # Adding title and abstract as pair with query for re-ranking for now to demonstrate re-ranking
    # In actual use case, I will take full text, chunk it and then use the chunks for re-ranking
    pairs = [(query, c["title"] + " " + c["abstract"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
