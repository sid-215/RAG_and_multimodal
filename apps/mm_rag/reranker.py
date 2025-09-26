from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[Dict], top_k: int = 5) -> List[Dict]:
        if not hits:
            return []
        pairs = [(query, h["meta"]["text"]) for h in hits]
        scores = self.model.predict(pairs)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        sorted_hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return sorted_hits[:top_k]
