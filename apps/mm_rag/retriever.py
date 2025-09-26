from pathlib import Path
from typing import Dict, Any
import faiss
from io_utils import load_jsonl
from indexer import load_faiss
from embeddings import TextEmbedder, ImageEmbedder
from normalize import normalize_query
from reranker import Reranker

PREF_ORDER = {"table_row": 0, "image_kv": 1, "image_caption": 2, "image_ocr": 3, "text": 4, "table_summary": 9}

def _filter_and_rank_text_hits(hits):
    cleaned = []
    for h in hits:
        m = h["meta"]
        if m.get("modality") == "table_summary":
            continue
        if m.get("is_header", False):
            continue
        cleaned.append(h)
    return sorted(cleaned, key=lambda h: (PREF_ORDER.get(h["meta"].get("modality", "text"), 99), -h["score"]))

def retrieve(query: str, data_root: Path, k_text: int = 20, k_img: int = 6, use_rerank=True) -> Dict[str, Any]:
    norm_q = normalize_query(query)

    text_index = load_faiss(data_root / "index" / "text.faiss")
    text_meta = load_jsonl(data_root / "index" / "text_meta.jsonl")
    t_emb = TextEmbedder()
    qv = t_emb.encode([norm_q]); faiss.normalize_L2(qv)
    D_t, I_t = text_index.search(qv, k_text)
    text_hits = [{"score": float(s), "meta": text_meta[i]} for s, i in zip(D_t[0], I_t[0])]
    text_hits = _filter_and_rank_text_hits(text_hits)

    if use_rerank and text_hits:
        rr = Reranker()
        text_hits = rr.rerank(norm_q, text_hits, top_k=5)

    img_index = load_faiss(data_root / "index" / "image.faiss")
    img_meta = load_jsonl(data_root / "index" / "image_meta.jsonl")
    i_emb = ImageEmbedder()
    qimg = i_emb.encode_text_for_clip([norm_q])
    D_i, I_i = img_index.search(qimg, k_img)
    img_hits = [{"score": float(s), "meta": img_meta[i]} for s, i in zip(D_i[0], I_i[0])]

    return {"text_hits": text_hits, "image_hits": img_hits, "normalized_query": norm_q}
