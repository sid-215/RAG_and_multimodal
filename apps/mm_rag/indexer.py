# apps/mm_rag/indexer.py
from pathlib import Path
import faiss
import numpy as np

def build_faiss_index(vectors: np.ndarray, metric: str = "cosine") -> faiss.Index:
    if vectors.size == 0:
        return faiss.IndexFlatIP(1)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def save_faiss(index: faiss.Index, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))

def load_faiss(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
