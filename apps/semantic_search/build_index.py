import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse

def build_index(corpus_file, index_dir="data/semantic_search/index", model_name="all-MiniLM-L6-v2"):
    # Load dataset
    docs = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    texts = [doc["full_text"] for doc in docs]
    ids = [doc["paper_id"] for doc in docs]

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # simple L2 index
    index.add(embeddings)

    # Save index + metadata
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{index_dir}/faiss.index")
    with open(f"{index_dir}/metadata.jsonl", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Built FAISS index with {len(docs)} documents. Saved to {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/semantic_search/corpus.jsonl")
    parser.add_argument("--index_dir", type=str, default="data/semantic_search/index")
    args = parser.parse_args()
    build_index(args.corpus, args.index_dir)