import argparse
from pathlib import Path
from retriever import retrieve

def format_response(query: str, res):
    best = res["text_hits"][0] if res["text_hits"] else {}
    previews = [{"page": ih["meta"]["page"], "path": ih["meta"]["path"], "score": ih["score"]}
                for ih in res["image_hits"][:3]]
    return {
        "query": query,
        "normalized_query": res["normalized_query"],
        "answer": best.get("meta", {}).get("text", "No good match"),
        "citation": best.get("meta"),
        "image_previews": previews
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--data_root", default="data/mm_rag")
    args = ap.parse_args()

    res = retrieve(args.q, Path(args.data_root))
    out = format_response(args.q, res)

    print("\nQ:", out["query"])
    print("Normalized Q:", out["normalized_query"])
    print("A:", out["answer"])
    if out["citation"]:
        print("Citation:", out["citation"])
    if out["image_previews"]:
        print("Image previews:", out["image_previews"])
