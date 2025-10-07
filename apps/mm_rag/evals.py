import argparse, json
from pathlib import Path
from retriever import retrieve

def load_gold(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def hit_in_hits(hits, gold_phrase, gold_page=None, k=5):
    for h in hits[:k]:
        txt = h["meta"].get("text", "").lower()
        pg = h["meta"].get("page")
        if gold_phrase.lower() in txt:
            if gold_page is None or pg == gold_page:
                return True
    return False

def reciprocal_rank(hits, gold_phrase, gold_page=None, k=10):
    for i, h in enumerate(hits[:k], start=1):
        txt = h["meta"].get("text", "").lower()
        pg = h["meta"].get("page")
        if gold_phrase.lower() in txt:
            if gold_page is None or pg == gold_page:
                return 1.0 / i
    return 0.0

def run_eval(gold_path: Path, data_root: Path, k: int = 5):
    gold = load_gold(gold_path)
    raw_metrics, rerank_metrics = [], []

    for item in gold:
        q, gold_phrase, gold_page = item["question"], item["expected_phrase"], item.get("expected_page")

        res = retrieve(q, data_root, k_text=k, use_rerank=False)
        hits = res["text_hits"]
        raw_metrics.append({
            "acc1": hit_in_hits(hits, gold_phrase, gold_page, 1),
            "rec": hit_in_hits(hits, gold_phrase, gold_page, k),
            "mrr": reciprocal_rank(hits, gold_phrase, gold_page, k)
        })

        res_rr = retrieve(q, data_root, k_text=k, use_rerank=True)
        hits_rr = res_rr["text_hits"]
        rerank_metrics.append({
            "acc1": hit_in_hits(hits_rr, gold_phrase, gold_page, 1),
            "rec": hit_in_hits(hits_rr, gold_phrase, gold_page, k),
            "mrr": reciprocal_rank(hits_rr, gold_phrase, gold_page, k)
        })

    def avg(m_list, key): return sum(m[key] for m in m_list) / len(m_list)

    print(f"\n Evaluated on {len(gold)} queries")

    print("\nBaseline Retriever:")
    print(f"Accuracy@1: {avg(raw_metrics,'acc1'):.3f}, Recall@{k}: {avg(raw_metrics,'rec'):.3f}, MRR: {avg(raw_metrics,'mrr'):.3f}")

    print("\nRetriever + Reranker:")
    print(f"Accuracy@1: {avg(rerank_metrics,'acc1'):.3f}, Recall@{k}: {avg(rerank_metrics,'rec'):.3f}, MRR: {avg(rerank_metrics,'mrr'):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=str, help="Path to gold_eval.jsonl")
    ap.add_argument("--data_root", required=True, type=str, help="Folder with FAISS indexes")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    run_eval(Path(args.gold), Path(args.data_root), k=args.k)
