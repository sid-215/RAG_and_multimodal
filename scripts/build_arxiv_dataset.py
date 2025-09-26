"""
Build a dataset of academic papers from arXiv.
Takes most recent n papers from specified categories, downloads their PDFs,
extracts text, and saves metadata and content in JSONL format in semantic_search/corpus.jsonl.

Fields:
- paper_id
- title
- abstract
- authors
- keywords (extracted from abstract)
- full_text (from PDF)
"""

import arxiv
import json
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import argparse
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Categories to pull papers from
CATEGORIES = [
    "econ.EM"
]

kw_model = KeyBERT(model="all-MiniLM-L6-v2")  # lightweight keyword model

def pdf_to_text(pdf_path: Path, max_pages: int = 40) -> str:
    """Extract text from the first N pages of a PDF."""
    try:
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            return ""
        doc = fitz.open(pdf_path)
        pages_to_read = min(max_pages, len(doc))
        texts = [doc[i].get_text("text") for i in range(pages_to_read)]
        return "\n".join(texts).strip()
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
        return ""

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

def extract_keywords(text: str, top_n: int = 5):
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),   # allow unigrams & bigrams
            stop_words="english",           # filter common words
            use_mmr=True,                   # diversify keywords
            diversity=0.7,                  # reduce repetition
            top_n=top_n * 2                 # get more, we'll filter
        )
        # Deduplicate & clean
        seen = set()
        clean_keywords = []
        for kw, _ in keywords:
            kw = kw.lower().strip()
            if kw not in seen and kw not in ENGLISH_STOP_WORDS:
                seen.add(kw)
                clean_keywords.append(kw)
            if len(clean_keywords) >= top_n:
                break
        return clean_keywords
    except Exception:
        return []

def fetch_papers(max_total=20, out_file="data/semantic_search/corpus.jsonl"):
    raw_dir = Path("data/raw_pdfs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    Path("data/semantic_search").mkdir(parents=True, exist_ok=True)

    all_papers = []
    count_per_cat = max_total // len(CATEGORIES)

    for cat in CATEGORIES:
        search = arxiv.Search(
            query=f"cat:{cat}",
            max_results=count_per_cat,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        for result in tqdm(arxiv.Client().results(search), desc=f"{cat}"):
            pdf_path = raw_dir / f"{result.get_short_id()}.pdf"

            # Download PDF if not already present
            if not pdf_path.exists():
                try:
                    result.download_pdf(filename=pdf_path)
                except Exception as e:
                    print(f"Could not download {result.get_short_id()}: {e}")
                    continue

            # Extract text
            full_text = pdf_to_text(pdf_path)
            if not full_text:
                continue

            # Extract keywords from abstract
            abstract = result.summary.strip()
            keywords = extract_keywords(abstract)

            paper = {
                "paper_id": result.get_short_id(),
                "title": result.title.strip(),
                "abstract": abstract,
                "authors": [a.name for a in result.authors],
                "keywords": keywords,
                "full_text": full_text,
            }
            all_papers.append(paper)

    # Save JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for paper in all_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f" Saved {len(all_papers)} papers to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_total", type=int, default=40)
    parser.add_argument("--out", type=str, default="data/semantic_search/corpus.jsonl")
    args = parser.parse_args()
    fetch_papers(args.max_total, args.out)