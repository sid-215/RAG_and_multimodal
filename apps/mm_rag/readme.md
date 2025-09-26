# 📘 Challenge 2 — Multi-Modal Document RAG

##  Objective
Build a split-modal RAG pipeline that answers questions from a PDF containing **text, tables, and charts/images**.  
Sample PDF: *Portfolio Analysis Sample (JPMorgan)*.

## Why Split-Modality?

- I chose a split-modal pipeline instead of going “native multimodal” (e.g., BLIP-2, LLaVA, PaLI) for three reasons:

- Cost and Efficiency: Running full multimodal encoders on every page is computationally expensive. By splitting modalities, I can independently index text, table rows, and image-derived facts with lighter models.

- Transparency and Debuggability: With split-modal chunks, I can inspect whether the answer came from a table row, an OCR’d chart, or a paragraph. This makes debugging easier compared to black-box multimodal embeddings.

- Extensibility: The design remains modular. Each channel (text, tables, images) can later be swapped with stronger models without rewriting the entire pipeline.

## Observations

While the split-modal approach works, the retrieval quality isn’t perfect:

- Table rows can sometimes lose context if headers aren’t strongly tied to each value.

- OCR on charts extracts percentages but may miss narrative context (“Technology is the largest sector at 23%”).

- BLIP captions are generic (“A pie chart of sectors”) and don’t always reflect numerical detail.

- EVals are not that good. It is mostly hit or miss. Given time constraints, I did not work deeply into it.


## Suggested improvements

If I had more time and compute, here’s what I’d focus on:

# Smarter Table + Chart Handling

- Use a table-aware model (like TAPAS) to better capture row/column relationships.

- Improve chart parsing so values and their context (“Technology is the largest sector at 23%”) stay tied together.

# Native Multimodal Embeddings

- Upgrade from split-modal to a model like BLIP-2 or LLaVA that jointly encodes text, tables, and images at the page level.

- This would improve accuracy for questions where context is spread across visuals and text.



---

## Setup

```bash
# Create & activate venv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

---

## Folder Structure

```
apps/mm_rag
    io_utils.py              # Utilities for directory creation, JSONL read/write
    parse_pdf.py             # Extracts text, tables, and images from PDF pages
    table_utils.py           # Processes tables into row-level and summary chunks
    image_info.py            # Generates BLIP captions and OCR key-value pairs for images/charts
    embeddings.py            # Encodes text, tables, and images into vector embeddings
    indexer.py               # Builds and loads FAISS indexes with metadata
    ingest_build_index.py    # Orchestrates parsing, embedding, and indexing pipeline
    normalize.py             # Expands acronyms and normalizes dates in queries
    reranker.py              # Cross-encoder reranking for retrieved candidates
    retriever.py             # Unified retrieval pipeline combining indexes and reranker
    query.py                 # CLI script for running a query against the index
    evals.py                 # Evaluation harness (Accuracy@1, Recall@k, MRR) using gold dataset

data/mm_rag
    Portfolio-Analysis-Sample.pdf   # Sample input PDF
    parsed                          # Extracted artifacts from PDF
        images                      # Cropped embedded images
        page_images                 # Full-page rendered snapshots
    index                           # FAISS indexes and metadata
        text.faiss
        text_meta.jsonl
        image.faiss
        image_meta.jsonl
    gold_eval.jsonl                 # Gold Q&A pairs for evaluation

```

**Folder purposes**
- `apps/mm_rag` → all core logic for parsing, embedding, indexing, retrieval, normalization, and evals  
- `data/mm_rag` → input/output workspace (PDF, parsed artifacts, FAISS indexes, eval dataset)   - Put pdf in data/mm_rag
- `parsed/` → extracted text blocks, images, tables, and page snapshots  
- `index/` → FAISS indexes + JSONL metadata  
- `gold_eval.jsonl` → gold Q&A pairs for evaluation  

---

##  How to Run

### 1) Parse PDF and build indexes
```bash
python apps/mm_rag/ingest_build_index.py --pdf data/mm_rag/Portfolio-Analysis-Sample.pdf
```

### 2) Run queries
```bash
python apps/mm_rag/query.py --q "What is the SEC yield for Portfolio 1?" 
```

### 3) Run evaluation
Prepare `data/mm_rag/gold_eval.jsonl`:

```jsonl
{"question": "What is the SEC yield for Portfolio 1?", "expected_phrase": "SEC Yield", "expected_page": 3}
{"question": "Which sector has the highest allocation?", "expected_phrase": "Technology", "expected_page": 8}
{"question": "What is the portfolio's duration?", "expected_phrase": "Duration", "expected_page": 4}
```

Run eval:
```bash
python apps/mm_rag/evals.py --gold data/mm_rag/gold_eval.jsonl --data_root data/mm_rag --k 5
```

### Sample Query

**Q:** What is the SEC yield for Portfolio 1?  
**Normalized Q:** What is the SEC (Securities and Exchange Commission) yield for Portfolio 1?  

**A:**  
Table row →  
| Total Return | Standard Deviation | Sharpe Ratio | R² | SEC Yield | 12 Month Yield |  
|--------------|-------------------|--------------|----|-----------|----------------|  
| 6.96%        | 13.04             | 0.47         | 0.96 | **1.93** | 1.90 |  

**Citation:**  
```json
{
  "id": "tbl_13_2_0",
  "modality": "table_row",
  "page": 13,
  "table_idx": 2,
  "row_idx": 0,
  "is_header": false,
  "text": "Table row →  | Total Return | Standard Deviation | Sharpe Ratio | R2 | SEC Yield | 12 Month Yield | Portfolio 1 | 6.96% | 13.04 | 0.47 | 0.96 | 1.93 | 1.90"
}


---

## Design Choices

- **Split-modal**: text, tables, and images handled separately for cost and modularity.  
- **Semantic chunking**: paragraphs, table rows with headers, chart KV pairs.  
- **Re-ranking**: cross-encoder (`ms-marco-MiniLM-L-6-v2`) over top-k candidates.  
- **Normalization**: acronyms (e.g., SEC → Securities and Exchange Commission) and dates (Q2 2023 → April–June 2023).  
- **Evaluation**: Accuracy@1, Recall@5, MRR against a gold Q&A dataset.  

---


## Future Work

- Native multimodal models (BLIP-2, LLaVA)  
- Layout-aware models (LayoutLMv3)  
- Multi-row and multi-table reasoning  
- Larger eval dataset with synonyms and OOD queries  
- Optional LLM for natural language answer generation  

---

## Summary

- End-to-end split-modal RAG pipeline built.  
- Handles text, tables, and images with captions + OCR.  
- Includes query normalization, re-ranking, and evaluation.  
- Extensible for future multimodal and layout-aware approaches.  
