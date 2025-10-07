# Semantic Search & Multi-Modal RAG Project

This project implements two challenges:

1. **Semantic Search Engine**  
   Retrieve the most relevant academic papers based on query intent & context.
2. **Multi-Modal RAG Pipeline**  
   Build QA over PDF documents with charts, tables, images, and text.

---

## Setup Instructions

### 1. Clone Repository & Setup Environment
```bash
git clone <your-repo-url> RAG_and_multimodal
cd RAG_and_multimodal

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Mac/Linux
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

---

## Commands

### 1. Fetch Arxiv Dataset
Fetch 40 academic papers (PDFs + metadata).
```bash
python scripts/build_arxiv_dataset.py
```
If you want to give your own papers as input, then place the papers in PDF format in /data/raw_pdfs. In this case, you can ignore the above command and directly run build index.

### 2. Build Index
Generate FAISS index + BM25 for hybrid retrieval.
```bash
python apps/semantic_search/build_index.py
```

### 3. Run Search
Perform retrieval with hybrid (dense + sparse) retriever + reranking.
```bash
python apps/semantic_search/search.py --query "Risk assessment" --mode hybrid --rerank --top_k 5
```

### 4. Run Evaluation
Evaluate on question-answer dataset (`rag_eval_dataset.jsonl`).
```bash
python apps/semantic_search/eval.py
```

---

## Project Structure

```
RAG_and_multimodal/
    
    apps/
        semantic_search/
            build_index.py        builds FAISS + BM25 indexes
            retriever.py          dense / sparse / hybrid retrievers
            reranker.py           cross-encoder reranking of candidates
            normalize.py          cleans queries (spellcheck, acronyms, dates)
            search.py             main script to run retrieval end-to-end
            eval.py               evaluation script (Accuracy, Recall@k, MRR)
            logger.py             logs queries/results for data flywheel
            __init__.py           makes the folder a Python package

    scripts/
        build_arxiv_dataset.py    downloads arXiv PDFs + extracts metadata/fulltext

    data/
        semantic_search/
            corpus.jsonl          dataset of academic papers (id, title, abstract, text)
            index/faiss.index     dense vector index built with FAISS
            query_logs.jsonl      user query logs (for flywheel/retraining)
            rag_eval_dataset.jsonl  QA pairs for evaluation

    requirements.txt              dependencies (faiss, bm25, sentence-transformers, etc.)

```

---

##  Design Choices

###  Retrieval
- **Dense**: Sentence embeddings (MiniLM) stored in **FAISS** index.  
- **Sparse**: BM25 keyword retriever.  
- **Hybrid**: Weighted fusion of both.  

###  Reranking
- Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) re-scores top candidates.  
- Improves **Accuracy@1** and **Recall@k**.  

###  Normalization
- **Spelling correction** → `pyspellchecker`. Used pyspell checker to correct spelling mistakes in queries 
- **Acronym expansion** → dictionary + fallback to LLM.  Used regex to detect any abbreviations. Used a map of abbreviation and acronym to fill it. If not, LLM is used as fallback option to fill the abbreviation based on query context
- **Date resolution** → Checks for any temporal keywords in the query and if so, today's date is calculated which along with the query is sent to LLM to generate filter dates based on the query. For example if query has last week, the LLM generates start and end date for 1 week period which can be used to filter the metadata. As the assignment does not have date column for now, I did not filter by date but logic that deals with dates is implemented  

###  Evaluation
- Metrics: **Accuracy@1, Recall@k, MRR**.  
- Uses `rag_eval_dataset.jsonl` with Q/A + source mapping.  

###  Data Flywheel
- Every query, retrieved results, and user feedback are logged into `query_logs.jsonl`.  
- This data can later be:  
  - Used to retrain BM25 / embeddings.  
  - Improve acronym/date handling.  
  - Build custom domain-specific retrievers.  
- I did not fully implement custom data flywheel but demonstrated logging for future implementation.

---

##  Example Output

```text
============================================================
ID: 2509.18857v1
Title: Optimal estimation for regression discontinuity design with binary outcomes
Authors: Takuya Ishihara, Masayuki Sawada, Kohei Yata
Abstract: We develop a finite-sample optimal estimator ...
Score: 0.914
============================================================
```

---

##  Evaluation Example

```
Retriever: hybrid, Rerank: True, Top-k: 5
Accuracy@1: 87.76% (43/49)
Recall@5: 97.96% (48/49)
MRR: 0.914
```

---

##  Summary

- **Semantic Search** → Retrieves papers using hybrid retrievers.  
- **Normalization** → Fixes queries (spellings, acronyms, dates).  
- **Reranking** → Improves ranking quality.  
- **Evaluation** → Measures retrieval effectiveness.  
- **Data Flywheel** → Logs user interactions to improve system iteratively.  
```