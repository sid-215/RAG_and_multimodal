import json
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = "data/semantic_search/query_logs.jsonl"

def log_interaction(query, normalized_query, retrieved, chosen_doc=None, feedback=None):
    """
    Log query interactions for future training.
    
    Args:
        query (str): raw user query
        normalized_query (str): query after normalization (spellcheck, acronym expansion, etc.)
        retrieved (list): list of retrieved paper_ids with scores
        chosen_doc (str): optional, paper_id of doc user clicked/selected
        feedback (str): optional, e.g. 'good', 'bad', 'needs expansion'
    """
    Path(os.path.dirname(LOG_FILE)).mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "normalized_query": normalized_query,
        "retrieved": retrieved,  # [{"paper_id": , "score": }]
        "chosen_doc": chosen_doc,
        "feedback": feedback,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")