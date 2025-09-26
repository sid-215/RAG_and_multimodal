"""
LLM helpers using a local LLaMA model via Ollama.
"""

import json
import ollama
from datetime import datetime

MODEL = "llama3.2:3b"  # you can replace with any model you pulled

# -----------------------------
# Acronym expansion
# -----------------------------
def llm_expand_acronyms(query: str) -> str:
    prompt = f"""
    Rewrite the following query by expanding the acronyms into their full forms.
    Keep both the acronym and the expansion in parentheses.
    Example: "ML methods" -> "ML (Machine Learning) methods".
    Do not modify any other part of the query except for expanding acronyms. 
    Just return the modified query, NOTHING ELSE.
    Query: {query}
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

# -----------------------------
# Date resolution
# -----------------------------
def llm_resolve_dates(query: str, today: str = None):
    if today is None:
        today = datetime.now().date().isoformat()

    prompt = f"""
    Today's date is {today}.
    Extract any time expressions in the query and convert them into an absolute date range.
    Return only and only the valid JSON in the format and NOTHING ELSE :
    {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
    If no time expression is found, return null.

    Query: {query}
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    text = response["message"]["content"].strip()

    try:
        return json.loads(text)
    except Exception:
        return None
