"""
Query normalization pipeline:
1. Spell correction (pyspellchecker)
2. Acronym expansion (dict → LLM fallback)
3. Date resolution (detect cues → LLM resolves to absolute ranges)
"""

import re
from datetime import datetime
from spellchecker import SpellChecker
from llm_helpers import llm_expand_acronyms, llm_resolve_dates

# -----------------------------
# Config
# -----------------------------
ACRONYM_MAP = {
    "NLP": "Natural Language Processing",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
    "DL": "Deep Learning",
    "RL": "Reinforcement Learning"
}

TIME_CUES = [
    "today", "yesterday", "now",
    "day", "days", "week", "weeks",
    "month", "months", "year", "years",
    "quarter", "decade", "since"
]

spell = SpellChecker(distance=1)

# -----------------------------
# Acronym expansion
# -----------------------------
def expand_acronyms(query: str):
    """
    Expand acronyms:
    -Checks for acronyms in the query using regex pattern of 2+ uppercase letters (optionally followed by digits).
    - If in dict → expand deterministically using the values in dict
    - Else → fallback to LLM to rewrite the query and determine the expansions based on context of the query
    """
    print(query)
    acronyms = re.findall(r"\b[A-Z]{2,}(?:[0-9]+)?\b", query)
    print(acronyms)
    for ac in acronyms:
        if ac in ACRONYM_MAP:
            print("lol")
            full = ACRONYM_MAP[ac]
            if f"{ac} (" not in query:  # avoid duplicate expansion
                query = query.replace(ac, f"{ac} ({full})")
        else:
            print("Explanding acronym via LLM:", ac)
            # Fallback → let LLM rewrite query with all expansions
            return llm_expand_acronyms(query)
    
    return query

# -----------------------------
# Temporal expression handling
# -----------------------------
def looks_temporal(query: str):
    q_lower = query.lower()
    return any(cue in q_lower for cue in TIME_CUES)


def resolve_dates(query: str):
    """
    If query has temporal cues, call LLM to resolve into [start_date, end_date].
    Else, leave query unchanged.
    """
    if not looks_temporal(query):
        return query, None, None
    
    print("Resolving dates for query:", query)

    today = datetime.now().date().isoformat()
    result = llm_resolve_dates(query, today)

    if result and "start_date" in result and "end_date" in result:
        start, end = result["start_date"], result["end_date"]
        # Validate ISO format
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        if re.fullmatch(date_pattern, start) and re.fullmatch(date_pattern, end):
            return query, start, end

    return query, None, None


# -----------------------------
# Spell correction
# -----------------------------
def correct_spelling(query: str) -> str:
    # find acronyms like ML, RNA, AI
    acronyms = re.findall(r"\b[A-Z]{2,}(?:[0-9]+)?\b", query)
    corrected_words = []
    for word in query.split():
        if word in acronyms:  
            corrected_words.append(word)  # keep acronyms untouched
        else:
            corrected_words.append(spell.correction(word) or word)
    return " ".join(corrected_words)

# -----------------------------
# Unified pipeline
# -----------------------------
def normalize_query(query: str):
    """
    Full normalization pipeline.
    Returns:
      - normalized query string
      - start_date, end_date (if resolved, else None)
    """
    # 1. Correct spelling
    query = correct_spelling(query)

    # 2. Acronym expansion (dict first, fallback to LLM)
    query = expand_acronyms(query)

    # 3. Date resolution (only if cues are present)
    query, start, end = resolve_dates(query)

    return query, start, end
