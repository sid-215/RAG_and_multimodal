import re

# We can expand more acronyms as needed and setup a bigger dictionary or use an LLM based on context.

ACRONYM_MAP = {
    "SEC": "Securities and Exchange Commission",
    "MM": "Money Market",
    "NAV": "Net Asset Value",
    "ETF": "Exchange Traded Fund",
    "YTM": "Yield to Maturity"
}

QUARTER_MAP = {
    "Q1": ("January", "March"),
    "Q2": ("April", "June"),
    "Q3": ("July", "September"),
    "Q4": ("October", "December")
}

def expand_acronyms(query: str) -> str:
    words = query.split()
    out = []
    for w in words:
        key = w.upper().strip(",.")
        if key in ACRONYM_MAP:
            out.append(f"{w} ({ACRONYM_MAP[key]})")
        else:
            out.append(w)
    return " ".join(out)

def normalize_dates(query: str) -> str:
    q_match = re.search(r"(Q[1-4])\s+(\d{4})", query, re.IGNORECASE)
    if q_match:
        q, year = q_match.group(1).upper(), q_match.group(2)
        if q in QUARTER_MAP:
            start, end = QUARTER_MAP[q]
            query = query.replace(q_match.group(0), f"{start} to {end} {year}")
    months = {m[:3].lower(): m for m in [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]}
    for abbr, full in months.items():
        query = re.sub(rf"\b{abbr}\s+(\d{{4}})", f"{full} \\1", query, flags=re.IGNORECASE)
    return query

def normalize_query(query: str) -> str:
    q = expand_acronyms(query)
    q = normalize_dates(q)
    return q
