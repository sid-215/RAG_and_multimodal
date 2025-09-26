# apps/mm_rag/image_info.py
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import pytesseract
import re
from dataclasses import dataclass

@dataclass
class Token:
    text: str
    x: int; y: int; w: int; h: int
    conf: float
    line_id: Tuple[int, int]

PERCENT_RE = re.compile(r"^-?\d{1,3}(?:[.,]\d+)?\s*%$")
NUMERIC_RE = re.compile(r"^-?\d{1,3}(?:[.,]\d+)?$")
LABEL_RE   = re.compile(r"^[A-Za-z][A-Za-z0-9/&\-\s\.]+$")

def _ocr_tokens(image_path: str, min_conf: int = 60) -> List[Token]:
    img = cv2.imread(image_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 2)
    data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT)
    tokens = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf = float(data.get("conf", ["0"]*n)[i])
        if conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        blk, ln = data.get("block_num", [0]*n)[i], data.get("line_num", [0]*n)[i]
        tokens.append(Token(text=txt, x=x, y=y, w=w, h=h, conf=conf, line_id=(blk, ln)))
    return tokens

def _linewise_pairs(tokens: List[Token]):
    by_line = {}
    for t in tokens:
        by_line.setdefault(t.line_id, []).append(t)
    pairs = []
    for toks in by_line.values():
        labels = [t for t in toks if LABEL_RE.match(t.text) and not PERCENT_RE.match(t.text)]
        percents = [t for t in toks if PERCENT_RE.match(t.text)]
        if not labels or not percents:
            continue
        for p in percents:
            cand = min(labels, key=lambda L: abs((L.x + L.w) - p.x))
            pairs.append((cand.text.strip(), p.text.strip()))
    return pairs

def _spatial_pairs(tokens: List[Token], max_dist: int = 120):
    labels = [t for t in tokens if LABEL_RE.match(t.text) and not PERCENT_RE.match(t.text)]
    percents = [t for t in tokens if PERCENT_RE.match(t.text)]
    pairs = []
    for p in percents:
        pcx, pcy = p.x + p.w/2, p.y + p.h/2
        best = None; best_d = 1e9
        for L in labels:
            Lcx, Lcy = L.x + L.w/2, L.y + L.h/2
            d = np.hypot(Lcx - pcx, Lcy - pcy)
            if d < best_d:
                best_d, best = d, L
        if best is not None and best_d <= max_dist:
            pairs.append((best.text.strip(), p.text.strip()))
    return pairs

def extract_chart_kv(image_path: str) -> Dict[str, Any]:
    toks = _ocr_tokens(image_path)
    if not toks:
        return {"kv": {}, "raw": []}
    pairs = _linewise_pairs(toks)
    if not pairs:
        pairs = _spatial_pairs(toks)
    seen = set(); kv = {}
    for label, val in pairs:
        key = (label.lower(), val.replace(" ", "").lower())
        if key in seen: 
            continue
        seen.add(key)
        kv[label] = val.replace(" ", "").replace(",", ".")
    # raw lines for audit
    raw_lines = {}
    for t in toks:
        raw_lines.setdefault(t.line_id, []).append(t.text)
    raw = [" ".join(v) for v in raw_lines.values()]
    return {"kv": kv, "raw": raw}
