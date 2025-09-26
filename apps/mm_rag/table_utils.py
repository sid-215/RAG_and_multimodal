def table_to_row_chunks(tables):
    chunks = []
    for t in tables:
        hdr = [h for h in t["header"]] if t.get("header") else []
        hdr_text = " | ".join(hdr) if hdr else ""
        page = t["page"]
        t_idx = t["table_idx"]
        if not t.get("rows"):
            continue
        for r_idx, row in enumerate(t["rows"]):
            row_text = " | ".join(row)
            joined = (hdr_text + "\n" + row_text) if hdr_text else row_text
            chunks.append({
                "id": f"tbl_{page}_{t_idx}_{r_idx}",
                "modality": "table_row",
                "page": page,
                "table_idx": t_idx,
                "row_idx": r_idx,
                "is_header": False,
                "text": f"Table row → {joined}"
            })
    return chunks

def table_to_summary_chunks(tables, max_rows=5):
    chunks = []
    for t in tables:
        page = t["page"]; t_idx = t["table_idx"]
        hdr = " | ".join(t["header"]) if t.get("header") else ""
        body = t.get("rows", [])[:max_rows]
        lines = [" | ".join([(c or "") for c in r]) for r in body]
        summary = f"Table summary → {hdr}\n" + "\n".join(lines) if hdr else "Table summary\n" + "\n".join(lines)
        chunks.append({
            "id": f"tbl_summary_{page}_{t_idx}",
            "modality": "table_summary",
            "page": page,
            "table_idx": t_idx,
            "is_header": True,
            "text": summary
        })
    return chunks
