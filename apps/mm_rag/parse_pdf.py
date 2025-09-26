from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber

def extract_text_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    out = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        for block in page.get_text("blocks"):
            if len(block) >= 5:
                txt = (block[4] or "").strip()
                if txt:
                    out.append({
                        "modality": "text",
                        "page": pno,
                        "text": txt,
                        "bbox": [float(block[0]), float(block[1]), float(block[2]), float(block[3])]
                    })
    doc.close()
    return out

def extract_tables(pdf_path: Path) -> List[Dict[str, Any]]:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables):
                if not table or len(table) == 0:
                    continue
                header = [(c or "").strip() for c in table[0]] if table[0] else []
                rows = []
                for r in table[1:] if len(table) > 1 else []:
                    rows.append([(c or "").strip() for c in r])
                out.append({
                    "page": pno,
                    "table_idx": t_idx,
                    "header": header,
                    "rows": rows
                })
    return out

def extract_images(pdf_path: Path, out_dir: Path):
    out = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            ext = base.get("ext", "png")
            img_path = out_dir / "parsed" / "images" / f"page_{pno:03d}_img_{idx}_{xref}.{ext}"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            out.append({
                "modality": "image",
                "page": pno,
                "xref": int(xref),
                "path": str(img_path)
            })
        pix = page.get_pixmap(dpi=150)
        page_img_path = out_dir / "parsed" / "page_images" / f"page_{pno:03d}.png"
        pix.save(str(page_img_path))
        out.append({
            "modality": "page_image",
            "page": pno,
            "path": str(page_img_path)
        })
    doc.close()
    return out
