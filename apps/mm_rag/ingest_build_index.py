# apps/mm_rag/ingest_build_index.py
import argparse, json
from pathlib import Path

from io_utils import ensure_dirs, write_jsonl
from parse_pdf import extract_text_blocks, extract_tables, extract_images
from table_utils import table_to_row_chunks, table_to_summary_chunks
from embeddings import TextEmbedder, ImageEmbedder, Captioner
from image_info import extract_chart_kv
from indexer import build_faiss_index, save_faiss

def ingest_and_index(pdf_path: Path, data_root: Path, use_captions: bool = True, use_image_kv: bool = True):
    ensure_dirs(data_root)

    # 1) Parse
    print("Extracting text blocks...")
    text_blocks = extract_text_blocks(pdf_path)

    print("Extracting tables...")
    tables = extract_tables(pdf_path)

    print("Extracting images & page snapshots...")
    media = extract_images(pdf_path, data_root)
    embedded_images = [m for m in media if m["modality"] == "image"]

    # Persist raw parsed
    parsed_dir = data_root / "parsed"
    write_jsonl(parsed_dir / "text_blocks.jsonl", text_blocks)
    write_jsonl(parsed_dir / "tables.jsonl", tables)
    write_jsonl(parsed_dir / "media.jsonl", media)

    # 2) Build text items
    text_items = []
    # text blocks
    for i, b in enumerate(text_blocks):
        text_items.append({
            "id": f"text_{i}",
            "modality": "text",
            "page": b["page"],
            "text": b["text"],
            "bbox": b.get("bbox"),
            "is_header": False
        })

    # table rows (answerable) + table summaries (recall only)
    row_chunks = table_to_row_chunks(tables)
    summary_chunks = table_to_summary_chunks(tables, max_rows=5)
    text_items.extend(row_chunks)
    text_items.extend(summary_chunks)

    # 3) Image-derived chunks
    img_items = []
    img_paths = [im["path"] for im in embedded_images]
    captions = []
    if use_captions and img_paths:
        captioner = Captioner()
        captions = captioner.caption_paths(img_paths)

    for k, im in enumerate(embedded_images):
        img_items.append({
            "id": f"image_{k}",
            "modality": "image",
            "page": im["page"],
            "path": im["path"],
            "xref": im.get("xref")
        })

        if use_captions and captions:
            text_items.append({
                "id": f"image_caption_{k}",
                "modality": "image_caption",
                "page": im["page"],
                "text": captions[k],
                "image_path": im["path"],
                "is_header": False
            })

        if use_image_kv:
            kv_res = extract_chart_kv(im["path"])
            if kv_res and kv_res.get("kv"):
                kv_pairs = "; ".join([f"{k_}: {v_}" for k_, v_ in kv_res["kv"].items()])
                text_items.append({
                    "id": f"image_kv_{k}",
                    "modality": "image_kv",
                    "page": im["page"],
                    "text": f"Chart values → {kv_pairs}",
                    "image_path": im["path"],
                    "is_header": False
                })

    # 4) Text index
    print(f"Embedding {len(text_items)} text/table/image-info chunks...")
    t_embedder = TextEmbedder()
    text_vecs = t_embedder.encode([ti["text"] for ti in text_items])
    text_index = build_faiss_index(text_vecs, metric="cosine")
    save_faiss(text_index, data_root / "index" / "text.faiss")
    with open(data_root / "index" / "text_meta.jsonl", "w", encoding="utf-8") as f:
        for ti in text_items:
            f.write(json.dumps(ti, ensure_ascii=False) + "\n")

    # 5) Image index (CLIP)
    print(f"Embedding {len(img_items)} images with CLIP...")
    i_embedder = ImageEmbedder()
    img_vecs = i_embedder.encode_paths([im["path"] for im in img_items])
    img_index = build_faiss_index(img_vecs, metric="cosine")
    save_faiss(img_index, data_root / "index" / "image.faiss")
    with open(data_root / "index" / "image_meta.jsonl", "w", encoding="utf-8") as f:
        for ii in img_items:
            f.write(json.dumps(ii, ensure_ascii=False) + "\n")

    print("\n✅ Ingest complete.")
    print(f"- Text vectors:  {len(text_items)}")
    print(f"- Image vectors: {len(img_items)}")
    print(f"- Data root:     {data_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=str)
    ap.add_argument("--data_root", default="data/mm_rag", type=str)
    ap.add_argument("--no_captions", action="store_true")
    ap.add_argument("--no_image_kv", action="store_true")
    args = ap.parse_args()

    ingest_and_index(
        Path(args.pdf).resolve(),
        Path(args.data_root).resolve(),
        use_captions=not args.no_captions,
        use_image_kv=not args.no_image_kv
    )
