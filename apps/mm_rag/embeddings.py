# apps/mm_rag/embeddings.py
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------- Text ----------
class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        vecs = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

# ---------- Images (CLIP) ----------
class ImageEmbedder:
    def __init__(self, clip_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_paths(self, image_paths: List[str]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, 512), dtype="float32")
        feats = []
        for p in tqdm(image_paths, desc="Embedding images (CLIP)"):
            img = Image.open(p).convert("RGB")
            t = self.preprocess(img).unsqueeze(0).to(self.device)
            v = self.model.encode_image(t)
            v = v / v.norm(dim=-1, keepdim=True)
            feats.append(v.cpu().numpy())
        return np.vstack(feats).astype("float32")

    @torch.no_grad()
    def encode_text_for_clip(self, queries: List[str]) -> np.ndarray:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tok = tokenizer(queries).to(self.device)
        v = self.model.encode_text(tok)
        v = v / v.norm(dim=-1, keepdim=True)
        return v.cpu().numpy().astype("float32")

# ---------- Captions (BLIP base) ----------
class Captioner:
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def caption_paths(self, image_paths: List[str], max_new_tokens: int = 36) -> List[str]:
        caps = []
        for p in tqdm(image_paths, desc="Captioning images (BLIP)"):
            image = Image.open(p).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.processor.decode(out[0], skip_special_tokens=True)
            caps.append(text.strip())
        return caps
