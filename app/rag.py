from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL = None  # lazy load for faster imports

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"

def _get_model():
    global EMB_MODEL
    if EMB_MODEL is None:
        EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
    return EMB_MODEL

def load_kb(kb_dir: str = "kb") -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        docs.append((str(p), p.read_text(encoding="utf-8")))
    return docs

def split_chunks(text: str, size: int = 700, overlap: int = 120) -> List[str]:
    chunks, i = [], 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int = 3, lambda_mult: float = 0.5) -> List[int]:
    """Simple MMR to diversify results (cosine sim w/ normalized vecs)."""
    selected: List[int] = []
    candidates = list(range(doc_vecs.shape[0]))
    sims = (doc_vecs @ query_vec.T).flatten()  # cosine since normalized
    for _ in range(min(top_k, len(candidates))):
        if not selected:
            i = int(np.argmax(sims))
            selected.append(i)
            candidates.remove(i)
            continue
        repulsion = np.max(doc_vecs[selected] @ doc_vecs.T, axis=0)
        scores = lambda_mult * sims - (1 - lambda_mult) * repulsion
        scores[selected] = -np.inf
        i = int(np.argmax(scores))
        selected.append(i)
        if i in candidates:
            candidates.remove(i)
    return selected

class VectorStore:
    def __init__(self):
        self.docs: list[tuple[str, str]] = []
        self.chunks: list[tuple[str, str]] = []
        self.index: faiss.Index | None = None
        self.vecs: np.ndarray | None = None

    def build(self, kb_dir: str = "kb", persist: bool = True):
        self.docs = load_kb(kb_dir)
        self.chunks.clear()
        for path, txt in self.docs:
            for ch in split_chunks(txt):
                self.chunks.append((path, ch))
        model = _get_model()
        embs = model.encode([c for _, c in self.chunks], convert_to_numpy=True)
        embs = embs.astype("float32")
        faiss.normalize_L2(embs)
        self.vecs = embs
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        if persist:
            faiss.write_index(self.index, str(INDEX_PATH))
            with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)

    def load(self) -> bool:
        if not (INDEX_PATH.exists() and CHUNKS_PATH.exists()):
            return False
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.chunks = [(str(p), str(c)) for p, c in raw]
        # Recompute vecs only if needed by tests; not strictly necessary for querying
        # but useful for MMR and size checks.
        model = _get_model()
        embs = model.encode([c for _, c in self.chunks], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embs)
        self.vecs = embs
        return True

    def search(self, q: str, k: int = 3) -> list[tuple[str, str]]:
        assert self.index is not None, "Index not built/loaded"
        model = _get_model()
        qv = model.encode([q], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, max(k * 4, k))  # a few extra for MMR
        candidates = I[0].tolist()
        # MMR selection using cached vecs
        if self.vecs is not None:
            chosen = mmr(qv[0], self.vecs[candidates], top_k=k)
            idxs = [candidates[i] for i in chosen]
        else:
            idxs = candidates[:k]
        return [self.chunks[i] for i in idxs]
