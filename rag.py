import json
import re
import numpy as np
import streamlit as st

import faiss
import os
import pickle

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi


load_dotenv()

FAISS_URI = os.getenv("FAISS_URI")
METADATA_PATH = os.getenv("FAISS_METADATA")

# Normalizing for Persian text
def normalize_persian(text: str) -> str:
    if not text:
        return ""

    text = text.replace("ي", "ی").replace("ك", "ک").replace("ئ", "ی")
    text = re.sub(r"[ـ]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()

# Loading the document chunks
@st.cache_resource
def load_chunks():
    with open("data/rag_chunks_v5.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

chunks = load_chunks()


# Storage for BM-25 Retrieval
@st.cache_resource
def store_bm25():
    bm25_texts = []
    bm25_id_map = []

    for c in chunks:
        norm_text = normalize_persian(c["embedding_text"])
        bm25_texts.append(norm_text.split())
        bm25_id_map.append(c["id"])

    bm25 = BM25Okapi(bm25_texts)
    return bm25, bm25_id_map

bm25, bm25_id_map = store_bm25()


# Loading the FAISS Index
@st.cache_resource
def get_faiss_db():
    index = faiss.read_index(FAISS_URI)
    
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

index, metadata = get_faiss_db()




# Retrieval Function
def hybrid_retrieve(query: str, embedder, top_k=5, alpha=0.5):
    # --- BM25 ---
    norm_q = normalize_persian(query)
    bm25_scores = bm25.get_scores(norm_q.split())
    bm25_scores = np.array(bm25_scores)

    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

    # --- Dense ---
    q_emb = embedder.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True
        ).astype("float32")

    scores, indices = index.search(q_emb, top_k * 5)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "score": float(score),
            "data": metadata[idx]
        })

    dense_scores = {}
    dense_data = {}

    for hit in results:
        data = hit["data"]
        cid = data["id"]
        score = hit["score"]

        dense_scores[cid] = score
        dense_data[cid] = data


    # Normalizing the dense scores for fusion
    dense_vals = np.array(list(dense_scores.values()))
    dense_min, dense_max = dense_vals.min(), dense_vals.max()

    dense_norm = {
        k: (v - dense_min) / (dense_max - dense_min + 1e-9)
        for k, v in dense_scores.items()
    }

    # --- Fusion ---
    final_scores = {}
    for idx, cid in enumerate(bm25_id_map):
        if cid in dense_scores:
            final_scores[cid] = (
                alpha * bm25_norm[idx] +
                (1 - alpha) * dense_norm[cid]
            )

    top_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]

    return [dense_data[i] for i in top_ids]


