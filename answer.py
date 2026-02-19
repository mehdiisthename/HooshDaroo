import ollama
import os
import streamlit as st
import json
import uuid

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rag import hybrid_retrieve
from graphrag import GraphRAGPipeline

from typing import Dict
from prompts import SYSTEM_PROMPT
from utils import unique_dicts


load_dotenv()

# Semantic Embedding using QWEN3-Embedding-0.6B
EMBEDDING = os.getenv("EMBEDDING_MODEL")

@st.cache_resource
def get_embedder():
    return SentenceTransformer(
        EMBEDDING,
        device="cuda"
    )

embedder = get_embedder()


@st.cache_resource
def get_graphrag_pipeline():
    return GraphRAGPipeline(embedder)

graph_rag = get_graphrag_pipeline()


# Save the Graph RAG retrieved context as a JSON file, to have references for later
REF_PATH = "graph_refs"

def graph_ref(graph_output: Dict):
    os.makedirs(REF_PATH, exist_ok=True)
    filename = f"{uuid.uuid4()}.json"
    file_path = os.path.join(REF_PATH, filename)

    # Save JSON with proper UTF-8 handling
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(graph_output, f, ensure_ascii=False, indent=4)

    return file_path



# Using Ollama for generation
def answer_question_stream(query: str, history: str):
    rag_contexts = hybrid_retrieve(query, embedder, alpha=0.5)
    graph_rag_context = graph_rag.execute(query)
    context_blocks = []
    sources = []

    for c in rag_contexts:
        meta = c.get("metadata") or {}
        section = meta.get("included_section_keys", "info")
        text = c.get("text")

        context_blocks.append(f"{section}\n{text}")

        source = {"name": meta.get('name_fa'),
                  "source": meta.get('snapshot_url'),
                  "sections": meta['included_section_keys'],
                 }
        sources.append(source)
    sources = unique_dicts(sources)

    context_text = "\n\n".join(context_blocks)

    user_prompt = f"""
** Retrieved Context from the Vector Database **:
{context_text}

----------------------------------

** Retrieved Context from the Knowledge Graph **:
{graph_rag_context}

----------------------------------

** Conversation history or summary **:
{history}

----------------------------------

** User's Question **:
{query}

Answer in Persian.
"""
    prompt = "\n\n".join([SYSTEM_PROMPT, user_prompt])

    stream = ollama.chat(
        messages=[{"role": "user", "content": prompt}],
        model="gemma3:4b",
        stream=True
    )

    def generator():
        for chunk in stream:
            yield chunk["message"]["content"]

    graph_ref(graph_rag_context)

    return generator(), sources
