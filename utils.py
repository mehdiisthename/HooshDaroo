import uuid
import json
from datetime import datetime

import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8192


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def split_text_into_chunks(text: str, max_tokens=MAX_TOKENS):
    tokens = ENCODER.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(ENCODER.decode(chunk_tokens))

    return chunks


def new_id(prefix: str):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now():
    return datetime.now().isoformat()


def unique_dicts(dict_list):
    seen = set()
    out = []
    for d in dict_list:
        key = json.dumps(d, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out