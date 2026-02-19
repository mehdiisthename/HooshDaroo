import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_embedding_index")

AVALAI_API_KEY = os.getenv("AVALAI_API_KEY")
AVALAI_BASE_URL = os.getenv("AVALAI_BASE_URL", "https://api.avalai.ir/v1")
AVALAI_LLM_MODEL = os.getenv("AVALAI_LLM_MODEL", "gpt-5-mini")
AVALAI_REASONING_EFFORT = os.getenv("AVALAI_REASONING_EFFORT", "minimal")
AVALAI_TEXT_VERBOSITY = os.getenv("AVALAI_TEXT_VERBOSITY", "low")
AVALAI_MAX_OUTPUT_TOKENS = int(os.getenv("AVALAI_MAX_OUTPUT_TOKENS", "600"))


client = OpenAI(
    base_url=AVALAI_BASE_URL,
    api_key=AVALAI_API_KEY,
)

MODEL = AVALAI_LLM_MODEL
REASONING_EFFORT = AVALAI_REASONING_EFFORT
TEXT_VERBOSITY = AVALAI_TEXT_VERBOSITY
MAX_OUT = AVALAI_MAX_OUTPUT_TOKENS

ALLOWED_TYPES = [
    "drug",
    "interaction_agent",
    "condition",
    "adverse_effect",
    "drug_class",
    "population",
    "context",
    "chemical",
]

ALLOWED_INTENTS = [
    "INDICATION",
    "ADVERSE_EFFECT",
    "INTERACTION",
    "CONTRAINDICATION",
    "CAUTION",
    "COMPARISON",
    "GENERAL_INFO",
    "POPULATION_SPECIFIC",
    "ASSOCIATION",
    "CAUSE",
]

ENTITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "language": {"type": "string"},
        "intents": {
            "type": "array",
            "items": {"type": "string", "enum": ALLOWED_INTENTS},
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "entity_type": {"type": "string", "enum": ALLOWED_TYPES},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                },
                "required": ["text", "entity_type", "confidence", "start", "end"],
            },
        },
    },
    "required": ["language", "intents", "entities"],
}


def extract_entities_avalai(question: str, debug: bool = False):
    system = (
        "You are a text annotation system.\n"
        "Task: extract entity mentions and intent label(s).\n"
        "No medical advice. No explanations.\n"
        "Return only the JSON object that matches the provided schema."
    )
    user = (
        "Extract entity mentions and user intent(s) from the question.\n"
        "Don’t output generic words مثل دارو/داروها/داروهایی/عوارض جانبی/تداخل\n"
        "Also choose one or more intent labels from ALLOWED_INTENTS that best match the question.\n"
        "Important:\n"
        "- Use the exact surface form from the question for text.\n"
        "- If uncertain, omit.\n"
        "- start/end are character offsets in the original question if possible, else 0.\n\n"
        f"Question: {question}"
    )
    raw = ""
    parsed = {"language": "unknown", "intents": [], "entities": []}
    resp = None

    for _attempt in range(3):
        resp = client.responses.create(
            model=MODEL,
            reasoning={"effort": REASONING_EFFORT},
            text={
                "verbosity": TEXT_VERBOSITY,
                "format": {
                    "type": "json_schema",
                    "name": "EntityExtraction",
                    "strict": True,
                    "schema": ENTITY_SCHEMA,
                },
            },
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            max_output_tokens=MAX_OUT,
            store=False,
        )

        raw = getattr(resp, "output_text", None) or ""
        parsed = json.loads(raw) if raw else {"language": "unknown", "intents": [], "entities": []}

        lang = (parsed.get("language") or "").strip()
        intents = parsed.get("intents") or []
        ents = parsed.get("entities") or []
        if lang and lang.lower() != "unknown" and len(intents) > 0 and len(ents) > 0:
            break

    if debug:
        print("\n--- status ---", getattr(resp, "status", None))
        print("\n--- output_text (repr) ---", repr(raw))
        try:
            print("\n--- usage ---", resp.usage)
        except Exception:
            pass

    return raw, parsed