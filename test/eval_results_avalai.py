from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI


EVAL_PROMPT_TEMPLATE = r"""
You are an expert evaluator for an AI-Powered Pharmaceutical Assistant.

Important: The system provides TWO context sources:
- RAG_CONTEXT: retrieved passages from a standard (simple) RAG pipeline
- GRAPHRAG_CONTEXT: retrieved evidence from a GraphRAG pipeline (graph-derived facts, paths, structured summaries, and/or supporting chunk text)

You must produce TWO independent evaluations (Groundedness and Correctness) for THREE settings:
A) rag_only: using only RAG_CONTEXT (ignore GRAPHRAG_CONTEXT completely)
B) graphrag_only: using only GRAPHRAG_CONTEXT (ignore RAG_CONTEXT completely)
C) both_combined: using RAG_CONTEXT + GRAPHRAG_CONTEXT together

Return ONLY valid JSON in the schema below. No markdown. No extra text.

========================
EVALUATION DEFINITIONS
========================

1) Groundedness (context-only):
- Judge whether the ANSWER is supported by the specified context set ONLY.
- Use ONLY the specified context set to judge support. Do NOT use outside knowledge.
- Identify key factual claims in the ANSWER (dose, contraindications, interactions, warnings, indications, mechanisms, populations, etc.).
- For each key claim, mark it as SUPPORTED, UNSUPPORTED, or CONTRADICTED against the specified context set.
- Quote evidence directly from the specified context set (<= 25 words). Never invent quotes.
- Penalize important medical specifics not present in the specified context set.

2) Correctness (medical QA):
- Judge whether the ANSWER is correct, safe, and useful for the QUESTION.
- You MAY use general medical knowledge to assess correctness and safety.
- Consider relevance, completeness, alignment with user intent, uncertainty calibration, and safety guidance.
- Treat the specified context set as the “available evidence.” If it is insufficient, a good answer should be cautious and avoid overconfident specifics.
- Penalize hallucinated specifics, unsafe dosing advice, missing major contraindications, or confidently wrong claims.

========================
SCORING
========================

Groundedness score (0–5) per setting:
5 = Fully grounded; almost all key claims supported; no major unsupported additions
4 = Mostly grounded; minor unsupported details, not clinically critical
3 = Mixed; some unsupported/overstated claims, could mislead
2 = Weak; many key claims unsupported/speculative
1 = Not grounded; mostly unrelated to context
0 = Hallucinated/contradicted; key claims contradict context or are invented

Correctness score (0–5) per setting:
5 = Correct, complete, safe, well-calibrated
4 = Mostly correct; minor omissions/wording issues
3 = Partially correct; noticeable gaps or questionable claims
2 = Likely incorrect or importantly incomplete
1 = Mostly incorrect or potentially unsafe
0 = Dangerous misinformation

========================
OUTPUT FORMAT
========================

Return ONLY valid JSON (no markdown, no extra text) in this schema:
{
  "rag_only": {
    "groundedness": {
      "score": 0-5,
      "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
      "supported_evidence": [
        {"answer_claim": "...", "context_quote": "..."}
      ],
      "unsupported_or_contradicted_claims": [
        {"answer_claim": "...", "issue": "UNSUPPORTED" | "CONTRADICTED"}
      ],
      "notes": "Max 4 sentences."
    },
    "correctness": {
      "score": 0-5,
      "verdict": "CORRECT" | "MOSTLY_CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT" | "DANGEROUS",
      "major_issues": [
        {"type": "INCORRECT_FACT" | "UNSAFE_ADVICE" | "OMISSION" | "NON_ANSWER" | "OVERCONFIDENT", "detail": "..."}
      ],
      "notes": "Max 4 sentences."
    }
  },
  "graphrag_only": {
    "groundedness": {
      "score": 0-5,
      "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
      "supported_evidence": [
        {"answer_claim": "...", "context_quote": "..."}
      ],
      "unsupported_or_contradicted_claims": [
        {"answer_claim": "...", "issue": "UNSUPPORTED" | "CONTRADICTED"}
      ],
      "notes": "Max 4 sentences."
    },
    "correctness": {
      "score": 0-5,
      "verdict": "CORRECT" | "MOSTLY_CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT" | "DANGEROUS",
      "major_issues": [
        {"type": "INCORRECT_FACT" | "UNSAFE_ADVICE" | "OMISSION" | "NON_ANSWER" | "OVERCONFIDENT", "detail": "..."}
      ],
      "notes": "Max 4 sentences."
    }
  },
  "both_combined": {
    "groundedness": {
      "score": 0-5,
      "verdict": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED",
      "supported_evidence": [
        {"answer_claim": "...", "context_quote": "...", "source": "RAG_CONTEXT" | "GRAPHRAG_CONTEXT"}
      ],
      "unsupported_or_contradicted_claims": [
        {"answer_claim": "...", "issue": "UNSUPPORTED" | "CONTRADICTED"}
      ],
      "notes": "Max 4 sentences."
    },
    "correctness": {
      "score": 0-5,
      "verdict": "CORRECT" | "MOSTLY_CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT" | "DANGEROUS",
      "major_issues": [
        {"type": "INCORRECT_FACT" | "UNSAFE_ADVICE" | "OMISSION" | "NON_ANSWER" | "OVERCONFIDENT", "detail": "..."}
      ],
      "notes": "Max 4 sentences."
    }
  }
}

Now evaluate:

QUESTION:
{{question}}

RAG_CONTEXT:
{{rag_context}}

GRAPHRAG_CONTEXT:
{{graphrag_context}}

ANSWER:
{{answer}}
""".strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 40] + "\n...[TRUNCATED]..."


def format_rag_context(rag_context: Any, max_total_chars: int = 18000) -> str:
    """
    results_1.json: rag_context is typically a list of dicts: {id, metadata, text}
    """
    if rag_context is None:
        return ""
    if isinstance(rag_context, str):
        return _truncate(rag_context, max_total_chars)

    parts: List[str] = []
    if isinstance(rag_context, list):
        for i, item in enumerate(rag_context, start=1):
            if isinstance(item, dict):
                rid = item.get("id", "")
                meta = item.get("metadata", {})
                drug = meta.get("name_en") or meta.get("name_fa") or meta.get("drug_id") or ""
                section_keys = meta.get("included_section_keys", "")
                text = item.get("text", "")
                parts.append(
                    f"[RAG {i}] id={rid}\n"
                    f"drug={drug}\n"
                    f"sections={section_keys}\n"
                    f"text:\n{text}\n"
                )
            else:
                parts.append(f"[RAG {i}] {str(item)}\n")
    else:
        parts.append(json.dumps(rag_context, ensure_ascii=False, indent=2))

    combined = "\n".join(parts).strip()
    return _truncate(combined, max_total_chars)


def format_graphrag_context(graphrag_context: Any, max_total_chars: int = 18000) -> str:
    """
    results_1.json: graph_rag_context is typically a dict with keys like:
    - chunk_texts: list[{chunk_id, text, title, section}]
    - graph_data: dict (facts / structured summaries)
    - chunks: list (ids / references)
    """
    if graphrag_context is None:
        return ""
    if isinstance(graphrag_context, str):
        return _truncate(graphrag_context, max_total_chars)

    parts: List[str] = []
    if isinstance(graphrag_context, dict):
        chunk_texts = graphrag_context.get("chunk_texts")
        graph_data = graphrag_context.get("graph_data")

        if isinstance(chunk_texts, list) and chunk_texts:
            parts.append("=== GRAPHRAG CHUNK_TEXTS ===")
            for i, ch in enumerate(chunk_texts, start=1):
                if isinstance(ch, dict):
                    cid = ch.get("chunk_id", "")
                    section = ch.get("section", "")
                    title = ch.get("title", "")
                    text = ch.get("text", "")
                    parts.append(
                        f"[GRAPHRAG CHUNK {i}] chunk_id={cid}\n"
                        f"title={title}\n"
                        f"section={section}\n"
                        f"text:\n{text}\n"
                    )
                else:
                    parts.append(f"[GRAPHRAG CHUNK {i}] {str(ch)}\n")

        if graph_data is not None:
            parts.append("=== GRAPHRAG GRAPH_DATA (STRUCTURED FACTS) ===")
            parts.append(json.dumps(graph_data, ensure_ascii=False, indent=2))

        # Include any remaining top-level fields (optional, trimmed)
        other_keys = {k: v for k, v in graphrag_context.items() if k not in {"chunk_texts", "graph_data"}}
        if other_keys:
            parts.append("=== GRAPHRAG OTHER_FIELDS ===")
            parts.append(json.dumps(other_keys, ensure_ascii=False, indent=2))

    else:
        parts.append(json.dumps(graphrag_context, ensure_ascii=False, indent=2))

    combined = "\n".join(parts).strip()
    return _truncate(combined, max_total_chars)


def build_prompt(question: str, rag_context: Any, graphrag_context: Any, answer: str) -> str:
    rag_text = format_rag_context(rag_context)
    graphrag_text = format_graphrag_context(graphrag_context)

    prompt = EVAL_PROMPT_TEMPLATE
    prompt = prompt.replace("{{question}}", question or "")
    prompt = prompt.replace("{{rag_context}}", rag_text or "")
    prompt = prompt.replace("{{graphrag_context}}", graphrag_text or "")
    prompt = prompt.replace("{{answer}}", answer or "")
    return prompt


def call_judge(client: OpenAI, model: str, prompt: str, seed: int = 42, max_retries: int = 4) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON matching the schema. No markdown. No extra text."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                seed=seed,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            return json.loads(text)

        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            # Retry with a stronger instruction appended
            prompt = prompt + "\n\nCRITICAL: Output MUST be valid JSON and MUST match the schema exactly."
        except Exception as e:
            last_err = e

        time.sleep(2 ** attempt)

    raise RuntimeError(f"Judge call failed after {max_retries} retries. Last error: {last_err}")


def summarize(judged: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(vals: List[float]) -> float:
        return round(sum(vals) / max(len(vals), 1), 4)

    out: Dict[str, Any] = {}
    for setting in ["rag_only", "graphrag_only", "both_combined"]:
        g_scores, c_scores = [], []
        for item in judged:
            ev = item.get("evaluation", {})
            block = ev.get(setting, {})
            g = block.get("groundedness", {}).get("score")
            c = block.get("correctness", {}).get("score")
            if isinstance(g, (int, float)):
                g_scores.append(float(g))
            if isinstance(c, (int, float)):
                c_scores.append(float(c))
        out[setting] = {
            "groundedness_avg": avg(g_scores),
            "correctness_avg": avg(c_scores),
            "n_scored": len(g_scores),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="results_1.json")
    parser.add_argument("--outfile", default="results_1_judged.json")
    parser.add_argument("--limit", type=int, default=0, help="0 = evaluate all rows")
    parser.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between requests")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("AVALAI_API_KEY")
    base_url = os.getenv("AVALAI_BASE_URL", "https://api.avalai.ir/v1")
    model = os.getenv("AVALAI_MODEL", "gpt-5.1")

    if not api_key:
        raise RuntimeError("Missing AVALAI_API_KEY in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(args.infile, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of objects.")

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    judged_rows: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Judging"):
        question = row.get("question", "")
        rag_context = row.get("rag_context", "")
        graphrag_context = row.get("graph_rag_context", "")
        answer = row.get("answer", "")

        prompt = build_prompt(question, rag_context, graphrag_context, answer)
        evaluation = call_judge(client, model, prompt)

        out_row = dict(row)
        out_row["evaluation"] = evaluation
        judged_rows.append(out_row)

        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = summarize(judged_rows)
    payload = {"summary": summary, "items": judged_rows}

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
