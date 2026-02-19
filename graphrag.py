import os
import re
import streamlit as st

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process
from hazm import Normalizer, word_tokenize

from entity_extractor import extract_entities_avalai
from utils import unique_dicts


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "chunk_embedding_index")

class QuestionType(Enum):
    INDICATION = "indication"
    ADVERSE_EFFECT = "adverse_effect"
    INTERACTION = "interaction"
    CONTRAINDICATION = "contraindication"
    CAUTION = "caution"
    COMPARISON = "comparison"
    GENERAL_INFO = "general_info"
    POPULATION_SPECIFIC = "population_specific"
    ALTERNATIVE = "alternative"
    CHEMICAL = "chemical"
    CONTEXT = "context"
    CAUSE = "cause"



class RouteStrategy(Enum):
    SEMANTIC_FIRST = "semantic_first"
    ENTITY_FIRST = "entity_first"
    HYBRID = "hybrid"


@dataclass
class EntityMention:
    text: str
    entity_type: str
    confidence: float
    position: Tuple[int, int]



@dataclass
class EntityMatch:
    text: str
    entity_type: str
    key: str
    confidence: float
    position: Tuple[int, int]


@dataclass
class AnalysisResult:
    original_question: str
    normalized_question: str
    language: str
    question_types: List[QuestionType]
    entities: List[EntityMatch]
    route_strategy: RouteStrategy
    keywords: List[str]
    confidence: float


class PersianTextProcessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.persian_chars = set('آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ')

    def normalize(self, text: str) -> str:
        text = self.normalizer.normalize(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        try:
            return word_tokenize(text)
        except:
            return text.split()

    def detect_language(self, text: str) -> str:
        persian_count = sum(1 for c in text if c in self.persian_chars)
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "unknown"
        ratio = persian_count / total_alpha
        if ratio > 0.7:
            return "fa"
        if ratio < 0.3:
            return "en"
        return "mixed"


class QueryRouter:
    def decide_route(self, analysis: AnalysisResult) -> RouteStrategy:
        return RouteStrategy.HYBRID



POPULATION_TERMS_FA = {"بارداری","حاملگی","شیردهی","مادر شیرده","کودک","نوزاد","سالمند"}
GENERIC_DRUG_FA = {"دارو","داروها","داروهایی","دارویی","داروی","داروهای"}
GENERIC_ADVERSE_FA = {"عوارض","عوارض جانبی","اثر جانبی","اثرات جانبی","عارضه","مضرات"}

def normalize_entity_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(مصرف\s+)", "", t).strip()
    t = re.sub(r"^(تداخل\s+)", "", t).strip()
    t = re.sub(r"^(درمان\s+)", "", t).strip()
    return t

def remap_entity_type(entity_type: str, text: str) -> str:
    if (text or "").strip() in POPULATION_TERMS_FA:
        return "population"
    return entity_type

def should_drop(entity_type: str, text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return True
    if entity_type == "drug" and t in GENERIC_DRUG_FA:
        return True
    if entity_type == "adverse_effect" and t in GENERIC_ADVERSE_FA:
        return True
    return False

def postprocess_mentions(parsed: dict) -> dict:
    ents = parsed.get("entities", []) if parsed else []
    out = []
    for e in ents:
        et = remap_entity_type((e.get("entity_type") or "").strip(), (e.get("text") or "").strip())
        txt = normalize_entity_text((e.get("text") or "").strip())

        if should_drop(et, txt):
            continue
        out.append({
            "text": txt,
            "entity_type": et,
            "confidence": float(e.get("confidence", 0.5)),
            "start": int(e.get("start", 0) or 0),
            "end": int(e.get("end", 0) or 0),
        })

    # de-dup by (type,text)
    uniq = {}
    for x in out:
        k = (x["entity_type"], x["text"])
        if k not in uniq or x["confidence"] > uniq[k]["confidence"]:
            uniq[k] = x
    parsed["entities"] = list(uniq.values())
    return parsed



class EntityLinker:
    def __init__(self, driver, embedding_model: SentenceTransformer):
        self.driver = driver
        self.embedding_model = embedding_model
        self.tp = PersianTextProcessor()

        self.entity_cache: Dict[str, List[dict]] = {}
        self.exact_map: Dict[str, Dict[str, str]] = {}

        self._load_cache()

    def _add(self, etype: str, key: str, name: str, extra: Optional[dict]=None):
        name_n = self.tp.normalize(name)
        self.entity_cache.setdefault(etype, []).append({"key": key, "name": name_n, **(extra or {})})
        self.exact_map.setdefault(etype, {})[name_n.lower()] = key

    def _load_cache(self):
        with self.driver.session() as session:
            # Drug
            self.entity_cache["drug"] = []
            self.exact_map["drug"] = {}
            res = session.run("""
                MATCH (d:Drug)
                RETURN d.drug_id AS key, d.name_fa AS name_fa, d.name_en AS name_en, d.brand_names AS brand_names
            """)
            for r in res:
                k = r["key"]
                if r["name_fa"]:
                    self._add("drug", k, r["name_fa"], {"lang":"fa"})
                if r["name_en"]:
                    self._add("drug", k, r["name_en"], {"lang":"en"})
                if r["brand_names"]:
                    for b in r["brand_names"]:
                        if b:
                            self._add("drug", k, b, {"lang":"fa"})

            # InteractionAgent
            self.entity_cache["interaction_agent"] = []
            self.exact_map["interaction_agent"] = {}
            res = session.run("""
                MATCH (i:InteractionAgent)
                RETURN i.key AS key, i.name_fa AS name_fa, i.agent_type AS agent_type
            """)
            for r in res:
                if r["name_fa"]:
                    self._add("interaction_agent", r["key"], r["name_fa"], {"agent_type": r["agent_type"]})

            # DrugClass
            self.entity_cache["drug_class"] = []
            self.exact_map["drug_class"] = {}
            res = session.run("""MATCH (dc:DrugClass) RETURN dc.key AS key, dc.name_fa AS name_fa""")
            for r in res:
                if r["name_fa"]:
                    self._add("drug_class", r["key"], r["name_fa"])

            # Context
            self.entity_cache["context"] = []
            self.exact_map["context"] = {}
            res = session.run("""MATCH (c:Context) RETURN c.key AS key, c.name_fa AS name_fa""")
            for r in res:
                if r["name_fa"]:
                    self._add("context", r["key"], r["name_fa"])

            # Chemical
            self.entity_cache["chemical"] = []
            self.exact_map["chemical"] = {}
            res = session.run("""MATCH (c:Chemical) RETURN c.key AS key, c.name_en AS name_en""")
            for r in res:
                if r["name_en"]:
                    self._add("chemical", r["key"], r["name_en"])

    def _cross_cache_exact_fuzzy(self, m: EntityMention, txt: str) -> Optional[EntityMatch]:
        """
        For drug/chemical/drug_class/interaction_agent:
        search across ALL these caches for exact+fuzzy and return best match.
        """
        search_types = ["drug", "chemical", "drug_class", "interaction_agent"]
    
        best: Optional[EntityMatch] = None
    
        # 1) exact across all
        for et in search_types:
            exact_key = self.exact_map.get(et, {}).get(txt.lower())
            if exact_key:
                em = EntityMatch(text=txt, entity_type=et, key=exact_key, confidence=1.0, position=m.position)
                if (best is None) or (em.confidence > best.confidence):
                    best = em
    
        # if any exact found, it's already max confidence
        if best and best.confidence >= 1.0:
            return best
    
        # 2) fuzzy across all
        for et in search_types:
            mm = EntityMention(text=m.text, entity_type=et, confidence=m.confidence, position=m.position)
            em = self._fuzzy_cache_link(mm, txt)
            if em and ((best is None) or (em.confidence > best.confidence)):
                best = em
    
        return best
    
    def link(self, mentions: List[EntityMention]) -> List[EntityMatch]:
        out: List[EntityMatch] = []
        for m in mentions:
            em = self._link_one(m)
            if em:
                out.append(em)

        # de-dup by (type,key)
        uniq = {}
        for e in out:
            k = (e.entity_type, e.key)
            if k not in uniq or e.confidence > uniq[k].confidence:
                uniq[k] = e
        return sorted(list(uniq.values()), key=lambda x: x.confidence, reverse=True)

    def _link_one(self, m: EntityMention) -> Optional[EntityMatch]:
        txt = self.tp.normalize(m.text)
        if not txt:
            return None
    
        # cross-cache exact+fuzzy for these entity types
        if m.entity_type in ["drug", "chemical", "drug_class", "interaction_agent"]:
            return self._cross_cache_exact_fuzzy(m, txt)
    
        # exact (original logic for other types)
        et = m.entity_type
        exact_key = self.exact_map.get(et, {}).get(txt.lower())
        if exact_key:
            return EntityMatch(text=txt, entity_type=et, key=exact_key, confidence=1.0, position=m.position)
    
        # fuzzy for these (original)
        if et in ["drug", "interaction_agent", "drug_class", "context", "chemical"]:
            return self._fuzzy_cache_link(m, txt)
    
        # vector for these (original)
        if et in ["condition", "adverse_effect", "population"]:
            return self._vector_link(m, txt)
    
        return None



    def _fuzzy_cache_link(self, m: EntityMention, txt: str) -> Optional[EntityMatch]:
        cand = self.entity_cache.get(m.entity_type, [])
        if not cand:
            return None
        names = [c["name"] for c in cand]
        best = process.extractOne(txt, names, scorer=fuzz.WRatio)
        if not best:
            return None
        best_name, best_score, best_idx = best
        if best_score < 75:
            return None
        return EntityMatch(
            text=best_name,
            entity_type=m.entity_type,
            key=cand[best_idx]["key"],
            confidence=float(best_score)/100.0,
            position=m.position
        )

    def _vector_link(self, m: EntityMention, txt: str) -> Optional[EntityMatch]:
        idx_map = {
            "condition": ("condition_name_vec", "key", "COALESCE(node.name_fa, node.key)"),
            "adverse_effect": ("adverse_name_vec", "key", "node.name_fa"),
            "population": ("population_name_vec", "key", "node.name_fa"),
        }
        index_name, key_prop, name_expr = idx_map[m.entity_type]
        emb = self.embedding_model.encode(txt).tolist()

        q = f"""
        CALL db.index.vector.queryNodes($index_name, 1, $embedding)
        YIELD node, score
        RETURN node.{key_prop} AS key, {name_expr} AS name, score
        ORDER BY score DESC
        """
        with self.driver.session() as session:
            rec = session.run(q, index_name=index_name, embedding=emb).single()
            if not rec:
                return None
            score = float(rec["score"])
            if score < 0.80:
                return None
            return EntityMatch(
                text=rec["name"] or txt,
                entity_type=m.entity_type,
                key=rec["key"],
                confidence=score,
                position=m.position
            )


@st.cache_resource
def get_graph_db():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class GraphRAGPipeline:
    def __init__(self, embedder: SentenceTransformer):
        # Embedding model
        self.embedding_model = embedder

        # Neo4j
        self.driver = get_graph_db()

        # Components
        self.tp = PersianTextProcessor()
        self.router = QueryRouter()
        self.linker = EntityLinker(self.driver, self.embedding_model)

        self.cypher_templates = self._load_cypher_templates()

    
    def _load_cypher_templates(self) -> Dict[QuestionType, List[str]]:
        return {
            QuestionType.INDICATION: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:INDICATED_FOR]->(condition:Condition)
                WHERE r.section_key = chunk.section_key
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       condition.name_fa AS condition, condition.key AS condition_key,
                       r.fact_id AS fact_id, r.section_key AS section, r.population AS population
                ORDER BY drug.name_fa, condition.name_fa
                """,
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:TREAT]->(condition:Condition)
                WHERE r.pmid IS NOT NULL
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       condition.name_fa AS condition, r.pmid AS pmid,
                       r.source AS source, r.confidence AS confidence
                ORDER BY toFloat(r.confidence) DESC
                """,
            ],
            QuestionType.ADVERSE_EFFECT: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:HAS_ADVERSE_EFFECT]->(adverse:AdverseEffect)
                WHERE r.section_key = chunk.section_key
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       adverse.name_fa AS adverse_effect, adverse.key AS adverse_key,
                       r.fact_id AS fact_id, r.section_key AS section,
                       r.system AS affected_system, r.population AS affected_population
                ORDER BY drug.name_fa, adverse.name_fa
                """,
            ],
            QuestionType.INTERACTION: [
                # ----------------------------
                # (A) Shortcut edge: Drug -[:INTERACTS_WITH]-> InteractionAgent
                # ----------------------------
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:INTERACTS_WITH]->(agent:InteractionAgent)
                WHERE r.section_key = chunk.section_key
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       coalesce(agent.name_fa, agent.name_en, agent.name_search) AS interacts_with,
                       coalesce(agent.agent_type, head(labels(agent))) AS agent_type,
                       agent.key AS agent_key,
                       r.severity AS severity,
                       r.effect AS effect,
                       r.recommendation AS recommendation,
                       r.fact_id AS fact_id,
                       r.section_key AS section
                ORDER BY CASE r.severity
                    WHEN 'severe' THEN 1 WHEN 'moderate' THEN 2
                    WHEN 'mild' THEN 3 ELSE 4 END
                """,
            
                # ----------------------------
                # (B) Fact-based: Chunk <-EVIDENCE- Fact(type="INTERACTS_WITH") <-HAS_FACT- Drug
                #     Fact -[:ABOUT]-> target (DrugClass/Drug/InteractionAgent/...)
                # ----------------------------
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
            
                MATCH (chunk)<-[:EVIDENCE]-(f:Fact {type:"INTERACTS_WITH"})
                MATCH (drug)-[:HAS_FACT]->(f)
                MATCH (f)-[:ABOUT]->(target)
            
                RETURN coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                       drug.drug_id AS drug_id,
                       coalesce(target.name_fa, target.name_en, target.name_search) AS interacts_with,
                       head(labels(target)) AS agent_type,
                       coalesce(target.key, target.drug_id) AS agent_key,
                       NULL AS severity,
                       coalesce(f.effect, f.object_key, f.note) AS effect,
                       coalesce(f.recommendation, f.action, f.note) AS recommendation,
                       f.fact_id AS fact_id,
                       chunk.section_key AS section
                """,
            ],
            QuestionType.CONTRAINDICATION: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:CONTRAINDICATED_IN]->(target)
                WHERE r.section_key = chunk.section_key
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       labels(target) AS contraindication_type,
                       target.name_fa AS contraindication, target.key AS target_key,
                       r.note AS note, r.population AS population,
                       r.fact_id AS fact_id, r.section_key AS section
                ORDER BY drug.name_fa, contraindication
                """,
            ],
            QuestionType.CAUTION: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r:CAUTION_IN]->(target)
                WHERE r.section_key = chunk.section_key
                RETURN drug.name_fa AS drug_name, drug.drug_id AS drug_id,
                       labels(target) AS caution_type,
                       target.name_fa AS caution_condition, target.key AS target_key,
                       r.note AS note, r.population AS population,
                       r.fact_id AS fact_id, r.section_key AS section
                ORDER BY drug.name_fa, caution_condition
                """,
            ],
            QuestionType.COMPARISON: [
                # ---------------------------------------
                # (A) drug is SUBJECT in the comparison
                # ---------------------------------------
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
            
                MATCH (drug)-[:HAS_COMPARISON_AS_SUBJECT]->(comp:Comparison)
                OPTIONAL MATCH (comp)-[:HAS_SUBJECT]->(subj)
                OPTIONAL MATCH (comp)-[:HAS_OBJECT]->(obj)
            
                RETURN coalesce(subj.name_fa, subj.name_en, subj.name_search) AS subject_name,
                       labels(subj) AS subject_labels,
                       coalesce(obj.name_fa, obj.name_en, obj.name_search) AS object_name,
                       labels(obj) AS object_labels,
                       comp.winner AS winner,
                       comp.comparison_outcome AS outcome,
                       comp.outcome_measure AS measure,
                       comp.certainty AS certainty,
                       comp.pmid AS pmid,
                       comp.rel_subtype AS rel_subtype
                ORDER BY comp.certainty DESC
                """,
            
                # ---------------------------------------
                # (B) drug is OBJECT in the comparison
                # ---------------------------------------
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
            
                MATCH (drug)-[:HAS_COMPARISON_AS_OBJECT]->(comp:Comparison)
                OPTIONAL MATCH (comp)-[:HAS_SUBJECT]->(subj)
                OPTIONAL MATCH (comp)-[:HAS_OBJECT]->(obj)
            
                RETURN coalesce(subj.name_fa, subj.name_en, subj.name_search) AS subject_name,
                       labels(subj) AS subject_labels,
                       coalesce(obj.name_fa, obj.name_en, obj.name_search) AS object_name,
                       labels(obj) AS object_labels,
                       comp.winner AS winner,
                       comp.comparison_outcome AS outcome,
                       comp.outcome_measure AS measure,
                       comp.certainty AS certainty,
                       comp.pmid AS pmid,
                       comp.rel_subtype AS rel_subtype
                ORDER BY comp.certainty DESC
                """,
            ],
            QuestionType.GENERAL_INFO: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                OPTIONAL MATCH (drug)-[:HAS_ASSOCIATION]->(chemical:Chemical)
                RETURN drug.drug_id AS drug_id, drug.name_fa AS name_fa,
                       drug.name_en AS name_en, drug.brand_names AS brand_names,
                       drug.drug_group AS drug_group, drug.chemical_group AS chemical_group,
                       collect(DISTINCT chemical.name_en) AS chemicals,
                       drug.original_url AS source_url
                """,
            ],
            QuestionType.ALTERNATIVE: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                WITH DISTINCT drug
                MATCH (alt:Drug)
                WHERE alt.drug_id <> drug.drug_id AND drug.drug_group IS NOT NULL AND alt.drug_group = drug.drug_group
                RETURN drug.name_fa AS original_drug, drug.drug_group AS drug_class,
                       collect(DISTINCT alt.name_fa)[..10] AS alternatives
                UNION
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                WITH DISTINCT drug
                MATCH (alt:Drug)
                WHERE alt.drug_id <> drug.drug_id AND drug.drug_group IS NULL
                  AND drug.chemical_group IS NOT NULL AND alt.chemical_group = drug.chemical_group
                RETURN drug.name_fa AS original_drug, drug.chemical_group AS drug_class,
                       collect(DISTINCT alt.name_fa)[..10] AS alternatives
                """,
            ],
            QuestionType.POPULATION_SPECIFIC: [
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
                MATCH (drug)-[r]->(target)
                WHERE r.section_key = chunk.section_key
                  AND trim(coalesce(r.population, "")) <> ""
                RETURN coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                       type(r) AS relationship_type,
                       labels(target) AS target_type,
                       coalesce(target.name_fa, target.name_en, target.name_search) AS target_name,
                       r.population AS population,
                       r.note AS note,
                       r.fact_id AS fact_id,
                       r.section_key AS section_key
                ORDER BY drug_name, relationship_type, target_name
                """,
            ],

             QuestionType.CAUSE: [
              # -------------------------------------------------------
              # (A) Fact-based (recommended): Chunk <-EVIDENCE- Fact(type="cause")
              #     Drug -HAS_FACT(type="cause")-> Fact -ABOUT-> Condition
              # -------------------------------------------------------
              """
              MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
              MATCH (chunk)<-[:EVIDENCE]-(f:Fact {type:"cause"})
              MATCH (drug:Drug)-[:HAS_FACT {type:"cause"}]->(f)
              WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
              MATCH (f)-[:ABOUT]->(con:Condition)

              // optional shortcut edge if exists
              OPTIONAL MATCH (drug)-[c:CAUSE]->(con)

              RETURN DISTINCT
                     coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                     drug.drug_id AS drug_id,

                     coalesce(con.name_fa, con.name_en, con.name_search) AS condition_name,
                     con.key AS condition_key,

                     f.fact_id AS fact_id,
                     coalesce(f.pmid, c.pmid) AS pmid,
                     coalesce(f.source, c.source) AS source,
                     coalesce(f.confidence, c.confidence) AS confidence,

                     f.cause_class AS cause_class,
                     f.cause_rel_refined AS cause_rel_refined,
                     f.cause_target_key AS cause_target_key,
                     f.cause_rel_type_raw AS cause_rel_type_raw,

                     f.evidence_quote AS evidence_quote,
                     NULL AS section_key,
                     NULL AS note
              ORDER BY toFloat(coalesce(f.confidence, c.confidence, 0.0)) DESC
              """,

              # -------------------------------------------------------
              # (B) Shortcut-only fallback: Drug -CAUSE-> Condition
              #     (in case Fact/EVIDENCE is missing for some records)
              # -------------------------------------------------------
              """
              MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
              MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
              WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
              MATCH (drug)-[c:CAUSE]->(con:Condition)

              RETURN DISTINCT
                     coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                     drug.drug_id AS drug_id,

                     coalesce(con.name_fa, con.name_en, con.name_search) AS condition_name,
                     con.key AS condition_key,

                     c.fact_id AS fact_id,
                     c.pmid AS pmid,
                     c.source AS source,
                     c.confidence AS confidence,

                     NULL AS cause_class,
                     NULL AS cause_rel_refined,
                     NULL AS cause_target_key,
                     NULL AS cause_rel_type_raw,

                     NULL AS evidence_quote,
                     c.section_key AS section_key,
                     c.note AS note
              ORDER BY toFloat(coalesce(c.confidence, 0.0)) DESC
              """,
              ],
            QuestionType.CONTEXT: [
                # (A) Context via CAUTION_IN / CONTRAINDICATED_IN
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
    
                MATCH (drug)-[r:CAUTION_IN|CONTRAINDICATED_IN]->(ctx:Context)
                WHERE r.section_key = chunk.section_key
    
                RETURN DISTINCT
                       coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                       drug.drug_id AS drug_id,
                       type(r) AS relationship_type,
                       ctx.name_fa AS context,
                       ctx.key AS context_key,
                       r.note AS note,
                       r.population AS population,
                       r.fact_id AS fact_id,
                       r.section_key AS section
                ORDER BY drug_name, relationship_type, context
                """,
    
                # (B) Fact-based: Chunk <-EVIDENCE- Fact -ABOUT-> Context ; Drug -HAS_FACT-> Fact
                """
                MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
                MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
                WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
    
                MATCH (chunk)<-[:EVIDENCE]-(f:Fact)
                MATCH (drug)-[:HAS_FACT]->(f)
                MATCH (f)-[:ABOUT]->(ctx:Context)
    
                RETURN DISTINCT
                       coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                       drug.drug_id AS drug_id,
                       ctx.name_fa AS context,
                       ctx.key AS context_key,
                       f.fact_id AS fact_id,
                       f.type AS fact_type,
                       f.section_key AS section,
                       coalesce(f.qual_note, f.note, f.object_key) AS note
                ORDER BY drug_name, context
                """,
            ],

            QuestionType.CHEMICAL: [
              """
              // -----------------------------
              // CHEMICAL (dedup shortcut vs pubmed associate facts)
              // Key: (drug_id, chemical_key, fact_id)
              // Prefer fact-based (has evidence_quote/relations_display)
              // -----------------------------

              MATCH (chunk:Chunk) WHERE chunk.chunk_id IN $chunk_ids
              MATCH (chunk)<-[:HAS_CHUNK]-(drug:Drug)
              WHERE ($drug_keys IS NULL OR drug.drug_id IN $drug_keys)
              WITH DISTINCT drug

              CALL {
              WITH drug
              // (A) Shortcut association to Chemical
              MATCH (drug)-[a:HAS_ASSOCIATION]->(chem:Chemical)
              RETURN
                     coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                     drug.drug_id AS drug_id,

                     coalesce(chem.name_fa, chem.name_en, chem.name_search) AS chemical_name,
                     chem.key AS chemical_key,

                     a.fact_id AS fact_id,
                     a.pmid AS pmid,
                     a.source AS source,

                     NULL AS fact_type,
                     NULL AS evidence_quote,
                     NULL AS relations_display

              UNION ALL

              WITH drug
              // (B) PubMed associate facts: Drug -HAS_FACT(type=associate)-> Fact(source=pubmed) -ABOUT-> Chemical
              MATCH (drug)-[b:HAS_FACT {type:"associate"}]->(f:Fact {source:"pubmed"})
              MATCH (f)-[:ABOUT]->(chem:Chemical)
              RETURN
                     coalesce(drug.name_fa, drug.name_en, drug.name_search) AS drug_name,
                     drug.drug_id AS drug_id,

                     coalesce(chem.name_fa, chem.name_en, chem.name_search) AS chemical_name,
                     chem.key AS chemical_key,

                     f.fact_id AS fact_id,
                     f.pmid AS pmid,
                     f.source AS source,

                     f.type AS fact_type,
                     f.evidence_quote AS evidence_quote,
                     f.relations_display AS relations_display
              }

              // ---- Dedup: collapse rows with same (drug_id, chemical_key, fact_id)
              WITH
              drug_name, drug_id, chemical_name, chemical_key, fact_id,
              max(pmid) AS pmid,
              max(source) AS source,
              max(fact_type) AS fact_type,
              max(evidence_quote) AS evidence_quote,
              max(relations_display) AS relations_display

              RETURN DISTINCT
              drug_name, drug_id,
              chemical_name, chemical_key,
              fact_id, pmid, source,
              fact_type, evidence_quote, relations_display
              ORDER BY drug_name, chemical_name
              """,
              ],
        }

    
    def analyze_question(self, question: str, debug_llm: bool=False) -> AnalysisResult:
        normalized = self.tp.normalize(question)

        language = self.tp.detect_language(normalized)

        _, parsed = extract_entities_avalai(normalized, debug=debug_llm)

        parsed = postprocess_mentions(parsed)

        mentions = [
            EntityMention(
                text=e["text"],
                entity_type=e["entity_type"],
                confidence=float(e.get("confidence", 0.5)),
                position=(int(e.get("start", 0)), int(e.get("end", 0))),
            )
            for e in parsed.get("entities", [])
        ]

        entities = self.linker.link(mentions)

        # intents may come as strings; normalize to QuestionType Enum
        qtypes_raw = parsed.get("intents")
        qtypes = []
        for x in qtypes_raw:
            if isinstance(x, QuestionType):
                qtypes.append(x)
                continue
        
            if isinstance(x, str):
                key = x.strip()
                if key in QuestionType.__members__:
                    qtypes.append(QuestionType[key])
                    continue
        
                key_up = key.upper()
                if key_up in QuestionType.__members__:
                    qtypes.append(QuestionType[key_up])
                    continue
        
                try:
                    qtypes.append(QuestionType(key))
                except ValueError:
                    pass

        tokens = self.tp.tokenize(normalized)

        keywords = [t for t in tokens if len(t) > 2]

        confidence = self._calc_conf(entities, qtypes)

        analysis = AnalysisResult(
            original_question=question,
            normalized_question=normalized,
            language=language,
            question_types=qtypes,
            entities=entities,
            route_strategy=RouteStrategy.SEMANTIC_FIRST,
            keywords=keywords,
            confidence=confidence,
        )
        analysis.route_strategy = self.router.decide_route(analysis)
        return analysis

    
    def _calc_conf(self, entities: List[EntityMatch], qtypes: List[QuestionType]) -> float:
        if not entities:
            entity_conf = 0.3
        else:
            top = sorted(entities, key=lambda x: x.confidence, reverse=True)[:3]
            entity_conf = sum(e.confidence for e in top) / len(top)
        intent_conf = 0.9 if qtypes and QuestionType.GENERAL_INFO not in qtypes else 0.5
        return 0.6*entity_conf + 0.4*intent_conf

    
    def semantic_search(self, question: str, top_k: int = 5) -> List[str]:
        emb = self.embedding_model.encode(question).tolist()
        with self.driver.session() as session:
            res = session.run("""
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id,
                       score AS score,
                       coalesce(node.embedding_text, node.text) AS chunk_text
                ORDER BY score DESC
            """, index_name=VECTOR_INDEX_NAME, top_k=top_k, embedding=emb)

            rows = [dict(r) for r in res]

            return [r["chunk_id"] for r in rows]

    def graph_intent_search(
        self,
        qtypes: List[QuestionType],
        entities: List[EntityMatch],
        limit: int = 25
    ) -> Dict[str, List[Dict]]:
        """Graph-first retrieval driven by (intent + entities).
    
        Returns a dict keyed by QuestionType.value, each value is a list of rows.
        NOTE: For INTERACTION we return ONLY:
          drug_id, name_fa, name_en, agent_type, qual_effect, chunk_text
        """
        out: Dict[str, List[Dict]] = {qt.value: [] for qt in qtypes}
    
        drug_ids = [e.key for e in entities if e.entity_type == "drug"]
        pop_keys = [e.key for e in entities if e.entity_type == "population"]
        cond_keys = [e.key for e in entities if e.entity_type == "condition"]
        agent_keys = [e.key for e in entities if e.entity_type == "interaction_agent"]
        adv_keys = [e.key for e in entities if e.entity_type == "adverse_effect"]
        class_keys = [e.key for e in entities if e.entity_type == "drug_class"]
        context_keys = [e.key for e in entities if e.entity_type == "context"]
        chem_keys = [e.key for e in entities if e.entity_type == "chemical"]
    
        def _rows(res):
            return [dict(r) for r in res]
    
        with self.driver.session() as session:
            # Fallback for no drug entities
            if not drug_ids:
                infer_q = """
                CALL () {
                  // Condition -> Drug (via Fact ABOUT)
                  WITH $cond_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (n:Condition) WHERE n.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(n)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // Population -> Drug (via Fact ABOUT)
                  WITH $pop_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (n:Population) WHERE n.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(n)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // InteractionAgent -> Drug (via Fact ABOUT)  (only ABOUT to avoid missing rel warnings)
                  WITH $agent_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (a:InteractionAgent) WHERE a.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(a)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // DrugClass -> Drug (via Fact ABOUT)
                  WITH $class_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (n:DrugClass) WHERE n.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(n)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // Context -> Drug (via Fact ABOUT)
                  WITH $context_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (n:Context) WHERE n.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(n)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // Chemical -> Drug (via HAS_ASSOCIATION)
                  WITH $chem_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (c:Chemical) WHERE c.key IN keys
                  MATCH (d:Drug)-[:HAS_ASSOCIATION]->(c)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // Chemical -> Drug (via Fact ABOUT)
                  WITH $chem_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (c:Chemical) WHERE c.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(c)
                  RETURN DISTINCT d.drug_id AS drug_id
                
                  UNION
                
                  // AdverseEffect -> Drug (via Fact ABOUT)
                  WITH $adv_keys AS keys
                  WHERE keys IS NOT NULL AND size(keys) > 0
                  MATCH (n:AdverseEffect) WHERE n.key IN keys
                  MATCH (d:Drug)-[:HAS_FACT]->(:Fact)-[:ABOUT]->(n)
                  RETURN DISTINCT d.drug_id AS drug_id
                }
                RETURN DISTINCT drug_id
                LIMIT $lim
            """
                infer_res = session.run(
                    infer_q,
                    cond_keys=cond_keys if cond_keys else None,
                    pop_keys=pop_keys if pop_keys else None,
                    agent_keys=agent_keys if agent_keys else None,
                    class_keys=class_keys if class_keys else None,
                    context_keys=context_keys if context_keys else None,
                    chem_keys=chem_keys if chem_keys else None,
                    adv_keys=adv_keys if adv_keys else None,
                    lim=max(limit * 10, 50),
                )
                drug_ids = [r["drug_id"] for r in infer_res if r.get("drug_id")]
    
            if not drug_ids:
                return out
    
            # ---------------------------------------------------------
            # MAIN: per-intent queries
            # ---------------------------------------------------------
            for qt in qtypes:
                # -----------------------------
                # ADVERSE EFFECT
                # -----------------------------
                if qt == QuestionType.ADVERSE_EFFECT:
                    q = """
                    MATCH (d:Drug)-[r:HAS_ADVERSE_EFFECT]->(n:AdverseEffect)
                    WHERE d.drug_id IN $drug_ids
                      AND ($adv_keys IS NULL OR n.key IN $adv_keys)
                    OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
                    WHERE r.fact_id IS NOT NULL AND f.fact_id = r.fact_id
                    OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS drug_name_fa,
                      d.name_en AS drug_name_en,
                      'HAS_ADVERSE_EFFECT' AS rel_type,
                      n.key AS target_key,
                      n.name_fa AS target_name_fa,
                      r.fact_id AS fact_id,
                      c.chunk_id AS chunk_id,
                      coalesce(c.embedding_text, c.text) AS chunk_text,
                      coalesce(r.section_key, f.section_key) AS section_key,
                      r.population AS rel_population
                    ORDER BY d.name_fa, n.name_fa
                    LIMIT $lim
                    """
                    res = session.run(q, drug_ids=drug_ids, adv_keys=adv_keys if adv_keys else None, lim=limit)
                    out[qt.value] = _rows(res)
                    continue
    
                # -----------------------------
                # CONTRAINDICATION / CAUTION
                # -----------------------------
                if qt in (QuestionType.CONTRAINDICATION, QuestionType.CAUTION):
                    if qt == QuestionType.CONTRAINDICATION:
                        rels = ["CONTRAINDICATED_IN", "CAUTION_IN"]
                        fact_types = ["contraindication", "caution"]
                    else:
                        rels = ["CAUTION_IN"]
                        fact_types = ["caution"]
    
                    targets = []
                    if pop_keys:
                        targets.append(("Population", pop_keys))
                    if cond_keys:
                        targets.append(("Condition", cond_keys))
                    if class_keys:
                        targets.append(("DrugClass", class_keys))
                    if agent_keys:
                        targets.append(("InteractionAgent", agent_keys))
                    if context_keys:
                        targets.append(("Context", context_keys))
    
                    if not targets:
                        targets = [
                            ("Population", None),
                            ("Condition", None),
                            ("DrugClass", None),
                            ("InteractionAgent", None),
                            ("Context", None),
                        ]
    
                    rows_all: List[Dict] = []
                    for (label, keys) in targets:
                        q = f"""
                        // (A) Shortcut edges: CONTRAINDICATED_IN / CAUTION_IN
                        MATCH (d:Drug)-[r]->(n:{label})
                        WHERE d.drug_id IN $drug_ids
                          AND type(r) IN $rels
                          AND ($keys IS NULL OR n.key IN $keys)
                        OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
                        WHERE r.fact_id IS NOT NULL AND f.fact_id = r.fact_id
                        OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                        RETURN
                          d.drug_id AS drug_id,
                          d.name_fa AS drug_name_fa,
                          d.name_en AS drug_name_en,
                          type(r) AS rel_type,
                          n.key AS target_key,
                          coalesce(n.name_fa, n.name_search, n.key) AS target_name_fa,
                          r.fact_id AS fact_id,
                          c.chunk_id AS chunk_id,
                          coalesce(c.embedding_text, c.text) AS chunk_text,
                          coalesce(r.section_key, f.section_key) AS section_key,
                          coalesce(r.population, f.qual_population) AS rel_population,
                          coalesce(r.note, f.note) AS rel_note
    
                        UNION
    
                        // (B) Fact-based fallback: HAS_FACT(type in contra/caution) -> ABOUT -> target
                        MATCH (d:Drug)-[:HAS_FACT]->(f:Fact)
                        WHERE d.drug_id IN $drug_ids
                          AND (f.type IN $fact_types OR f.type IS NULL)
                        MATCH (f)-[:ABOUT]->(n:{label})
                        WHERE ($keys IS NULL OR n.key IN $keys)
                        OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                        RETURN
                          d.drug_id AS drug_id,
                          d.name_fa AS drug_name_fa,
                          d.name_en AS drug_name_en,
                          'HAS_FACT' AS rel_type,
                          n.key AS target_key,
                          coalesce(n.name_fa, n.name_search, n.key) AS target_name_fa,
                          f.fact_id AS fact_id,
                          c.chunk_id AS chunk_id,
                          coalesce(c.embedding_text, c.text) AS chunk_text,
                          f.section_key AS section_key,
                          f.qual_population AS rel_population,
                          f.note AS rel_note
    
                        ORDER BY drug_name_fa, target_name_fa
                        LIMIT $lim
                        """
                        res = session.run(
                            q,
                            drug_ids=drug_ids,
                            keys=keys,
                            rels=rels,
                            fact_types=fact_types,
                            lim=limit
                        )
                        rows_all.extend(_rows(res))
    
                    out[qt.value] = rows_all[:limit]
                    continue
    
                # -----------------------------
                # INTERACTION  (ONLY return requested properties)
                # drug_id, name_fa, name_en, agent_type, qual_effect, chunk_text
                # -----------------------------
                if qt == QuestionType.INTERACTION:
                    q = """
                    // (A) Shortcut edge if exists + evidence
                    MATCH (d:Drug)
                    WHERE d.drug_id IN $drug_ids
                    OPTIONAL MATCH (d)-[r:INTERACTS_WITH]->(a:InteractionAgent)
                    WHERE ($agent_keys IS NULL OR (a IS NOT NULL AND a.key IN $agent_keys))
                    OPTIONAL MATCH (d)-[:HAS_FACT]->(f1:Fact)
                    WHERE r.fact_id IS NOT NULL AND f1.fact_id = r.fact_id
                    OPTIONAL MATCH (f1)-[:EVIDENCE]->(c1:Chunk)
                    WITH d,
                         a,
                         coalesce(f1.qual_effect, r.effect) AS qual_effect,
                         coalesce(c1.embedding_text, c1.text) AS chunk_text
                    WHERE a IS NOT NULL
    
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS name_fa,
                      d.name_en AS name_en,
                      a.agent_type AS agent_type,
                      qual_effect AS qual_effect,
                      chunk_text AS chunk_text
    
                    UNION
    
                    // (B) Fact-based fallback (higher recall)
                    MATCH (d:Drug)-[:HAS_FACT]->(f:Fact)
                    WHERE d.drug_id IN $drug_ids
                      AND (f.type = "interaction" OR f.type IS NULL)
                    OPTIONAL MATCH (f)-[:ABOUT|HAS_AGENT|INVOLVES_AGENT]->(a:InteractionAgent)
                    WHERE ($agent_keys IS NULL OR (a IS NOT NULL AND a.key IN $agent_keys))
                    OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                    WITH d, f, a, c
                    WHERE a IS NOT NULL
    
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS name_fa,
                      d.name_en AS name_en,
                      a.agent_type AS agent_type,
                      f.qual_effect AS qual_effect,
                      coalesce(c.embedding_text, c.text) AS chunk_text
    
                    ORDER BY name_fa
                    LIMIT $lim
                    """
                    res = session.run(
                        q,
                        drug_ids=drug_ids,
                        agent_keys=agent_keys if agent_keys else None,
                        lim=limit
                    )
                    out[qt.value] = _rows(res)
                    continue
    
                # -----------------------------
                # INDICATION (INDICATED_FOR / TREAT)
                # -----------------------------
                if qt == QuestionType.INDICATION:
                    q = """
                    MATCH (d:Drug)-[r:INDICATED_FOR|TREAT]->(n:Condition)
                    WHERE d.drug_id IN $drug_ids
                      AND ($cond_keys IS NULL OR n.key IN $cond_keys)
                    OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
                    WHERE r.fact_id IS NOT NULL AND f.fact_id = r.fact_id
                    OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS drug_name_fa,
                      d.name_en AS drug_name_en,
                      type(r) AS rel_type,
                      n.key AS target_key,
                      n.name_fa AS target_name_fa,
                      r.fact_id AS fact_id,
                      c.chunk_id AS chunk_id,
                      coalesce(c.embedding_text, c.text) AS chunk_text,
                      coalesce(r.section_key, f.section_key) AS section_key,
                      r.population AS rel_population,
                      r.source AS source,
                      r.pmid AS pmid,
                      r.confidence AS rel_confidence
                    ORDER BY d.name_fa, n.name_fa
                    LIMIT $lim
                    """
                    res = session.run(q, drug_ids=drug_ids, cond_keys=cond_keys if cond_keys else None, lim=limit)
                    out[qt.value] = _rows(res)
                    continue
    
                # -----------------------------
                # CAUSE
                # -----------------------------
                if qt == QuestionType.CAUSE:
                    q = """
                    MATCH (d:Drug)-[r:CAUSE]->(n:Condition)
                    WHERE d.drug_id IN $drug_ids
                      AND ($cond_keys IS NULL OR n.key IN $cond_keys)
                    OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
                    WHERE r.fact_id IS NOT NULL AND f.fact_id = r.fact_id
                    OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS drug_name_fa,
                      d.name_en AS drug_name_en,
                      'CAUSE' AS rel_type,
                      n.key AS target_key,
                      n.name_fa AS target_name_fa,
                      r.fact_id AS fact_id,
                      c.chunk_id AS chunk_id,
                      coalesce(c.embedding_text, c.text) AS chunk_text,
                      r.source AS source,
                      r.pmid AS pmid,
                      r.confidence AS rel_confidence
                    ORDER BY d.name_fa, n.name_fa
                    LIMIT $lim
                    """
                    res = session.run(q, drug_ids=drug_ids, cond_keys=cond_keys if cond_keys else None, lim=limit)
                    out[qt.value] = _rows(res)
                    continue
    
                # -----------------------------
                # CHEMICAL association
                # -----------------------------
                if qt == QuestionType.CHEMICAL:
                    q = """
                    MATCH (d:Drug)-[r:HAS_ASSOCIATION]->(n:Chemical)
                    WHERE d.drug_id IN $drug_ids
                      AND ($chem_keys IS NULL OR n.key IN $chem_keys)
                    OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
                    WHERE r.fact_id IS NOT NULL AND f.fact_id = r.fact_id
                    OPTIONAL MATCH (f)-[:EVIDENCE]->(c:Chunk)
                    RETURN
                      d.drug_id AS drug_id,
                      d.name_fa AS drug_name_fa,
                      d.name_en AS drug_name_en,
                      'HAS_ASSOCIATION' AS rel_type,
                      n.key AS target_key,
                      coalesce(n.name_fa, n.name_en, n.key) AS target_name_fa,
                      r.fact_id AS fact_id,
                      c.chunk_id AS chunk_id,
                      coalesce(c.embedding_text, c.text) AS chunk_text,
                      r.source AS source,
                      r.pmid AS pmid,
                      r.confidence AS rel_confidence
                    ORDER BY d.name_fa, target_name_fa
                    LIMIT $lim
                    """
                    res = session.run(q, drug_ids=drug_ids, chem_keys=chem_keys if chem_keys else None, lim=limit)
                    out[qt.value] = _rows(res)
                    continue
    
    
        return out


    def expand_with_cypher(self, chunk_ids: List[str], qtypes: List[QuestionType], entities: List[EntityMatch]) -> Dict:
        results = {"chunks": chunk_ids, "graph_data": {}}

        drug_keys = [e.key for e in entities if e.entity_type == "drug"]
        params = {"chunk_ids": chunk_ids, "drug_keys": drug_keys if drug_keys else None}

        with self.driver.session() as session:
            chunk_res = session.run("""
                MATCH (chunk:Chunk)
                WHERE chunk.chunk_id IN $chunk_ids
                RETURN chunk.chunk_id AS chunk_id, chunk.text AS text,
                       chunk.title_fa AS title, chunk.section_key AS section
            """, chunk_ids=chunk_ids)
            results["chunk_texts"] = [dict(r) for r in chunk_res]

            for qt in qtypes:
                qt_key = qt.value if hasattr(qt, "value") else str(qt)
            
                rows = []
                for tpl in self.cypher_templates.get(qt, []):
                    try:
                        qr = session.run(tpl, **params)
                        rows += [dict(r) for r in qr]
                    except Exception as e:
                        print(f"[Cypher error] {qt_key}: {e}")
            
                results["graph_data"][qt_key] = unique_dicts(rows)
        return results

    
    def execute(self, question: str, top_k: int = 5, debug_llm: bool=False) -> Dict:
        analysis = self.analyze_question(question, debug_llm=debug_llm)

        # Graph-first retrieval driven by (intent + entities)
        graph_rows = self.graph_intent_search(analysis.question_types, analysis.entities, limit=top_k*5)
        graph_chunk_ids: List[str] = []
        for _qt, _rows in graph_rows.items():
            for _r in _rows:
                _cid = _r.get("chunk_id")
                if _cid:
                    graph_chunk_ids.append(_cid)
        graph_chunk_ids = list(set(graph_chunk_ids))

        if analysis.route_strategy == RouteStrategy.ENTITY_FIRST:
            chunk_ids = self.semantic_search(analysis.normalized_question, top_k=top_k)

        elif analysis.route_strategy == RouteStrategy.SEMANTIC_FIRST:
            chunk_ids = self.semantic_search(analysis.normalized_question, top_k=top_k)

        else:  # HYBRID
            chunk_ids = list(set(
                self.semantic_search(analysis.normalized_question, top_k=top_k)
            ))

        # Merge evidence chunks from graph-first retrieval into chunk candidates
        chunk_ids = list(set(chunk_ids + graph_chunk_ids))

        chunk_ids = list(set(chunk_ids))[: top_k * 2]

        out = self.expand_with_cypher(chunk_ids, analysis.question_types, analysis.entities)
        # Attach graph-first rows (debuggable structured evidence)
        out.setdefault("graph_data", {})
        out["graph_data"]["intent_graph"] = graph_rows
        out["analysis"] = {
            "language": analysis.language,
            "intents": [x.value for x in analysis.question_types],
            "entities": [{"text": e.text, "type": e.entity_type, "key": e.key, "confidence": e.confidence} for e in analysis.entities],
            "route": analysis.route_strategy.value,
            "confidence": analysis.confidence,
        }
        return out


    def close(self):
        self.driver.close()
