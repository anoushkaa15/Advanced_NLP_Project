"""Advanced multilingual clinical symptom routing chatbot.

Key upgrades:
- True BERT embedding backend (mean pooled transformer hidden states)
- Hybrid retrieval (dense BERT + TF-IDF lexical backoff)
- Embedding cache on disk
- Confidence-gated multi-turn clarification loop
- Lightweight clinical NER normalization and severity/duration weighting
- Optional OpenAI validator/reranker over top candidates
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

try:
    from openai import OpenAI
except Exception:  # optional dependency
    OpenAI = None

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
SIMILARITY_THRESHOLD = 0.66
TOP_N_MATCHES = 6
MAX_CLARIFICATION_ROUNDS = 3
EMBEDDING_CACHE_PATH = "symptom_sentence_bert_embeddings.npy"
OPENAI_MODEL = os.getenv("OPENAI_VALIDATOR_MODEL", "gpt-4o-mini")

SEVERITY_WEIGHTS = {"mild": 0.96, "moderate": 1.0, "severe": 1.08}
SYMPTOM_SYNONYMS = {
    "stomach ache": "abdominal pain",
    "pet dard": "abdominal pain",
    "ulti": "vomiting",
    "saans phoolna": "shortness of breath",
    "chakkar": "dizziness",
    "bukhar": "fever",
    "runny nose": "rhinorrhea",
    "cold": "common cold",
}
ACUTE_KEYWORDS = {"acute", "infection", "viral", "dengue", "cholera", "pneumonia", "flu"}
CHRONIC_KEYWORDS = {"chronic", "arthritis", "asthma", "diabetes", "hypertension", "psoriasis"}

ACUTE_KEYWORDS = {"acute", "infection", "viral", "dengue", "cholera", "pneumonia"}
CHRONIC_KEYWORDS = {"chronic", "arthritis", "asthma", "diabetes", "hypertension", "psoriasis"}


@dataclass
class SessionState:
    user_language: str = "en"
    turns: List[str] = field(default_factory=list)
    clarification_rounds: int = 0

@dataclass
class SessionState:
    user_language: str = "en"
    turns: List[str] = field(default_factory=list)
    clarification_rounds: int = 0


class LanguageProcessor:
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "en"


class ClinicalNER:
    severity_patterns = {
        "severe": r"\b(severe|intense|extreme|bahut zyada)\b",
        "mild": r"\b(mild|light|slight|thora|thoda)\b",
        "moderate": r"\b(moderate)\b",
    }
    duration_pattern = re.compile(
        r"\b(for\s+)?(\d+)\s*(day|days|week|weeks|month|months)\b|\b(since\s+last\s+month)\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> Dict[str, Optional[str]]:
        lowered = text.lower()
        severity = next((k for k, p in self.severity_patterns.items() if re.search(p, lowered)), None)
        duration_match = self.duration_pattern.search(lowered)
        duration = duration_match.group(0) if duration_match else None
        normalized_text = lowered
        for src, tgt in SYMPTOM_SYNONYMS.items():
            normalized_text = re.sub(rf"\b{re.escape(src)}\b", tgt, normalized_text)
        return {"normalized_text": normalized_text, "severity": severity, "duration": duration}


class BertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            emb = self._mean_pool(self.model(**tok).last_hidden_state, tok["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
        return np.vstack(out)


class OpenAIClinicalValidator:
    def __init__(self):
        self.enabled = OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
        self.client = OpenAI() if self.enabled else None

    def rerank(self, user_query: str, candidates: List[Dict]) -> List[Dict]:
        if not self.enabled or not candidates:
            return candidates

        payload = [{"index": i, "disease": c["disease"], "department": c["department"]} for i, c in enumerate(candidates)]
        prompt = (
            "You are a clinical triage consistency checker. "
            "Given a symptom query and candidate disease-department pairs, return JSON with: "
            "{valid_order:[indices best-to-worst], reject:[indices], rationale:'short'}. "
            "Reject candidates that are implausible for symptoms.\n"
            f"Symptoms: {user_query}\nCandidates: {json.dumps(payload)}"
        )

        try:
            resp = self.client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=0,
            )
            text = resp.output_text
            data = json.loads(text)
            reject = set(data.get("reject", []))
            order = [i for i in data.get("valid_order", []) if i not in reject and 0 <= i < len(candidates)]
            leftovers = [i for i in range(len(candidates)) if i not in reject and i not in order]
            final = order + leftovers
            reranked = [dict(candidates[i]) for i in final]
            for c in reranked:
                c["validated_by_openai"] = True
            return reranked
        except Exception:
            return candidates


class SymptomMatcher:
    def __init__(self, dataset_path: str, embedding_cache_path: str = EMBEDDING_CACHE_PATH):
        self.df = pd.read_csv(dataset_path)
        required = {"input_text", "disease", "department"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(f"Dataset must include columns: {required}")

        self.embedder = BertEmbedder()
        self.embedding_cache_path = embedding_cache_path
        self.embeddings = self._load_or_build_embeddings()

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["input_text"].astype(str))

    def _load_or_build_embeddings(self) -> np.ndarray:
        if os.path.exists(self.embedding_cache_path):
            return np.load(self.embedding_cache_path)
        embs = self.embedder.encode(self.df["input_text"].astype(str).tolist())
        np.save(self.embedding_cache_path, embs)
        return embs

    def _apply_attribute_adjustments(self, scores: np.ndarray, ner_info: Dict[str, Optional[str]]) -> np.ndarray:
        out = scores.copy()
        sev = ner_info.get("severity")
        if sev in SEVERITY_WEIGHTS:
            out *= SEVERITY_WEIGHTS[sev]
        duration = (ner_info.get("duration") or "").lower()
        if "month" in duration:
            acute_mask = self.df["disease"].str.lower().apply(lambda d: any(k in d for k in ACUTE_KEYWORDS))
            out[acute_mask.values] *= 0.90
        elif "day" in duration:
            chronic_mask = self.df["disease"].str.lower().apply(lambda d: any(k in d for k in CHRONIC_KEYWORDS))
            out[chronic_mask.values] *= 0.95
        return out

    def find_matches(self, query: str, ner_info: Dict[str, Optional[str]]) -> List[Dict]:
        dense_scores = cosine_similarity(self.embedder.encode([query]), self.embeddings)[0]
        sparse_scores = cosine_similarity(self.tfidf.transform([query]), self.tfidf_matrix)[0]
        scores = self._apply_attribute_adjustments(0.88 * dense_scores + 0.12 * sparse_scores, ner_info)
        top_idx = np.argsort(scores)[-TOP_N_MATCHES:][::-1]

        # de-duplicate by disease+department
        seen = set()
        matches = []
        for i in top_idx:
            key = (self.df.iloc[i]["disease"], self.df.iloc[i]["department"])
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "disease": key[0],
                "department": key[1],
                "similarity": float(scores[i]),
                "matched_text": self.df.iloc[i]["input_text"],
            })
            if len(matches) >= 3:
                break
        return matches


class MedicalChatbot:
    def __init__(self, dataset_path: str):
        self.lang_processor = LanguageProcessor()
        self.ner = ClinicalNER()
        self.matcher = SymptomMatcher(dataset_path)
        self.validator = OpenAIClinicalValidator()
        self.session = SessionState()

    def _build_followup_question(self, last_match: Optional[Dict]) -> str:
        if not last_match:
            return "Please share symptom severity and how long you have had it."
        return f"Can you add severity and duration details for symptoms like '{last_match['matched_text']}'?"

    def _format_prediction(self, matches: List[Dict]) -> str:
        top = matches[0]
        lines = [
            "Most likely clinical routing:",
            f"1) {top['disease']} → {top['department']} (confidence {top['similarity']:.1%})",
        ]
        for i, m in enumerate(matches[1:3], 2):
            lines.append(f"{i}) {m['disease']} → {m['department']} ({m['similarity']:.1%})")
        if matches and matches[0].get("validated_by_openai"):
            lines.append("✅ Top candidates were plausibility-checked by OpenAI.")
        lines.append("⚠️ Not a diagnosis. Please consult a medical professional.")
        return "\n".join(lines)

    def process_turn(self, user_input: str) -> Tuple[str, bool]:
        if not self.session.turns:
            self.session.user_language = self.lang_processor.detect_language(user_input)

        self.session.turns.append(user_input)
        context = " ".join(self.session.turns)
        ner_info = self.ner.extract(context)

        matches = self.matcher.find_matches(ner_info["normalized_text"], ner_info)
        matches = self.validator.rerank(context, matches)
        score = matches[0]["similarity"] if matches else 0.0

        if score >= SIMILARITY_THRESHOLD or self.session.clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
            self.session = SessionState()
            return self._format_prediction(matches), True

        self.session.clarification_rounds += 1
        return self._build_followup_question(matches[0] if matches else None), False

    def chat(self):
        print("🩺 Advanced Multilingual Clinical Router (BERT + OpenAI Validator)")
        print("Type 'exit' to quit.\n")
        while True:
            text = input("You: ").strip()
            if text.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if not text:
                continue
            reply, done = self.process_turn(text)
            print(("Bot" if done else "Bot (clarification)") + f": {reply}\n")


if __name__ == "__main__":
    path = "symptom_sentence_dataset_with_department.csv"
    if not os.path.exists(path):
        print(f"❌ Dataset not found: {path}")
    else:
        MedicalChatbot(path).chat()
