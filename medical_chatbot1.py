"""
Multilingual Clinical Symptom Routing Chatbot
---------------------------------------------
Enhancements implemented:
- MiniLM multilingual semantic embeddings (replacing TF-IDF behavior)
- Optional precomputed embedding cache (.npy)
- Confidence-gated multi-turn clarification loop
- Lightweight clinical NER + symptom normalization
- Attribute-aware score weighting (severity + duration)
- Optional Google Gemini follow-up question generation hook
"""

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.65
TOP_N_MATCHES = 5
MAX_CLARIFICATION_ROUNDS = 3
EMBEDDING_CACHE_PATH = "symptom_sentence_embeddings.npy"

SEVERITY_WEIGHTS = {
    "mild": 0.96,
    "moderate": 1.00,
    "severe": 1.07,
}

# Minimal heuristic disease flags for duration/severity filtering
ACUTE_KEYWORDS = {
    "acute",
    "infection",
    "viral",
    "food poisoning",
    "influenza",
    "dengue",
    "cholera",
    "pneumonia",
}

CHRONIC_KEYWORDS = {
    "chronic",
    "arthritis",
    "asthma",
    "diabetes",
    "hypertension",
    "psoriasis",
    "migraine",
}

SYMPTOM_SYNONYMS = {
    "stomach ache": "abdominal pain",
    "pet dard": "abdominal pain",
    "ulti": "vomiting",
    "saans phoolna": "shortness of breath",
    "chakkar": "dizziness",
    "bukhar": "fever",
}


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
    """Lightweight rule-based NER for symptom, severity, and duration extraction."""

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

        severity = None
        for label, pattern in self.severity_patterns.items():
            if re.search(pattern, lowered, re.IGNORECASE):
                severity = label
                break

        duration_match = self.duration_pattern.search(lowered)
        duration = duration_match.group(0) if duration_match else None

        normalized_text = lowered
        for src, tgt in SYMPTOM_SYNONYMS.items():
            normalized_text = re.sub(rf"\b{re.escape(src)}\b", tgt, normalized_text)

        return {
            "normalized_text": normalized_text,
            "severity": severity,
            "duration": duration,
        }


class SymptomMatcher:
    def __init__(self, dataset_path: str, embedding_cache_path: str = EMBEDDING_CACHE_PATH):
        self.df = pd.read_csv(dataset_path)

        required_cols = {"input_text", "disease", "department"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        self.model = SentenceTransformer(MODEL_NAME)
        self.embedding_cache_path = embedding_cache_path
        self.embeddings = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        if os.path.exists(self.embedding_cache_path):
            return np.load(self.embedding_cache_path)

        embeddings = self.model.encode(
            self.df["input_text"].astype(str).tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(self.embedding_cache_path, embeddings)
        return embeddings

    def _apply_attribute_adjustments(
        self,
        base_scores: np.ndarray,
        ner_info: Dict[str, Optional[str]],
    ) -> np.ndarray:
        adjusted = base_scores.copy()

        severity = ner_info.get("severity")
        if severity in SEVERITY_WEIGHTS:
            adjusted = adjusted * SEVERITY_WEIGHTS[severity]

        duration = (ner_info.get("duration") or "").lower()
        if "month" in duration:
            # Penalize likely acute diseases for long durations
            acute_mask = self.df["disease"].str.lower().apply(
                lambda d: any(k in d for k in ACUTE_KEYWORDS)
            )
            adjusted[acute_mask.values] *= 0.90
        elif "day" in duration and "30" not in duration:
            # Slight penalty for chronic-only diseases on short duration
            chronic_mask = self.df["disease"].str.lower().apply(
                lambda d: any(k in d for k in CHRONIC_KEYWORDS)
            )
            adjusted[chronic_mask.values] *= 0.95

        return adjusted

    def find_matches(self, query: str, ner_info: Dict[str, Optional[str]]) -> List[Dict]:
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        similarities = self._apply_attribute_adjustments(similarities, ner_info)

        top_indices = np.argsort(similarities)[-TOP_N_MATCHES:][::-1]

        matches = []
        for idx in top_indices:
            score = float(similarities[idx])
            matches.append(
                {
                    "disease": self.df.iloc[idx]["disease"],
                    "department": self.df.iloc[idx]["department"],
                    "similarity": score,
                    "matched_text": self.df.iloc[idx]["input_text"],
                }
            )
        return matches


class MedicalChatbot:
    def __init__(self, dataset_path: str):
        self.lang_processor = LanguageProcessor()
        self.ner = ClinicalNER()
        self.matcher = SymptomMatcher(dataset_path)
        self.session = SessionState()

    def _build_followup_question(self, last_match: Optional[Dict], user_context: str) -> str:
        # Gemini integration hook: replace this with API call in production.
        if last_match:
            return (
                f"I need one more detail to route you better. "
                f"Besides '{last_match['matched_text']}', are you also experiencing "
                f"any related symptoms, and for how long?"
            )
        return "Could you share symptom severity (mild/moderate/severe) and duration?"

    def _format_prediction(self, matches: List[Dict]) -> str:
        top = matches[0]
        response = [
            "Based on your symptoms, here are the most likely options:",
            f"1. {top['disease']} ({top['department']}) - Confidence: {top['similarity']:.1%}",
        ]
        for i, m in enumerate(matches[1:3], start=2):
            response.append(
                f"{i}. {m['disease']} ({m['department']}) - Confidence: {m['similarity']:.1%}"
            )
        response.append("⚠️ This is not a diagnosis. Please consult a qualified doctor.")
        return "\n".join(response)

    def process_turn(self, user_input: str) -> Tuple[str, bool]:
        if not self.session.turns:
            self.session.user_language = self.lang_processor.detect_language(user_input)

        self.session.turns.append(user_input)
        combined_context = " ".join(self.session.turns)

        ner_info = self.ner.extract(combined_context)
        normalized_query = ner_info["normalized_text"]
        matches = self.matcher.find_matches(normalized_query, ner_info)

        top_score = matches[0]["similarity"] if matches else 0.0

        if top_score >= SIMILARITY_THRESHOLD or self.session.clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
            self.session = SessionState()
            return self._format_prediction(matches), True

        self.session.clarification_rounds += 1
        question = self._build_followup_question(matches[0] if matches else None, combined_context)
        return question, False

    def chat(self):
        print("=" * 60)
        print("🩺 MULTILINGUAL CLINICAL ROUTING CHATBOT")
        print("=" * 60)
        print("Describe symptoms. Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            try:
                reply, completed = self.process_turn(user_input)
                prefix = "Bot" if completed else "Bot (clarification)"
                print(f"\n{prefix}: {reply}\n" + "-" * 60)
            except Exception as exc:
                print(f"⚠️ Error: {exc}")


if __name__ == "__main__":
    DATASET_PATH = "symptom_sentence_dataset_with_department.csv"
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
    else:
        MedicalChatbot(DATASET_PATH).chat()
