"""Advanced multilingual clinical symptom routing chatbot.

Key upgrades:
- True BERT embedding backend (mean pooled transformer hidden states)
- Hybrid retrieval (dense BERT + TF-IDF lexical backoff)
- Embedding cache on disk
- Confidence-gated multi-turn clarification loop
- Lightweight clinical NER normalization and severity/duration weighting
"""

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

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
SIMILARITY_THRESHOLD = 0.66
TOP_N_MATCHES = 5
MAX_CLARIFICATION_ROUNDS = 3
EMBEDDING_CACHE_PATH = "symptom_sentence_bert_embeddings.npy"

SEVERITY_WEIGHTS = {"mild": 0.96, "moderate": 1.0, "severe": 1.08}

SYMPTOM_SYNONYMS = {
    "stomach ache": "abdominal pain",
    "pet dard": "abdominal pain",
    "ulti": "vomiting",
    "saans phoolna": "shortness of breath",
    "chakkar": "dizziness",
    "bukhar": "fever",
}

ACUTE_KEYWORDS = {"acute", "infection", "viral", "dengue", "cholera", "pneumonia"}
CHRONIC_KEYWORDS = {"chronic", "arthritis", "asthma", "diabetes", "hypertension", "psoriasis"}


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
        severity = next(
            (k for k, p in self.severity_patterns.items() if re.search(p, lowered, re.IGNORECASE)),
            None,
        )
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
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**tokens)
            emb = self._mean_pool(out.last_hidden_state, tokens["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vectors.append(emb.cpu().numpy())
        return np.vstack(vectors)


class SymptomMatcher:
    def __init__(self, dataset_path: str, embedding_cache_path: str = EMBEDDING_CACHE_PATH):
        self.df = pd.read_csv(dataset_path)
        required = {"input_text", "disease", "department"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

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
        dense_q = self.embedder.encode([query])
        dense_scores = cosine_similarity(dense_q, self.embeddings)[0]

        sparse_q = self.tfidf.transform([query])
        sparse_scores = cosine_similarity(sparse_q, self.tfidf_matrix)[0]

        # Hybrid score: dense semantics + lexical precision
        scores = 0.85 * dense_scores + 0.15 * sparse_scores
        scores = self._apply_attribute_adjustments(scores, ner_info)

        top_idx = np.argsort(scores)[-TOP_N_MATCHES:][::-1]
        return [
            {
                "disease": self.df.iloc[i]["disease"],
                "department": self.df.iloc[i]["department"],
                "similarity": float(scores[i]),
                "matched_text": self.df.iloc[i]["input_text"],
            }
            for i in top_idx
        ]


class MedicalChatbot:
    def __init__(self, dataset_path: str):
        self.lang_processor = LanguageProcessor()
        self.ner = ClinicalNER()
        self.matcher = SymptomMatcher(dataset_path)
        self.session = SessionState()

    def _build_followup_question(self, last_match: Optional[Dict]) -> str:
        if not last_match:
            return "Please share symptom severity and how long you have had it."
        return (
            f"To improve routing confidence: for '{last_match['matched_text']}', "
            "do you also have related symptoms, and duration/severity details?"
        )

    def _format_prediction(self, matches: List[Dict]) -> str:
        top = matches[0]
        lines = [
            "Most likely clinical routing:",
            f"1) {top['disease']} → {top['department']} (confidence {top['similarity']:.1%})",
        ]
        for i, m in enumerate(matches[1:3], 2):
            lines.append(f"{i}) {m['disease']} → {m['department']} ({m['similarity']:.1%})")
        lines.append("⚠️ Not a diagnosis. Please consult a medical professional.")
        return "\n".join(lines)

    def process_turn(self, user_input: str) -> Tuple[str, bool]:
        if not self.session.turns:
            self.session.user_language = self.lang_processor.detect_language(user_input)

        self.session.turns.append(user_input)
        context = " ".join(self.session.turns)
        ner_info = self.ner.extract(context)

        matches = self.matcher.find_matches(ner_info["normalized_text"], ner_info)
        score = matches[0]["similarity"] if matches else 0.0

        if score >= SIMILARITY_THRESHOLD or self.session.clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
            self.session = SessionState()
            return self._format_prediction(matches), True

        self.session.clarification_rounds += 1
        return self._build_followup_question(matches[0] if matches else None), False

    def chat(self):
        print("🩺 Advanced Multilingual Clinical Router (BERT)")
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
