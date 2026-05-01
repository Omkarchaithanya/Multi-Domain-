import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DOMAINS = ("hackerrank", "claude", "visa")
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".json", ".csv"}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_']+", text.lower())


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        data = data.strip()
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


class SimpleBM25:
    """Small BM25 fallback used when rank_bm25 is unavailable."""

    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs: List[Counter] = [Counter(doc) for doc in tokenized_corpus]
        df: Dict[str, int] = defaultdict(int)
        for doc in tokenized_corpus:
            for term in set(doc):
                df[term] += 1
        n_docs = len(tokenized_corpus)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.tokenized_corpus), dtype=np.float32)
        if not query_tokens or not self.tokenized_corpus:
            return scores

        for idx, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_len[idx] or 1
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scores[idx] = score
        return scores


class HashingSemanticEncoder:
    """Offline semantic fallback: deterministic hashed bag-of-words vectors."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Iterable[str], normalize_embeddings: bool = True, **_: object) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimensions, dtype=np.float32)
            tokens = tokenize(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign

            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm:
                    vec /= norm
            vectors.append(vec)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimensions), dtype=np.float32)


class CorpusIndexer:
    def __init__(
        self,
        data_dir: str = "data/",
        chunk_tokens: int = 300,
        stride_tokens: int = 150,
        allow_csv_fallback: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_tokens = chunk_tokens
        self.stride_tokens = stride_tokens
        self.allow_csv_fallback = allow_csv_fallback

        self.chunks: List[Dict] = []
        self.domain_chunks: Dict[str, List[Dict]] = {domain: [] for domain in DOMAINS}
        self.bm25: Dict[str, object] = {}
        self.semantic_embeddings: Dict[str, np.ndarray] = {}
        self.semantic_model: object = HashingSemanticEncoder()
        self.semantic_model_name = "hashing-fallback"
        self.used_csv_fallback = False

    def build(self) -> None:
        self.chunks = []
        self.domain_chunks = {domain: [] for domain in DOMAINS}

        # Diagnostic output
        print(f"[corpus] Attempting to load from: {self.data_dir.resolve()}")
        if self.data_dir.exists():
            print(f"[corpus] data_dir exists. Contents:")
            try:
                for item in self.data_dir.iterdir():
                    print(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
            except Exception as e:
                print(f"[corpus] Error listing: {e}")
        else:
            print(f"[corpus] data_dir does NOT exist")

        files_per_domain = {domain: 0 for domain in DOMAINS}
        
        for domain in DOMAINS:
            domain_dir = self.data_dir / domain
            if not domain_dir.exists():
                print(f"[corpus] {domain:12} dir not found: {domain_dir}")
                continue
            
            domain_files = []
            # Recursively find all files in domain directory and subdirectories
            for path in sorted(domain_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    domain_files.append(path)
                    rel_path = path.relative_to(domain_dir)
                    text = self._read_document(path)
                    self._add_document(domain, str(rel_path), text)
                    print(f"[corpus]   {domain:10} {str(rel_path):50}")
            
            files_per_domain[domain] = len(domain_files)
            print(f"[corpus] {domain:12} loaded {len(domain_files):3} files → {len(self.domain_chunks[domain]):4} chunks")

        if not self.chunks and self.allow_csv_fallback:
            print(f"[corpus] No corpus files found; falling back to sample CSV")
            self._load_sample_csv_fallback()

        # Print final stats before indexing
        total_chunks = sum(len(chunks) for chunks in self.domain_chunks.values())
        print(f"[corpus] Total chunks indexed: {total_chunks}")
        
        self._build_bm25_indexes()
        self._build_semantic_indexes()

    def _read_document(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        suffix = path.suffix.lower()
        if suffix == ".html":
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            return unescape(parser.text())
        if suffix == ".json":
            try:
                return self._json_to_text(json.loads(raw))
            except json.JSONDecodeError:
                return raw
        return raw

    def _json_to_text(self, value: object) -> str:
        if isinstance(value, dict):
            # For our scraped-chunk format {source, domain, text, url}: use only 'text'
            if "text" in value:
                return str(value["text"])
            return " ".join(self._json_to_text(v) for v in value.values())
        if isinstance(value, list):
            return " ".join(self._json_to_text(v) for v in value)
        if value is None:
            return ""
        return str(value)

    def _add_document(self, domain: str, source: str, text: str, answer: Optional[str] = None) -> None:
        clean = re.sub(r"\s+", " ", text).strip()
        words = tokenize(clean)
        if not words:
            return

        spans = self._chunk_word_spans(len(words))
        raw_words = re.findall(r"\S+", clean)
        if abs(len(raw_words) - len(words)) > max(20, len(words) * 0.2):
            raw_words = clean.split()

        for idx, (start, end) in enumerate(spans):
            chunk_words = raw_words[start:end] if end <= len(raw_words) else words[start:end]
            chunk_text = " ".join(chunk_words).strip()
            if not chunk_text:
                continue
            chunk = {
                "chunk_id": f"{domain}:{source}:{idx}",
                "domain": domain,
                "source": source,
                "text": chunk_text,
                "token_count": len(tokenize(chunk_text)),
            }
            if answer:
                chunk["answer"] = re.sub(r"\s+", " ", answer).strip()
            self.chunks.append(chunk)
            self.domain_chunks[domain].append(chunk)

    def _chunk_word_spans(self, token_count: int) -> List[Tuple[int, int]]:
        if token_count <= self.chunk_tokens:
            return [(0, token_count)]

        spans: List[Tuple[int, int]] = []
        start = 0
        while start < token_count:
            end = min(start + self.chunk_tokens, token_count)
            spans.append((start, end))
            if end == token_count:
                break
            start += self.stride_tokens
        return spans

    def _load_sample_csv_fallback(self) -> None:
        root = self.data_dir.parent if self.data_dir.name == "data" else Path.cwd()
        candidates = [
            root / "support_tickets" / "support_tickets" / "sample_support_tickets.csv",
            root / "support_tickets" / "sample_support_tickets.csv",
        ]
        sample_path = next((path for path in candidates if path.exists()), None)
        if not sample_path:
            return

        self.used_csv_fallback = True
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=1):
                company = (row.get("Company") or "").lower()
                domain = self._domain_from_text(company)
                if domain not in DOMAINS:
                    domain = self._domain_from_text(" ".join(row.values()).lower())
                if domain not in DOMAINS:
                    continue
                text = " ".join(
                    part for part in [
                        row.get("Subject", ""),
                        row.get("Issue", ""),
                        row.get("Response", ""),
                    ]
                    if part
                )
                self._add_document(domain, f"sample_support_tickets.csv#{idx}", text, answer=row.get("Response", ""))

    def _domain_from_text(self, text: str) -> str:
        if "hackerrank" in text or "hacker rank" in text:
            return "hackerrank"
        if "claude" in text or "anthropic" in text:
            return "claude"
        if "visa" in text:
            return "visa"
        return "unknown"

    def _build_bm25_indexes(self) -> None:
        self.bm25 = {}
        for domain, chunks in self.domain_chunks.items():
            tokenized = [tokenize(chunk["text"]) for chunk in chunks]
            if not tokenized:
                self.bm25[domain] = SimpleBM25([])
                continue
            try:
                from rank_bm25 import BM25Okapi

                self.bm25[domain] = BM25Okapi(tokenized)
            except Exception:
                self.bm25[domain] = SimpleBM25(tokenized)

    def _build_semantic_indexes(self) -> None:
        self.semantic_model = self._load_sentence_transformer()
        self.semantic_embeddings = {}
        for domain, chunks in self.domain_chunks.items():
            texts = [chunk["text"] for chunk in chunks]
            self.semantic_embeddings[domain] = self._encode(texts)

    def _load_sentence_transformer(self) -> object:
        # Try to load sentence-transformers model (allows downloading if needed)
        try:
            from sentence_transformers import SentenceTransformer

            print("[corpus] Loading sentence-transformers model: all-MiniLM-L6-v2...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.semantic_model_name = "all-MiniLM-L6-v2"
            print("[corpus] ✓ Sentence-Transformers model loaded successfully")
            return model
        except ImportError:
            print("[corpus] sentence-transformers not installed; falling back to hashing")
            self.semantic_model_name = "hashing-fallback"
            return HashingSemanticEncoder()
        except Exception as e:
            print(f"[corpus] Failed to load sentence-transformers: {e}; falling back to hashing")
            self.semantic_model_name = "hashing-fallback"
            return HashingSemanticEncoder()

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self.semantic_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def encode_query(self, text: str) -> np.ndarray:
        embedding = self.semantic_model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = np.asarray(embedding, dtype=np.float32)[0]
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
