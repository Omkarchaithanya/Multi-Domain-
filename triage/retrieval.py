"""TF-IDF corpus indexing and retrieval."""
import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Any


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def build_index(corpus: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Build TF-IDF vectors and IDF table for the corpus."""
    N = len(corpus)
    df: Dict[str, int] = defaultdict(int)
    for doc in corpus:
        for token in set(tokenize(doc["text"])):
            df[token] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}

    vectors = []
    for doc in corpus:
        tokens = tokenize(doc["text"])
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        n = len(tokens) or 1
        vec = {t: (c / n) * idf.get(t, 0) for t, c in tf.items()}
        vectors.append(vec)

    return vectors, idf


def _cosine(v1: Dict, v2: Dict) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    n1  = math.sqrt(sum(x * x for x in v1.values()))
    n2  = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def retrieve(
    query: str,
    corpus: List[Dict],
    vectors: List[Dict],
    idf: Dict,
    ecosystem: str,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Return top-N corpus entries ranked by TF-IDF cosine similarity."""
    tokens = tokenize(query)
    tf: Dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    n = len(tokens) or 1
    qvec = {t: (c / n) * idf.get(t, 0) for t, c in tf.items()}

    scores = []
    for i, (doc, vec) in enumerate(zip(corpus, vectors)):
        company = doc["company"].lower()
        # Boost docs from the same ecosystem
        boost = 1.2 if ecosystem != "unknown" and ecosystem in company else 1.0
        scores.append((_cosine(qvec, vec) * boost, i))

    scores.sort(reverse=True)
    return [
        {"doc_id": corpus[i]["doc_id"], "score": s, "doc": corpus[i]}
        for s, i in scores[:top_n]
        if s > 0.0
    ]
