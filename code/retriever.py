from typing import Dict, List

import numpy as np

from corpus_indexer import CorpusIndexer, tokenize

# Tracks chunk_ids that have already been returned as the top result for a
# previous ticket in the same run, enabling diversity penalty logic.
_used_top_chunks: set = set()


def retrieve(ticket_text: str, domain: str, indexer: CorpusIndexer) -> List[Dict]:
    chunks = _hybrid_retrieve(ticket_text, domain, indexer)

    # Diversity penalty: chunks already used as a top result get 35% score cut
    for chunk in chunks:
        if chunk["chunk_id"] in _used_top_chunks:
            chunk["score"] = round(chunk["score"] * 0.65, 4)

    # Re-sort after penalty
    chunks.sort(key=lambda x: x["score"], reverse=True)

    # Register the new top chunk so it can be penalised for future tickets
    if chunks:
        _used_top_chunks.add(chunks[0]["chunk_id"])

    # Query expansion fallback (unchanged)
    if not chunks or chunks[0]["score"] < 0.22:
        expanded = _expand_query(ticket_text, domain)
        if expanded != ticket_text:
            retry = _hybrid_retrieve(expanded, domain, indexer)
            chunks = _merge_ranked_lists(chunks, retry)[:5]
            for chunk in chunks:
                chunk["multi_hop"] = True

    return chunks[:5]


def _hybrid_retrieve(ticket_text: str, domain: str, indexer: CorpusIndexer) -> List[Dict]:
    domain_chunks = indexer.domain_chunks.get(domain, [])
    if not domain_chunks:
        return []

    bm25_hits = _bm25_hits(ticket_text, domain, indexer, top_k=5)
    semantic_hits = _semantic_hits(ticket_text, domain, indexer, top_k=5)

    merged: Dict[str, Dict] = {}
    for rank, hit in enumerate(bm25_hits, start=1):
        entry = merged.setdefault(hit["chunk_id"], {**hit, "bm25_score": 0.0, "semantic_score": 0.0})
        entry["bm25_score"] = max(entry["bm25_score"], hit["bm25_score"])
        entry["rank_bonus"] = entry.get("rank_bonus", 0.0) + 1.0 / (rank + 1)

    for rank, hit in enumerate(semantic_hits, start=1):
        entry = merged.setdefault(hit["chunk_id"], {**hit, "bm25_score": 0.0, "semantic_score": 0.0})
        entry["semantic_score"] = max(entry["semantic_score"], hit["semantic_score"])
        entry["rank_bonus"] = entry.get("rank_bonus", 0.0) + 1.0 / (rank + 1)

    results = []
    for hit in merged.values():
        score = 0.52 * hit.get("bm25_score", 0.0) + 0.43 * hit.get("semantic_score", 0.0) + 0.05 * hit.get("rank_bonus", 0.0)
        results.append({
            "chunk_id": hit["chunk_id"],
            "domain": hit["domain"],
            "source": hit["source"],
            "text": hit["text"],
            "answer": hit.get("answer", ""),
            "score": round(float(min(score, 1.0)), 4),
            "bm25_score": round(float(hit.get("bm25_score", 0.0)), 4),
            "semantic_score": round(float(hit.get("semantic_score", 0.0)), 4),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:5]


def _bm25_hits(ticket_text: str, domain: str, indexer: CorpusIndexer, top_k: int) -> List[Dict]:
    chunks = indexer.domain_chunks.get(domain, [])
    scores = np.asarray(indexer.bm25[domain].get_scores(tokenize(ticket_text)), dtype=np.float32)
    if scores.size == 0:
        return []
    max_score = float(scores.max())
    normalized = scores / max_score if max_score > 0 else scores
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**chunks[int(idx)], "bm25_score": float(normalized[int(idx)])}
        for idx in top_indices
        if scores[int(idx)] > 0
    ]


def _semantic_hits(ticket_text: str, domain: str, indexer: CorpusIndexer, top_k: int) -> List[Dict]:
    chunks = indexer.domain_chunks.get(domain, [])
    embeddings = indexer.semantic_embeddings.get(domain)
    if embeddings is None or embeddings.size == 0:
        return []

    query = indexer.encode_query(ticket_text)
    scores = embeddings @ query
    normalized = np.clip((scores + 1.0) / 2.0, 0.0, 1.0)
    top_indices = np.argsort(normalized)[::-1][:top_k]
    return [
        {**chunks[int(idx)], "semantic_score": float(normalized[int(idx)])}
        for idx in top_indices
        if normalized[int(idx)] > 0
    ]


def _merge_ranked_lists(first: List[Dict], second: List[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for hit in first + second:
        current = merged.get(hit["chunk_id"])
        if current is None or hit["score"] > current["score"]:
            merged[hit["chunk_id"]] = hit
    results = list(merged.values())
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _expand_query(ticket_text: str, domain: str) -> str:
    lower = ticket_text.lower()
    expansions = []
    if domain == "visa":
        expansions.extend(["card issuer", "transaction dispute", "chargeback", "merchant", "fraud"])
    elif domain == "hackerrank":
        expansions.extend(["assessment", "candidate test", "proctoring", "coding challenge", "recruiter"])
    elif domain == "claude":
        expansions.extend(["claude.ai", "conversation", "workspace", "subscription", "message limit"])

    if "refund" in lower or "charge" in lower:
        expansions.extend(["billing dispute", "payment"])
    if "login" in lower or "access" in lower:
        expansions.extend(["account access", "verification"])
    if "bug" in lower or "error" in lower:
        expansions.extend(["technical issue", "troubleshooting"])

    deduped = [term for term in dict.fromkeys(expansions) if term not in lower]
    return f"{ticket_text} {' '.join(deduped)}".strip()
