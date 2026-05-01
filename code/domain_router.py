import os
import re
from typing import Dict, Tuple


DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "visa": (
        "visa",
        "card",
        "transaction",
        "chargeback",
        "pin",
        "cvv",
        "merchant",
        "issuer",
        "debit",
        "credit card",
        "unauthorized charge",
    ),
    "hackerrank": (
        "hackerrank",
        "hacker rank",
        "assessment",
        "test",
        "candidate",
        "proctoring",
        "coding challenge",
        "hire",
        "recruiter",
        "interview",
        "score",
    ),
    "claude": (
        "claude",
        "anthropic",
        "conversation",
        "message limit",
        "subscription",
        "plan",
        "artifact",
        "claude.ai",
        "workspace",
        "model",
    ),
}


def _count_keyword_hits(ticket_text: str, keywords: Tuple[str, ...]) -> int:
    text = ticket_text.lower()
    hits = 0
    for keyword in keywords:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            hits += 1
    return hits


def _score_domains(ticket_text: str) -> Dict[str, int]:
    text = ticket_text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                score += 2 if keyword in {"visa", "hackerrank", "claude", "anthropic"} else 1
        scores[domain] = score
    return scores


def route_domain_with_confidence(ticket_text: str) -> Tuple[str, float, str]:
    scores = _score_domains(ticket_text)
    if scores.get("visa", 0) > 0 and scores.get("hackerrank", 0) > 0:
        visa_hits = _count_keyword_hits(ticket_text, DOMAIN_KEYWORDS["visa"])
        hackerrank_hits = _count_keyword_hits(ticket_text, DOMAIN_KEYWORDS["hackerrank"])
        if hackerrank_hits >= visa_hits:
            best_domain = "hackerrank"
            best_score = scores["hackerrank"]
            second_score = scores["visa"]
        else:
            best_domain = "visa"
            best_score = scores["visa"]
            second_score = scores["hackerrank"]
    else:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_domain, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score >= 2 and best_score > second_score:
        confidence = min(0.95, 0.55 + 0.12 * best_score + 0.08 * (best_score - second_score))
        return best_domain, round(confidence, 2), f"keyword score={scores}"

    if best_score == 1 and second_score == 0:
        return best_domain, 0.58, f"weak keyword score={scores}"

    llm_domain = _route_domain_with_claude(ticket_text)
    if llm_domain:
        return llm_domain, 0.72, f"ambiguous keyword score={scores}; claude_router={llm_domain}"

    fallback = best_domain if best_score > 0 else "claude"
    return fallback, 0.35 if best_score == 0 else 0.48, f"ambiguous keyword score={scores}; fallback={fallback}"


def route_domain(ticket_text: str) -> str:
    domain, _, _ = route_domain_with_confidence(ticket_text)
    return domain


def _route_domain_with_claude(ticket_text: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8,
            temperature=0,
            system="Classify the support ticket into exactly one domain: hackerrank, claude, or visa. Return only the domain.",
            messages=[{"role": "user", "content": ticket_text[:3000]}],
        )
        content = "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip().lower()
        for domain in DOMAIN_KEYWORDS:
            if domain in content:
                return domain
    except Exception:
        return ""
    return ""

