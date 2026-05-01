import json
import os
import re
from typing import Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are a friendly customer support agent. A customer sent you a support ticket. You have been given documentation excerpts that may or may not directly answer the question.

Your job:
1. Read the customer's question carefully
2. Use whatever relevant information exists in the docs
3. Write a complete, polished support response in 2-4 sentences
4. ALWAYS start with a complete sentence - never mid-sentence
5. NEVER output raw API parameters, code snippets, or JSON
6. NEVER start with a lowercase letter
7. If docs are not relevant, say: "Thank you for reaching out. Based on our documentation, I wasn't able to find a specific answer to your question. A human support agent will follow up with you shortly."
8. End every reply with: "Let us know if you need further help."
"""

UNCOVERED_RESPONSE = "I was unable to find specific guidance on this issue in our documentation. A human support agent will follow up with you."


def generate_response(ticket_text: str, chunks: List[Dict], tone: str = "neutral") -> str:
    """Generate a response with tone adaptation and grounding verification."""
    if not chunks:
        return UNCOVERED_RESPONSE

    documentation = _format_chunks(chunks)
    response = _generate_with_claude(ticket_text, documentation, tone)
    if not response:
        response = _extractive_response(chunks)

    # Innovation 3: Self-Verification Grounding Loop
    grounding_score, verdict, unsupported_claims = verify_grounding(response, chunks)
    
    if verdict == "escalate" or grounding_score < 0.7:
        # Response is not sufficiently grounded; escalate
        return UNCOVERED_RESPONSE
    elif verdict == "revise":
        # Attempt revision with explicit warning
        revision_doc = f"PREVIOUS ATTEMPT ISSUES:\nUnsupported claims: {', '.join(unsupported_claims)}\n\nREVISE using ONLY documentation:\n{documentation}"
        response = _generate_with_claude(ticket_text, revision_doc, tone)
        if not response:
            return UNCOVERED_RESPONSE

    if not _is_response_grounded(response, chunks):
        response = UNCOVERED_RESPONSE

    # Add proactive follow-up if available
    followup = generate_proactive_followup(ticket_text, response, chunks)
    if followup:
        response = f"{response}\n\n{followup}"

    sources = _source_citation(chunks)
    if sources and "source:" not in response.lower() and "sources:" not in response.lower():
        response = f"{response.rstrip()} Source: {sources}."

    final = _clean_response(_under_word_limit(response, limit=150))

    # Ensure first character is uppercase
    if final and final != UNCOVERED_RESPONSE and final[0].islower():
        final = final[0].upper() + final[1:]

    # Ensure response ends with the required closing phrase
    _CLOSING = "Let us know if you need further help."
    if final and final != UNCOVERED_RESPONSE and _CLOSING.lower() not in final.lower():
        final = final.rstrip(" .") + " " + _CLOSING

    return final


def _generate_with_claude(ticket_text: str, documentation: str, tone: str = "neutral") -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic

        # Get tone-specific instructions
        tone_instruction = _get_tone_instruction(tone)
        
        system_prompt = SYSTEM_PROMPT + f"\nTONE INSTRUCTION: {tone_instruction}"

        user_message = f"""Customer ticket:
{ticket_text}

Available documentation:
{documentation}

Write a complete 2-4 sentence support response. Start with a capital letter. Do not copy raw text from docs."""

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=220,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip()
    except Exception:
        return ""


def _get_tone_instruction(tone: str) -> str:
    """Get LLM instruction based on detected tone."""
    instructions = {
        "frustrated": "Acknowledge the user's frustration explicitly in the first sentence. Be brief and action-oriented.",
        "urgent": "Lead immediately with the fastest resolution path. No preamble.",
        "confused": "Use simple language. Explain each step clearly without jargon.",
        "angry": "Stay calm and professional. Acknowledge the issue sincerely before explaining.",
        "neutral": "Use a warm, professional, and helpful tone.",
    }
    return instructions.get(tone, instructions["neutral"])


def verify_grounding(response: str, chunks: List[Dict]) -> Tuple[float, str, List[str]]:
    """
    Innovation 3: Self-Verification Grounding Loop
    
    Verifies that every factual claim in the response is supported by documentation.
    Prevents hallucinated policies, prices, or procedures.
    
    Returns: (grounding_score: float 0-1, verdict: "pass"|"revise"|"escalate", unsupported_claims: List[str])
    """
    if response == UNCOVERED_RESPONSE:
        return 1.0, "pass", []
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        # Fallback to basic grounding check
        score = 0.8 if _is_response_grounded(response, chunks) else 0.4
        return score, "pass", []
    
    try:
        import anthropic
        
        documentation = "\n\n".join([chunk["text"] for chunk in chunks])
        
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0,
            system="""You are a fact-checker for support responses. Evaluate whether the response is fully grounded in the provided documentation.
Return JSON only: {"score": 0.0-1.0, "unsupported_claims": ["claim1", "claim2"], "verdict": "pass"|"revise"|"escalate"}
- 1.0 = every claim directly supported by docs
- 0.5 = some claims unverifiable
- 0.0 = claims contradict or go beyond docs""",
            messages=[{
                "role": "user",
                "content": f"RESPONSE:\n{response}\n\nDOCUMENTATION:\n{documentation}"
            }],
        )
        
        response_text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()
        
        try:
            result = json.loads(response_text)
            score = float(result.get("score", 0.5))
            unsupported = result.get("unsupported_claims", [])
            verdict = result.get("verdict", "pass").lower()
            if verdict not in ("pass", "revise", "escalate"):
                verdict = "pass"
            return min(1.0, max(0.0, score)), verdict, unsupported
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        
        return 0.8, "pass", []
    
    except Exception:
        return 0.8, "pass", []


def generate_proactive_followup(ticket_text: str, response_text: str, chunks: List[Dict]) -> Optional[str]:
    """
    Innovation 4: Proactive Follow-up Anticipation
    
    Anticipates the most likely follow-up question and provides a subtle hint.
    Makes the agent feel genuinely intelligent rather than reactive.
    
    Returns: Formatted follow-up section or None
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    
    if not response_text or response_text == UNCOVERED_RESPONSE:
        return None
    
    try:
        import anthropic
        
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            temperature=0,
            system="""You are a support expert. Given a support ticket and the response provided, 
what is the single most likely follow-up question the user will ask?
Return a one-sentence question, or respond with 'none' if no obvious follow-up exists.""",
            messages=[{
                "role": "user",
                "content": f"TICKET:\n{ticket_text[:1000]}\n\nRESPONSE PROVIDED:\n{response_text}"
            }],
        )
        
        followup_question = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()
        
        if followup_question.lower() in ("none", "n/a", ""):
            return None
        
        # Generate answer for the follow-up question
        documentation = "\n\n".join([chunk["text"] for chunk in chunks])
        answer_message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            temperature=0,
            system="""You are a support agent. Answer this follow-up question using ONLY the provided documentation.
Keep response under 80 words. Do not invent information.""",
            messages=[{
                "role": "user",
                "content": f"QUESTION:\n{followup_question}\n\nDOCUMENTATION:\n{documentation}"
            }],
        )
        
        followup_answer = "".join(
            block.text for block in answer_message.content if getattr(block, "type", "") == "text"
        ).strip()
        
        if followup_answer and len(followup_answer) > 10:
            return f"\nYou might also want to know: {followup_answer}"
    
    except Exception:
        pass
    
    return None



def _format_chunks(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        if text:
            parts.append(f"[Doc {i}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _extractive_response(chunks: List[Dict]) -> str:
    if not chunks:
        return UNCOVERED_RESPONSE

    _DUMP_PATTERNS = [
        r"OAuth 2\.0.*Bearer token",
        r"^(HackerRank\s+)?(Troubleshooting Guide|API Documentation)\b",
        r"^SUBMISSION FAILURES\b",
        # Raw API parameter lists
        r"^(?:[a-z_]+ ){3,}(?:array|object|string|integer|boolean|response|request|token|id|url)\b",
        # Table dumps: multi-word phrase repeated ("Required when Required when")
        r"(\b\w+(?:\s+\w+){1,4})\s+\1\b",
        # Navigation boilerplate (no anchor — prefix words vary)
        r"Is this what you.re looking for\?",
        r"Consumer Support.*\|\s*Visa",
        r"Travel.*Support.*\|\s*Visa",
        r"Contact Us.*\|\s*Visa",
        r"Customer Service.*\|\s*Visa",
        r"Get support \*Callers in certain countries",
        r"\|\s*Visa\b.*\|\s*Visa\b",  # multiple "| Visa" occurrences = nav page
    ]

    candidates: List[str] = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in _DUMP_PATTERNS):
            continue

        # Skip bullet-list-heavy chunks (instruction dumps)
        if text.count("- ") + text.count("\n- ") > 2:
            continue

        # Strip metadata artifacts
        text = re.sub(r"^HackerRank\s+(Troubleshooting Guide|Support Guide|API Documentation)\b.*?(?=\n|$)",
                      "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^[\w\-]+\.json\s*", "", text).strip()
        text = re.sub(r"^articles_\S+\s*", "", text).strip()
        text = re.sub(r"^(hackerrank|claude|visa)\s+", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*\|\s*[\w][\w\s]+?(Knowledge Base|Help Center|Support Center)\b", " ", text).strip()
        text = re.sub(r"\b(HackerRank|Claude|Visa)\s+(Knowledge Base|Help Center|Support Center)\s+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"^(.{5,80}?)\s+(?:HackerRank|Claude|Visa)(?:\s+\w+){0,3}\s+\1\s+", r"\1 ", text, flags=re.IGNORECASE)
        text = re.sub(r"^(.{10,80}?)\s+\1\s+", r"\1 ", text)

        if not text:
            continue

        # Prefer chunks that naturally start at a sentence boundary (capital letter)
        starts_at_boundary = text[0].isupper()

        # Split into sentences and deduplicate (exact + near-duplicate via token overlap)
        raw_sentences = [s.strip() for s in text.split(". ") if s.strip()]
        seen: List[str] = []
        for s in raw_sentences:
            # Collapse inline repeated phrases within this sentence first
            s = re.sub(r"((?:\S+\s+){1,6}\S+)\s+\1(?=\s|$)", r"\1", s, flags=re.IGNORECASE)
            norm = re.sub(r"\s+", " ", s.lower())
            is_dup = any(norm == re.sub(r"\s+", " ", prev.lower()) for prev in seen)
            is_label_dup = bool(re.match(r"^(.{3,40}):\s+\1\b", s, re.IGNORECASE))
            # Near-duplicate: use overlap / min(len_a, len_b) so substrings are caught
            s_toks = set(re.findall(r"[a-z0-9]+", norm))
            is_near_dup = False
            if s_toks:
                for prev in seen:
                    p_toks = set(re.findall(r"[a-z0-9]+", re.sub(r"\s+", " ", prev.lower())))
                    if p_toks and len(s_toks & p_toks) / min(len(s_toks), len(p_toks)) > 0.8:
                        is_near_dup = True
                        break
            if not is_dup and not is_label_dup and not is_near_dup:
                seen.append(s)

        clean = ". ".join(seen[:4]).strip()
        if not clean.endswith("."):
            clean += "."
        if clean and clean[0].islower():
            clean = clean[0].upper() + clean[1:]

        if len(clean.split()) >= 8:
            if starts_at_boundary:
                return clean
            candidates.append(clean)

    if candidates:
        return candidates[0]

    return UNCOVERED_RESPONSE


def _is_response_grounded(response: str, chunks: List[Dict]) -> bool:
    if response == UNCOVERED_RESPONSE:
        return True

    evidence = " ".join(chunk["text"] for chunk in chunks).lower()
    evidence_tokens = set(_content_tokens(evidence))
    response_tokens = _content_tokens(response)
    if not response_tokens:
        return False

    covered = sum(1 for token in response_tokens if token in evidence_tokens)
    coverage = covered / len(response_tokens)
    risky_terms = ("guarantee", "refund will", "approved", "must", "price", "$", "within 24 hours")
    has_unsupported_risky_term = any(term in response.lower() and term not in evidence for term in risky_terms)
    return coverage >= 0.45 and not has_unsupported_risky_term


def _content_tokens(text: str) -> List[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "your", "you",
        "our", "we", "is", "are", "be", "this", "that", "from", "as", "by", "at", "it", "if",
        "source", "sources",
    }
    return [token for token in re.findall(r"[a-zA-Z0-9_']+", text.lower()) if len(token) > 2 and token not in stop]


def _source_citation(chunks: List[Dict]) -> str:
    sources = []
    for chunk in chunks:
        source = chunk.get("source", "")
        if source and source not in sources:
            sources.append(source)
        if len(sources) == 3:
            break
    return ";".join(sources)


def _clean_response(response: str) -> str:
    """Strip filename artifacts, URL fragments, and navigation boilerplate."""
    # 1. Strip raw guide/API dump headers FIRST (before domain stripping removes prefix)
    response = re.sub(r"^HackerRank\s+(Troubleshooting Guide|Support Guide|API Documentation)\b.*?(?=\n|$)",
                      "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"^(Troubleshooting Guide|API Documentation)\b", "", response, flags=re.IGNORECASE).strip()
    response = re.sub(r"^SUBMISSION FAILURES\s*", "", response).strip()
    # 2. Remove OAuth API dump (check before other stripping)
    if re.search(r"\bOAuth 2\.0\b.*\bBearer token\b", response, re.IGNORECASE | re.DOTALL):
        return UNCOVERED_RESPONSE
    # 3. Remove leading JSON filenames
    response = re.sub(r"^[\w\-]+\.json\s+", "", response).strip()
    response = re.sub(r"^articles_\S+\s+", "", response).strip()
    # 4. Remove leading bare domain name (e.g. "hackerrank ")
    response = re.sub(r"^(hackerrank|claude|visa)\s+", "", response, flags=re.IGNORECASE).strip()
    # 5. Strip "| <Site> Knowledge Base" / "| <Site> Help Center" page-title patterns
    response = re.sub(r"\s*\|\s*[\w][\w\s]+?(Knowledge Base|Help Center|Support Center)\b", " ", response)
    # 5b. Strip inline "HackerRank Knowledge Base" / "Claude Help Center" labels embedded in text
    response = re.sub(r"\b(HackerRank|Claude|Visa)\s+(Knowledge Base|Help Center|Support Center)\s+", " ", response, flags=re.IGNORECASE)
    # 6. Strip leading "HackerRank" after removing its suffix
    response = re.sub(r"^(hackerrank|claude|visa)\s+", "", response, flags=re.IGNORECASE).strip()
    # 7. De-duplicate titles: "Foo Bar HackerRank [qualifier] Foo Bar" → "Foo Bar"
    response = re.sub(r"^(.{5,80}?)\s+(?:HackerRank|Claude|Visa)(?:\s+\w+){0,3}\s+\1\s+", r"\1 ", response, flags=re.IGNORECASE)
    # 8. Remove exact duplicate phrase at start
    response = re.sub(r"^(.{10,80}?)\s+\1\s+", r"\1 ", response)
    # 9. Strip article-title prefixes (Title Case phrase ending with known title words)
    response = re.sub(
        r"^[A-Z][A-Za-z0-9 \-]{5,60}(?:Notification|Guide|Practices|Overview|Summary|Introduction|FAQ)\s+",
        "", response,
    ).strip()
    # 10. Strip "This article outlines/explains/describes..." meta-sentences entirely
    response = re.sub(
        r"^This article (?:outlines|explains|describes|provides|covers)[^.]+\.\s*",
        "", response,
    ).strip()
    # 12. Remove lines that are just a filename
    lines = [
        line for line in response.splitlines()
        if not re.match(r"^[\w\-]+\.json\s*$", line.strip())
    ]
    return "\n".join(lines).strip()


def _under_word_limit(response: str, limit: int) -> str:
    words = response.split()
    if len(words) <= limit:
        return response
    return " ".join(words[:limit]).rstrip(" ,;") + "."
