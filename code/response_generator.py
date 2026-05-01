import json
import os
import re
from typing import Dict, List, Optional, Tuple


SYSTEM_PROMPT = """You are a support agent. Answer the user's support ticket using ONLY the documentation excerpts provided below. 
Do NOT invent policies, prices, features, or procedures not mentioned in the docs.
If the documentation does not cover the issue, respond: "I was unable to find specific guidance on this issue in our documentation. A human support agent will follow up with you."
Keep responses concise (under 150 words), professional, and actionable."""

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

    return _under_word_limit(response, limit=150)


def _generate_with_claude(ticket_text: str, documentation: str, tone: str = "neutral") -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic

        # Get tone-specific instructions
        tone_instruction = _get_tone_instruction(tone)
        
        system_prompt = f"""You are a support agent. Answer the user's support ticket using ONLY the documentation excerpts provided below. 
Do NOT invent policies, prices, features, or procedures not mentioned in the docs.
If the documentation does not cover the issue, respond: "I was unable to find specific guidance on this issue in our documentation. A human support agent will follow up with you."
Keep responses concise (under 150 words), professional, and actionable.

TONE INSTRUCTION: {tone_instruction}"""

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=220,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": f"TICKET: {ticket_text}\n\nDOCUMENTATION:\n{documentation}"}],
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
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted.append(f"[{idx}] SOURCE: {chunk['source']}\n{chunk['text']}")
    return "\n\n".join(formatted)


def _extractive_response(chunks: List[Dict]) -> str:
    best = chunks[0]
    text = re.sub(r"\s+", " ", best.get("answer") or best["text"]).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = []
    for sentence in sentences:
        if 35 <= len(sentence) <= 260:
            selected.append(sentence)
        if len(" ".join(selected).split()) >= 90:
            break

    if not selected:
        selected = [text[:500].rstrip()]

    reply = " ".join(selected)
    if len(reply.split()) < 8:
        return UNCOVERED_RESPONSE
    return reply


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


def _under_word_limit(response: str, limit: int) -> str:
    words = response.split()
    if len(words) <= limit:
        return response
    return " ".join(words[:limit]).rstrip(" ,;") + "."
