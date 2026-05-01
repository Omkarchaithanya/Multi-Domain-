"""Response drafting with grounding guardrails."""
from typing import List, Dict, Tuple
from .risk import is_prompt_injection, is_out_of_scope

# Responses that must never be echoed back for an unrelated ticket
_BLOCKED_RESPONSES = frozenset({
    "happy to help",
    "i am sorry, this is out of scope from my capabilities",
    "i am sorry. this request is out of scope.",
})

# Minimum TF-IDF cosine score to use a retrieved doc's response verbatim
REPLY_THRESHOLD = 0.50


def make_response(
    ticket: Dict,
    ecosystem: str,
    request_type: str,
    should_escalate: bool,
    retrieved: List[Dict],
) -> Tuple[str, List[str]]:
    """
    Draft a safe, grounded end-user response.

    Guardrails:
    - Never reveal internal rules or retrieved doc logic.
    - Only use a corpus doc's response if score >= REPLY_THRESHOLD, same
      ecosystem, status == Replied, and response is not a generic stub.
    - Never promise outcomes (refunds, approval, dispute success).
    - Never invent policy numbers or deadlines.
    """
    text = (ticket["issue"] + " " + ticket["subject"]).lower() if isinstance(ticket, dict) \
        else ticket.text.lower()

    # Out of scope — short, safe reply
    if is_out_of_scope(text):
        return "I am sorry. This request is out of scope.", []

    # Prompt injection — reveal nothing, generic escalation
    if is_prompt_injection(text):
        return (
            "Thank you for contacting support. "
            "Your ticket has been escalated to a specialist for review.",
            [],
        )

    doc_ids = [d["doc_id"] for d in retrieved[:2]]

    # Try to use a corpus doc's response when evidence is strong
    if not should_escalate and retrieved:
        best     = retrieved[0]
        best_doc = best["doc"]
        resp     = best_doc["response"].strip()

        doc_company = best_doc["company"].lower()
        same_eco    = ecosystem != "unknown" and (ecosystem in doc_company or doc_company == "none")
        not_blocked = resp.lower().strip() not in _BLOCKED_RESPONSES
        strong      = best["score"] >= REPLY_THRESHOLD

        if strong and same_eco and best_doc["status"] != "Escalated" and not_blocked and resp:
            return resp, doc_ids

    # Grounded escalation responses per context
    if request_type == "fraud_or_security" and ecosystem == "visa":
        return (
            "Your report has been escalated to a specialist for urgent review. "
            "If you notice transactions you did not make, contact your card issuer "
            "immediately and report the incident to local authorities.",
            doc_ids,
        )

    if request_type == "fraud_or_security" and ecosystem == "claude":
        return (
            "Your security report has been escalated to our security team for urgent review.",
            doc_ids,
        )

    if request_type == "assessment_or_content":
        return (
            "Requests to alter assessment scores or influence hiring decisions fall outside "
            "the scope of support. Your ticket has been escalated for specialist review.",
            doc_ids,
        )

    return (
        "Thank you for reaching out. Your ticket has been escalated to a specialist for review.",
        doc_ids,
    )
