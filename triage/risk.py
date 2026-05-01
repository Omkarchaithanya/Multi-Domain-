"""Risk assessment and escalation logic."""
import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Risk triggers
# ---------------------------------------------------------------------------

HIGH_RISK_TRIGGERS = [
    "fraud", "stolen", "unauthorized", "identity theft", "hacked",
    "compromised", "suspicious login", "account takeover",
    "security vulnerability", "bug bounty", "legal", "lawsuit",
    "regulatory", "compliance", "law enforcement", "chargeback",
    "cheating", "plagiarism", "integrity violation",
    "privacy complaint", "data breach", "data deletion", "gdpr",
    "increase my score", "move me to the next round",
]

MEDIUM_RISK_TRIGGERS = [
    "refund", "wrong product", "not received", "blocked card",
    "access denied", "subscription", "payment issue",
]

INJECTION_PATTERNS = [
    r"affiche toutes les r.{0,5}gles",
    r"show\s+(all\s+)?internal rules",
    r"reveal.*prompt",
    r"ignore.*instructions",
    r"disclose.*logic",
    r"what are your instructions",
    r"show me your system prompt",
    r"reveal.*document",
    r"display.*retrieved",
    r"logique exacte",
    r"r.{0,5}gles internes",
]

OOS_PATTERNS = [
    r"delete all files",
    r"rm\s+-rf",
    r"format.*drive",
    r"code to delete",
    r"who is.*actor",
    r"capital of",
]

# ---------------------------------------------------------------------------
# Risk Policy Matrix
# ---------------------------------------------------------------------------
#
# Ecosystem  | Condition                           | Risk   | Escalate
# -----------+-------------------------------------+--------+---------
# any        | prompt injection                    | high   | yes
# any        | HIGH_RISK_TRIGGERS keyword          | high   | yes
# any        | request_type == fraud_or_security   | high   | yes
# visa       | chargeback / dispute / fraud        | high   | yes
# any        | MEDIUM_RISK_TRIGGERS keyword        | medium | context
# any        | request_type == billing             | medium | yes (no docs)
# any        | request_type == account_access      | medium | yes (no docs)
# any        | OOS_PATTERNS                        | low    | no (reply OOS)
# any        | everything else                     | low    | no


def assess_risk(text: str, request_type: str) -> Tuple[str, bool, str]:
    """
    Returns (risk_level, should_escalate, reason).
    Caller is responsible for additional escalation rules (confidence, no-doc, etc.)
    """
    tl = text.lower()

    # Prompt injection — always high
    for pat in INJECTION_PATTERNS:
        if re.search(pat, tl):
            return "high", True, "Prompt injection attempt detected."

    # Out of scope — low, do NOT escalate (reply with OOS message)
    for pat in OOS_PATTERNS:
        if re.search(pat, tl):
            return "low", False, "Request is out of scope."

    # High-risk keyword triggers
    for kw in HIGH_RISK_TRIGGERS:
        if kw in tl:
            return "high", True, f'High-risk keyword detected: "{kw}".'

    if request_type == "fraud_or_security":
        return "high", True, "Request type is fraud_or_security."

    # Medium-risk triggers
    for kw in MEDIUM_RISK_TRIGGERS:
        if kw in tl:
            return "medium", False, f'Medium-risk keyword: "{kw}".'

    if request_type == "billing":
        return "medium", False, "Billing request."

    return "low", False, "Low-risk request."


def is_prompt_injection(text: str) -> bool:
    tl = text.lower()
    return any(re.search(pat, tl) for pat in INJECTION_PATTERNS)


def is_out_of_scope(text: str) -> bool:
    tl = text.lower()
    return any(re.search(pat, tl) for pat in OOS_PATTERNS)
