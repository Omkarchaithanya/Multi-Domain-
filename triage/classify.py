"""Ecosystem detection, request-type, and product-area classification."""
import re
from collections import defaultdict
from typing import Tuple

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

ECOSYSTEM_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "assessment", "coding test", "test score",
        "mock interview", "hiring", "candidate", "employer", "contest",
        "submission", "certificate", "recruiter", "resume builder",
    ],
    "claude": [
        "claude", "anthropic", "workspace", "bedrock", "lti", "model",
        "conversation", "ai model", "crawl", "claude api",
    ],
    "visa": [
        "visa", "card", "merchant", "chargeback", "transaction", "pin",
        "issuer", "cardholder", "acquirer", "payment card", "traveller",
        "cheque", "stolen card", "lost card", "spend minimum",
    ],
}

REQUEST_TYPE_KEYWORDS = {
    "billing": [
        "billing", "payment", "invoice", "subscription", "charge", "refund",
        "money", "order", "fee", "cost", "paid", "pay", "give me my money",
    ],
    "technical_issue": [
        "not working", "down", "error", "bug", "broken", "failing", "failed",
        "issue", "problem", "crash", "unable", "cannot", "blocked", "stopped",
        "submissions", "blocker",
    ],
    "account_access": [
        "access", "login", "password", "locked", "seat", "remove user",
        "delete account", "lost access",
    ],
    "fraud_or_security": [
        "fraud", "stolen", "unauthorized", "suspicious", "hacked",
        "compromised", "identity theft", "phishing", "security vulnerability",
        "bug bounty", "breach",
    ],
    "assessment_or_content": [
        "assessment", "test", "score", "certificate", "question", "contest",
        "challenge", "graded", "answer",
    ],
    "feature_or_usage_question": [
        "how do i", "how to", "can i", "what is", "setup", "configure",
        "lti", "integrate", "minimum spend", "why so", "inactivity",
        "crawl", "how long", "dispute a charge", "how do i dispute",
    ],
    "feedback_or_complaint": [
        "complaint", "feedback", "unhappy", "disappointed", "frustrated",
    ],
}

PRODUCT_AREA_KEYWORDS = {
    "hackerrank": {
        "assessments":           ["assessment", "test", "score", "certificate",
                                  "answer", "graded", "compatible", "zoom",
                                  "proctoring", "reschedule"],
        "interviews":            ["interview", "mock interview", "inactivity",
                                  "screen share", "lobby", "interviewer"],
        "candidate_platform":    ["apply", "submission", "resume", "practice",
                                  "profile", "community"],
        "employer_platform":     ["hiring", "recruiter", "invite", "infosec",
                                  "subscription", "seat", "remove user",
                                  "employee", "plan"],
        "billing":               ["billing", "subscription", "payment",
                                  "invoice", "order", "money"],
        "account_access":        ["login", "access", "password", "account",
                                  "delete account"],
        "community_or_contests": ["contest", "community", "leaderboard",
                                  "ranking"],
    },
    "claude": {
        "workspaces":            ["workspace", "team", "seat", "admin"],
        "api_or_integrations":   ["api", "bedrock", "integration", "lti",
                                  "sdk", "key", "aws"],
        "billing":               ["billing", "subscription", "payment",
                                  "invoice"],
        "product_access":        ["access", "not working", "failing", "down",
                                  "responding", "stopped"],
        "safety_or_policy":      ["safety", "vulnerability", "security",
                                  "bug bounty", "policy", "harm"],
        "data_privacy":          ["data", "privacy", "crawl", "train",
                                  "model improvement", "personal data",
                                  "used for"],
        "account_access":        ["login", "access", "account"],
    },
    "visa": {
        "disputes_or_chargebacks": ["dispute", "chargeback", "wrong product",
                                    "refund", "charge"],
        "security_or_fraud":       ["fraud", "stolen", "unauthorized",
                                    "identity theft", "blocked", "suspicious"],
        "cardholder_support":      ["card", "cardholder", "blocked", "lost",
                                    "cash", "minimum spend", "traveller",
                                    "cheque", "pin", "spend"],
        "merchant_support":        ["merchant", "seller", "retailer"],
        "online_payments":         ["online", "e-commerce", "purchase"],
        "billing":                 ["billing", "fee"],
        "account_access":          ["access", "account"],
    },
}

# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def detect_ecosystem(ticket) -> Tuple[str, float]:
    """Return (ecosystem, confidence) from company field + keyword fallback."""
    company = ticket.company.lower().strip() if hasattr(ticket, "company") else ticket["company"].lower().strip()
    if "hackerrank" in company:
        return "hackerrank", 0.95
    if "claude" in company:
        return "claude", 0.95
    if "visa" in company:
        return "visa", 0.95

    if company in ("", "none"):
        text_field = ticket.text if hasattr(ticket, "text") else (ticket["issue"] + " " + ticket["subject"])
        tl = text_field.lower()
        scores = {eco: sum(1 for kw in kws if kw in tl)
                  for eco, kws in ECOSYSTEM_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "unknown", 0.20
        total = sum(scores.values())
        conf = round(min(0.80, 0.45 + (scores[best] / total) * 0.45), 2)
        return best, conf

    return "unknown", 0.25


def classify_request_type(text: str) -> Tuple[str, float]:
    """Return (request_type, confidence) via keyword scoring."""
    tl = text.lower()
    scores: dict = defaultdict(int)
    for rtype, kws in REQUEST_TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in tl:
                scores[rtype] += 1
    if not scores:
        return "other", 0.40
    best  = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round(min(0.90, 0.50 + (scores[best] / total) * 0.40), 2)
    return best, conf


def classify_product_area(text: str, ecosystem: str) -> Tuple[str, float]:
    """Return (product_area, confidence) for the given ecosystem."""
    tl     = text.lower()
    kw_map = PRODUCT_AREA_KEYWORDS.get(ecosystem, {})
    if not kw_map:
        return "other", 0.40
    scores: dict = defaultdict(int)
    for area, kws in kw_map.items():
        for kw in kws:
            if kw in tl:
                scores[area] += 1
    if not scores:
        return "other", 0.40
    best  = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round(min(0.90, 0.50 + (scores[best] / total) * 0.40), 2)
    return best, conf
