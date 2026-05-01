"""
SupportTriageAgent — Multi-Domain Support Triage Pipeline
Reads support_tickets.csv, triages each ticket using the local corpus,
writes output.csv and log.txt.
"""

import csv
import re
import math
import os
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "support_tickets", "support_tickets")
CORPUS_PATH  = os.path.join(DATA_DIR, "sample_support_tickets.csv")
TICKETS_PATH = os.path.join(DATA_DIR, "support_tickets.csv")
OUTPUT_PATH  = os.path.join(DATA_DIR, "output.csv")
LOG_PATH     = os.path.join(DATA_DIR, "log.txt")

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

# Internal rich taxonomy (drives risk / escalation logic)
_INTERNAL_REQUEST_TYPES = [
    "billing", "technical_issue", "account_access", "fraud_or_security",
    "assessment_or_content", "feature_or_usage_question",
    "feedback_or_complaint", "other",
]

# Output 4-label taxonomy
REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

_REQUEST_TYPE_MAP = {
    "billing":                   "product_issue",
    "technical_issue":           "bug",
    "account_access":            "product_issue",
    "fraud_or_security":         "product_issue",
    "assessment_or_content":     "product_issue",
    "feature_or_usage_question": "feature_request",
    "feedback_or_complaint":     "product_issue",
    "other":                     "invalid",
}

# ---------------------------------------------------------------------------
# Keyword Rules
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
        "delete account", "lost access", "remove.*from",
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
        "crawl", "data.*used", "how long", "dispute a charge",
        "how do i dispute",
    ],
    "feedback_or_complaint": [
        "complaint", "feedback", "unhappy", "disappointed", "frustrated",
    ],
}

PRODUCT_AREA_KEYWORDS = {
    "hackerrank": {
        "assessments":      ["assessment", "test", "score", "certificate",
                             "answer", "graded", "compatible", "zoom",
                             "proctoring", "reschedule"],
        "interviews":       ["interview", "mock interview", "inactivity",
                             "screen share", "lobby", "interviewer"],
        "candidate_platform": ["apply", "submission", "resume", "practice",
                               "profile", "community"],
        "employer_platform": ["hiring", "recruiter", "invite", "infosec",
                              "subscription", "seat", "remove user",
                              "employee", "plan"],
        "billing":          ["billing", "subscription", "payment", "invoice",
                             "order", "money"],
        "account_access":   ["login", "access", "password", "account",
                             "delete account"],
        "community_or_contests": ["contest", "community", "leaderboard",
                                  "ranking"],
    },
    "claude": {
        "workspaces":       ["workspace", "team", "seat", "admin"],
        "api_or_integrations": ["api", "bedrock", "integration", "lti",
                                "sdk", "key", "aws"],
        "billing":          ["billing", "subscription", "payment", "invoice"],
        "product_access":   ["access", "not working", "failing", "down",
                             "responding", "stopped"],
        "safety_or_policy": ["safety", "vulnerability", "security",
                             "bug bounty", "policy", "harm"],
        "data_privacy":     ["data", "privacy", "crawl", "train",
                             "model improvement", "personal data", "used for"],
        "account_access":   ["login", "access", "account"],
    },
    "visa": {
        "disputes_or_chargebacks": ["dispute", "chargeback", "wrong product",
                                    "refund", "charge"],
        "security_or_fraud": ["fraud", "stolen", "unauthorized",
                              "identity theft", "blocked", "suspicious"],
        "cardholder_support": ["card", "cardholder", "blocked", "lost",
                               "cash", "minimum spend", "traveller",
                               "cheque", "pin", "spend"],
        "merchant_support": ["merchant", "seller", "retailer"],
        "online_payments":  ["online", "e-commerce", "purchase"],
        "billing":          ["billing", "fee"],
        "account_access":   ["access", "account"],
    },
}

# Keywords that always trigger HIGH risk + escalation
HIGH_RISK_TRIGGERS = [
    "fraud", "stolen", "unauthorized", "identity theft", "hacked",
    "compromised", "suspicious login", "account takeover",
    "security vulnerability", "bug bounty", "legal", "lawsuit",
    "regulatory", "compliance", "law enforcement", "chargeback",
    "cheating", "plagiarism", "integrity violation",
    "privacy complaint", "data breach", "data deletion", "gdpr",
    "increase my score", "move me to the next round",
]

# Medium-risk keywords
MEDIUM_RISK_TRIGGERS = [
    "refund", "wrong product", "not received", "blocked card",
    "access denied", "subscription", "payment issue",
]

# Prompt injection patterns
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

# Out-of-scope patterns (reply, not escalate)
OOS_PATTERNS = [
    r"delete all files",
    r"rm\s+-rf",
    r"format.*drive",
    r"code to delete",
    r"who is.*actor",
    r"capital of",
]

# ---------------------------------------------------------------------------
# Extra corpus docs (provided as supplementary evidence during session)
# ---------------------------------------------------------------------------

EXTRA_CORPUS_DOCS = [
    {
        "doc_id": "visa_014",
        "issue": "unauthorized transactions on visa card",
        "subject": "How to report unauthorized transactions",
        "company": "Visa",
        "response": (
            "If you notice transactions you did not make, contact your card issuer "
            "immediately and follow the dispute process described by your issuer."
        ),
        "product_area": "security_or_fraud",
        "status": "Replied",
        "request_type": "fraud_or_security",
        "text": (
            "unauthorized transactions visa card contact card issuer immediately "
            "dispute process"
        ),
    },
    {
        "doc_id": "visa_022",
        "issue": "how to dispute a charge on visa card chargeback",
        "subject": "Visa dispute and chargeback overview",
        "company": "Visa",
        "response": (
            "To dispute a charge on your Visa card, please contact your card issuer "
            "directly. Chargeback eligibility and timelines vary depending on the issuer "
            "and the type of dispute. Your issuer will guide you through the steps "
            "applicable to your specific case."
        ),
        "product_area": "disputes_or_chargebacks",
        "status": "Replied",
        "request_type": "disputes_or_chargebacks",
        "text": (
            "dispute charge visa card chargeback eligibility timelines issuer "
            "dispute category contact issuer next steps"
        ),
    },
]

# ---------------------------------------------------------------------------
# Corpus Loading
# ---------------------------------------------------------------------------

def load_corpus():
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = " ".join(filter(None, [
                row.get("Issue", ""),
                row.get("Subject", ""),
                row.get("Response", ""),
            ]))
            corpus.append({
                "doc_id":       f"sample-doc-{i+1}",
                "issue":        row.get("Issue", ""),
                "subject":      row.get("Subject", ""),
                "company":      row.get("Company", ""),
                "response":     row.get("Response", ""),
                "product_area": row.get("Product Area", ""),
                "status":       row.get("Status", ""),
                "request_type": row.get("Request Type", ""),
                "text":         text,
            })
    corpus.extend(EXTRA_CORPUS_DOCS)
    return corpus

def load_tickets():
    tickets = []
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            tickets.append({
                "ticket_id": f"ticket-{i+1:03d}",
                "issue":     row.get("Issue", ""),
                "subject":   row.get("Subject", ""),
                "company":   row.get("Company", "").strip(),
            })
    return tickets

# ---------------------------------------------------------------------------
# TF-IDF Retrieval
# ---------------------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_index(corpus):
    N = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for token in set(tokenize(doc["text"])):
            df[token] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in df}

    vectors = []
    for doc in corpus:
        tokens = tokenize(doc["text"])
        tf = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        n = len(tokens) or 1
        vec = {t: (c / n) * idf.get(t, 0) for t, c in tf.items()}
        vectors.append(vec)
    return vectors, idf

def cosine(v1, v2):
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot  = sum(v1[t] * v2[t] for t in common)
    n1   = math.sqrt(sum(x * x for x in v1.values()))
    n2   = math.sqrt(sum(x * x for x in v2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0

def retrieve(query, corpus, vectors, idf, ecosystem, top_n=3):
    tokens = tokenize(query)
    tf = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    n = len(tokens) or 1
    qvec = {t: (c / n) * idf.get(t, 0) for t, c in tf.items()}

    scores = []
    for i, (doc, vec) in enumerate(zip(corpus, vectors)):
        # Boost docs from the same ecosystem
        company = doc["company"].lower()
        boost = 1.2 if ecosystem != "unknown" and ecosystem in company else 1.0
        scores.append((cosine(qvec, vec) * boost, i))

    scores.sort(reverse=True)
    return [
        {"doc_id": corpus[i]["doc_id"], "score": s, "doc": corpus[i]}
        for s, i in scores[:top_n]
        if s > 0.0
    ]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def detect_ecosystem(ticket):
    company = ticket["company"].lower().strip()
    if "hackerrank" in company:
        return "hackerrank", 0.95
    if "claude" in company:
        return "claude", 0.95
    if "visa" in company:
        return "visa", 0.95
    if company in ("", "none"):
        # Try keyword detection from text
        text = (ticket["issue"] + " " + ticket["subject"]).lower()
        scores = {eco: sum(1 for kw in kws if kw in text)
                  for eco, kws in ECOSYSTEM_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "unknown", 0.20
        total = sum(scores.values())
        conf = round(min(0.80, 0.45 + (scores[best] / total) * 0.45), 2)
        return best, conf
    # Unknown company string
    return "unknown", 0.25


def classify_request_type(text):
    tl = text.lower()
    scores = defaultdict(int)
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


def classify_product_area(text, ecosystem):
    tl = text.lower()
    kw_map = PRODUCT_AREA_KEYWORDS.get(ecosystem, {})
    if not kw_map:
        return "other", 0.40
    scores = defaultdict(int)
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

# ---------------------------------------------------------------------------
# Risk Assessment
# ---------------------------------------------------------------------------

def assess_risk(text, request_type):
    tl = text.lower()

    # Prompt injection — always high + escalate
    for pat in INJECTION_PATTERNS:
        if re.search(pat, tl):
            return "high", True, "Prompt injection attempt detected."

    # Out of scope — reply with OOS message (not escalate)
    for pat in OOS_PATTERNS:
        if re.search(pat, tl):
            return "low", False, "Request is out of scope."

    # High-risk triggers
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

# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

def make_response(ticket, ecosystem, request_type, should_escalate, retrieved):
    text = (ticket["issue"] + " " + ticket["subject"]).lower()

    # Out of scope
    for pat in OOS_PATTERNS:
        if re.search(pat, text):
            return "I am sorry. This request is out of scope.", []

    # Prompt injection — reveal nothing
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text):
            return (
                "Thank you for contacting support. "
                "Your ticket has been escalated to a specialist for review.",
                [],
            )

    doc_ids = [d["doc_id"] for d in retrieved[:2]]

    # Generic responses that must never be echoed back for an unrelated ticket
    BLOCKED_RESPONSES = {
        "happy to help",
        "i am sorry, this is out of scope from my capabilities",
        "i am sorry. this request is out of scope.",
    }

    if not should_escalate and retrieved:
        best = retrieved[0]
        best_doc = best["doc"]
        resp = best_doc["response"].strip()
        resp_lower = resp.lower().strip()

        # Only use doc response if:
        #   1. similarity is high enough (0.50)
        #   2. doc is from the same ecosystem
        #   3. doc is a Replied (not Escalated) entry
        #   4. response is not a generic/unrelated reply
        doc_company = best_doc["company"].lower()
        same_eco = (
            ecosystem != "unknown"
            and (ecosystem in doc_company or doc_company == "none")
        )
        not_blocked = resp_lower not in BLOCKED_RESPONSES
        high_enough = best["score"] >= 0.50

        if high_enough and same_eco and best_doc["status"] != "Escalated" and not_blocked and resp:
            return resp, doc_ids

    # Escalation responses — grounded where possible
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

# ---------------------------------------------------------------------------
# Full Triage Pipeline
# ---------------------------------------------------------------------------

def triage(ticket, corpus, vectors, idf):
    text = ticket["issue"] + " " + ticket["subject"]

    # 1. Ecosystem
    ecosystem, eco_conf = detect_ecosystem(ticket)

    # 2. Request type
    request_type, rt_conf = classify_request_type(text)

    # 3. Product area
    product_area, pa_conf = classify_product_area(text, ecosystem)

    # 4. Risk
    risk_level, should_escalate, risk_reason = assess_risk(text, request_type)

    # 5. Unknown ecosystem → escalate
    if ecosystem == "unknown":
        should_escalate = True

    # 6. Retrieve
    retrieved = retrieve(text, corpus, vectors, idf, ecosystem)

    # If best doc was escalated and is a close match → escalate
    if (not should_escalate and retrieved
            and retrieved[0]["score"] >= 0.30
            and retrieved[0]["doc"]["status"] == "Escalated"):
        should_escalate = True
        risk_reason += " Top retrieved doc was previously escalated."

    # Billing and account_access always escalate — no billing/account policy in corpus
    if not should_escalate and request_type in ("billing", "account_access"):
        should_escalate = True
        risk_reason += f" No {request_type} policy docs available in corpus."

    # Medium-risk with no strong doc match → escalate
    if not should_escalate and risk_level == "medium":
        top_score = retrieved[0]["score"] if retrieved else 0.0
        if top_score < 0.50:
            should_escalate = True
            risk_reason += " Medium-risk with insufficient corpus support."

    # 7. Confidence
    confidence = round((eco_conf + rt_conf + pa_conf) / 3, 2)

    # If confidence < 0.65 and not OOS → escalate
    is_oos = any(re.search(p, text.lower()) for p in OOS_PATTERNS)
    if confidence < 0.65 and not is_oos:
        should_escalate = True

    decision = "escalate" if should_escalate else "reply"

    # 8. Response
    response, doc_ids = make_response(
        ticket, ecosystem, request_type, should_escalate, retrieved
    )

    # Consistency check: if response is a generic escalation message but
    # decision is "reply", force escalation (no grounded reply was possible)
    if decision == "reply" and "escalated" in response.lower() and "out of scope" not in response.lower():
        decision = "escalate"
        should_escalate = True
        risk_reason += " No sufficiently matched corpus doc; defaulting to escalate."

    # Fallback doc_ids
    if not doc_ids and retrieved:
        doc_ids = [retrieved[0]["doc_id"]]

    # Build reason
    reason = risk_reason
    if retrieved:
        reason += f" Best retrieved: {retrieved[0]['doc_id']} (score={retrieved[0]['score']:.2f})."
    if ecosystem == "unknown":
        reason += " Ecosystem undetermined."

    # -----------------------------------------------------------------------
    # Output schema mapping
    # -----------------------------------------------------------------------
    is_oos_ticket    = any(re.search(p, text.lower()) for p in OOS_PATTERNS)
    is_injection     = any(re.search(p, text.lower()) for p in INJECTION_PATTERNS)
    truly_unroutable = (ecosystem == "unknown" and request_type == "other")

    if is_oos_ticket or is_injection or truly_unroutable:
        mapped_request_type = "invalid"
    else:
        mapped_request_type = _REQUEST_TYPE_MAP.get(request_type, "invalid")

    # OOS and unroutable tickets get a direct "replied" (not escalated)
    if is_oos_ticket or truly_unroutable:
        should_escalate = False
        decision = "reply"
        if truly_unroutable and not is_oos_ticket:
            response = "I am sorry. This request is out of scope."
            doc_ids  = []

    output_status = "escalated" if should_escalate else "replied"

    result = {
        # Internal fields (used by log writer + evaluate.py)
        "ticket_id":         ticket["ticket_id"],
        "issue":             ticket["issue"],
        "subject":           ticket["subject"],
        "company":           ticket["company"],
        "ecosystem":         ecosystem,
        "_request_type_internal": request_type,
        "risk_level":        risk_level,
        "decision":          decision,
        "confidence":        confidence,
        "retrieved_doc_ids": doc_ids,
        "reason":            reason.strip(),
        # Output schema fields
        "status":            output_status,
        "product_area":      ecosystem,
        "request_type":      mapped_request_type,
        "response":          response,
    }
    result["justification"] = generate_justification(result)
    return result

# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def generate_justification(r):
    parts = [f"Ecosystem: {r['ecosystem']}."]
    parts.append(f"Risk: {r['risk_level']}.")
    if r["retrieved_doc_ids"]:
        docs = ", ".join(r["retrieved_doc_ids"][:2])
        parts.append(f"Supporting docs: {docs}.")
    parts.append(r["reason"])
    return " ".join(parts)


def write_output_csv(results):
    fields = ["ticket_id", "status", "product_area", "request_type",
              "response", "justification"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "ticket_id":     r["ticket_id"],
                "status":        r["status"],
                "product_area":  r["product_area"],
                "request_type":  r["request_type"],
                "response":      r["response"],
                "justification": r["justification"],
            })


def write_log(results):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"SupportTriageAgent  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            snippet = r["issue"][:80] + ("..." if len(r["issue"]) > 80 else "")
            f.write(f"TICKET {r['ticket_id']}  |  {r['subject'] or '(no subject)'}\n")
            f.write(f"  Issue  : {snippet}\n")
            f.write(f"\n[Triage]     ecosystem    = {r['ecosystem']}  (company: {r['company']})\n")
            f.write(f"[Classifier] request_type = {r['_request_type_internal']} -> {r['request_type']}\n")
            f.write(f"[Classifier] product_area = {r['product_area']}\n")
            f.write(f"[Risk]       level={r['risk_level']}  decision={r['decision'].upper()}"
                    f"  confidence={r['confidence']}\n")
            f.write(f"[Retrieval]  docs={r['retrieved_doc_ids']}\n")
            f.write(f"[Reason]     {r['reason']}\n")
            resp_snippet = r["response"][:160] + ("..." if len(r["response"]) > 160 else "")
            f.write(f"[Response]   {resp_snippet}\n")
            f.write("-" * 70 + "\n\n")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print("[SupportTriageAgent] Loading corpus ...")
    corpus  = load_corpus()
    vectors, idf = build_index(corpus)
    print(f"[SupportTriageAgent] {len(corpus)} docs indexed.\n")

    print("[SupportTriageAgent] Loading tickets ...")
    tickets = load_tickets()
    print(f"[SupportTriageAgent] Processing {len(tickets)} tickets ...\n")

    results = []
    for ticket in tickets:
        r = triage(ticket, corpus, vectors, idf)
        results.append(r)
        icon = "+ REPLY   " if r["decision"] == "reply" else "^ ESCALATE"
        print(f"  [{icon}] {r['ticket_id']}  eco={r['ecosystem']:12}"
              f"  type={r['request_type']:16}  conf={r['confidence']}")

    write_output_csv(results)
    write_log(results)

    replied   = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"\n[SupportTriageAgent] Done.")
    print(f"  Replied  : {replied}")
    print(f"  Escalated: {escalated}")
    print(f"  output.csv -> {os.path.basename(OUTPUT_PATH)}")
    print(f"  log.txt    -> {os.path.basename(LOG_PATH)}")


if __name__ == "__main__":
    main()
