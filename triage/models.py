"""Data models and taxonomy constants."""
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

ECOSYSTEMS = ["hackerrank", "claude", "visa", "unknown"]

REQUEST_TYPES = [
    "billing", "technical_issue", "account_access", "fraud_or_security",
    "assessment_or_content", "feature_or_usage_question",
    "feedback_or_complaint", "other",
]

PRODUCT_AREAS = {
    "hackerrank": ["candidate_platform", "employer_platform", "assessments",
                   "interviews", "community_or_contests", "billing",
                   "account_access", "other"],
    "claude":     ["workspaces", "billing", "api_or_integrations",
                   "product_access", "safety_or_policy", "data_privacy",
                   "account_access", "other"],
    "visa":       ["cardholder_support", "merchant_support",
                   "disputes_or_chargebacks", "online_payments",
                   "security_or_fraud", "account_access", "billing", "other"],
    "unknown":    ["other"],
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    ticket_id: str
    issue: str
    subject: str
    company: str

    @property
    def text(self) -> str:
        return f"{self.issue} {self.subject}"


@dataclass
class TriageResult:
    ticket_id: str
    issue: str
    subject: str
    company: str
    ecosystem: str
    request_type: str
    product_area: str
    risk_level: str
    decision: str          # "reply" | "escalate"
    confidence: float
    retrieved_doc_ids: List[str]
    reason: str
    response: str

    @property
    def status(self) -> str:
        return "Escalated" if self.decision == "escalate" else "Replied"

    def to_csv_row(self) -> dict:
        return {
            "issue":         self.issue,
            "subject":       self.subject,
            "company":       self.company,
            "response":      self.response,
            "product_area":  self.product_area,
            "status":        self.status,
            "request_type":  self.request_type,
            "justification": self.reason,
        }
