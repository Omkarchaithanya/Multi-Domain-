"""Corpus loading — reads sample CSV plus any extra docs in data/."""
import csv
import os
from typing import List, Dict

# Supplementary docs provided as evidence during the triage session
EXTRA_CORPUS_DOCS: List[Dict] = [
    {
        "doc_id":       "visa_014",
        "issue":        "unauthorized transactions on visa card",
        "subject":      "How to report unauthorized transactions",
        "company":      "Visa",
        "response":     (
            "If you notice transactions you did not make, contact your card issuer "
            "immediately and follow the dispute process described by your issuer."
        ),
        "product_area": "security_or_fraud",
        "status":       "Replied",
        "request_type": "fraud_or_security",
        "text":         (
            "unauthorized transactions visa card contact card issuer immediately "
            "dispute process"
        ),
    },
    {
        "doc_id":       "visa_022",
        "issue":        "how to dispute a charge on visa card chargeback",
        "subject":      "Visa dispute and chargeback overview",
        "company":      "Visa",
        "response":     (
            "To dispute a charge on your Visa card, please contact your card issuer "
            "directly. Chargeback eligibility and timelines vary depending on the issuer "
            "and the type of dispute. Your issuer will guide you through the steps "
            "applicable to your specific case."
        ),
        "product_area": "disputes_or_chargebacks",
        "status":       "Replied",
        "request_type": "disputes_or_chargebacks",
        "text":         (
            "dispute charge visa card chargeback eligibility timelines issuer "
            "dispute category contact issuer next steps"
        ),
    },
]


def _row_to_doc(row: Dict, doc_id: str) -> Dict:
    text = " ".join(filter(None, [
        row.get("Issue", ""),
        row.get("Subject", ""),
        row.get("Response", ""),
    ]))
    return {
        "doc_id":       doc_id,
        "issue":        row.get("Issue", ""),
        "subject":      row.get("Subject", ""),
        "company":      row.get("Company", ""),
        "response":     row.get("Response", ""),
        "product_area": row.get("Product Area", ""),
        "status":       row.get("Status", ""),
        "request_type": row.get("Request Type", ""),
        "text":         text,
    }


def load_corpus(corpus_path: str, data_dir: str = None) -> List[Dict]:
    """
    Load the main sample corpus CSV and any additional .csv files found
    in data_dir (if provided).  Returns a list of doc dicts.
    """
    corpus: List[Dict] = []

    # Primary corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            corpus.append(_row_to_doc(row, f"sample-doc-{i+1}"))

    # Additional CSV files in data/ directory
    if data_dir and os.path.isdir(data_dir):
        extra_id = len(corpus) + 1
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(data_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and "Issue" in reader.fieldnames:
                        for row in reader:
                            corpus.append(_row_to_doc(row, f"data-doc-{extra_id}"))
                            extra_id += 1
            except Exception:
                pass  # Skip unreadable files silently

    corpus.extend(EXTRA_CORPUS_DOCS)
    return corpus
