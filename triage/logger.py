"""Internal reasoning transcript writer."""
from datetime import datetime
from typing import List


def write_log(results: List, log_path: str) -> None:
    """Write a chat-like triage transcript to log_path."""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"SupportTriageAgent  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            # Support both TriageResult dataclass and plain dict
            tid      = r.ticket_id   if hasattr(r, "ticket_id")   else r["ticket_id"]
            subj     = r.subject     if hasattr(r, "subject")      else r["subject"]
            issue    = r.issue       if hasattr(r, "issue")        else r["issue"]
            eco      = r.ecosystem   if hasattr(r, "ecosystem")    else r["ecosystem"]
            company  = r.company     if hasattr(r, "company")      else r["company"]
            rtype    = r.request_type if hasattr(r, "request_type") else r["request_type"]
            parea    = r.product_area if hasattr(r, "product_area") else r["product_area"]
            risk     = r.risk_level  if hasattr(r, "risk_level")   else r["risk_level"]
            decision = r.decision    if hasattr(r, "decision")     else r["decision"]
            conf     = r.confidence  if hasattr(r, "confidence")   else r["confidence"]
            doc_ids  = r.retrieved_doc_ids if hasattr(r, "retrieved_doc_ids") else r["retrieved_doc_ids"]
            reason   = r.reason      if hasattr(r, "reason")       else r["reason"]
            response = r.response    if hasattr(r, "response")     else r["response"]

            snippet  = issue[:80] + ("..." if len(issue) > 80 else "")
            f.write("=== TICKET: ===\n")
            f.write(f"TICKET {tid}  |  {subj or '(no subject)'}\n")
            f.write(f"  Issue  : {snippet}\n")
            f.write(f"\n[Triage]     ecosystem    = {eco}  (company: {company})\n")
            f.write(f"[Classifier] request_type = {rtype}\n")
            f.write(f"[Classifier] product_area = {parea}\n")
            f.write(f"[Risk]       level={risk}  decision={decision.upper()}"
                    f"  confidence={conf}\n")
            f.write(f"[Retrieval]  docs={doc_ids}\n")
            f.write(f"[Reason]     {reason}\n")
            resp_snip = response[:160] + ("..." if len(response) > 160 else "")
            f.write(f"[Response]   {resp_snip}\n")
            f.write("-" * 70 + "\n\n")
