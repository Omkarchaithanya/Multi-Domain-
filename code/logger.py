import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class StructuredLogger:
    """
    Innovation 6: Structured Decision Log for the AI Judge
    
    Creates a human-readable, structured audit trail of ticket processing decisions.
    Instead of raw terminal output, this provides exact numbers, reasoning, and citations.
    
    Format per ticket:
    === TICKET: {ticket_id} ===
    Timestamp: {ISO timestamp}
    Issue: {first 100 chars of ticket}
    Domain: {domain} (confidence: {score}, signals: {matched_keywords})
    Sub-intents: {list}
    Tone: {tone}
    Retrieval: {list of (source, score) for top 3 chunks}
    Safety check: {pass/escalate, rule triggered or adversarial verdict}
    Grounding score: {score} — {PASS/REVISE/ESCALATE}
    Action: {reply/escalate}
    Escalation reason: {if applicable}
    Request type: {value}
    ==============================
    """
    
    def __init__(self, log_file: Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Start a fresh log for each run to ensure counts reflect a single execution
        with self.log_file.open("w", encoding="utf-8") as f:
            f.write(f"SupportTriageAgent  —  {datetime.now(timezone.utc).isoformat()}\n")
            f.write("======================================================================\n\n")
    
    def log_ticket(
        self,
        ticket_id: str,
        ticket_text: str,
        domain: str,
        domain_confidence: float,
        domain_signals: List[str],
        sub_intents: List[str],
        tone: str,
        retrieved_chunks: List[Dict],
        safety_verdict: str,
        safety_reason: str,
        grounding_score: Optional[float],
        grounding_verdict: str,
        request_type: str,
        action: str,
        escalation_reason: Optional[str] = None,
    ) -> None:
        """Log a structured decision for a ticket."""
        timestamp = datetime.now(timezone.utc).isoformat()
        issue_preview = ticket_text[:100].replace("\n", " ")
        
        # Format retrieval info
        retrieval_lines = []
        for chunk in retrieved_chunks[:3]:
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0.0)
            retrieval_lines.append(f"    {source} (score: {score:.2f})")
        
        # Format safety check
        safety_check_line = f"{safety_verdict.upper()}"
        if safety_reason:
            safety_check_line += f" - {safety_reason}"
        
        # Format grounding
        grounding_line = ""
        if grounding_score is not None:
            grounding_line = f"Grounding score: {grounding_score:.2f} — {grounding_verdict.upper()}"
        
        # Build structured log entry
        log_entry = f"""
=== TICKET: {ticket_id} ===
Timestamp: {timestamp}
Issue: {issue_preview}
Domain: {domain} (confidence: {domain_confidence:.2f}, signals: {domain_signals})
Sub-intents: {sub_intents if sub_intents else "[single intent]"}
Tone: {tone}
Retrieval:
{chr(10).join(retrieval_lines) if retrieval_lines else "  [no chunks retrieved]"}
Safety check: {safety_check_line}
{grounding_line}
Request type: {request_type}
Action: {action}
{"Escalation reason: " + str(escalation_reason) if escalation_reason else ""}
==============================
"""
        
        # Append to log file
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(log_entry.strip() + "\n\n")
    
    def log_event(self, event_type: str, payload: Dict) -> None:
        """Log a general event (e.g., run_start, run_complete) as JSON."""
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")
