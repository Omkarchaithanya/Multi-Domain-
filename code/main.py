import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from corpus_indexer import CorpusIndexer
from domain_router import route_domain_with_confidence
from escalation import classify_request_type_with_confidence, escalation_decision, adversarial_check
from intent_decomposer import decompose_ticket
from tone_detector import detect_tone
from response_generator import UNCOVERED_RESPONSE, generate_response, verify_grounding
from retriever import retrieve
from logger import StructuredLogger


OUTPUT_FIELDS = ["ticket_id", "request_type", "product_area", "action", "retrieved_docs", "response"]
ESCALATION_RESPONSE = "This issue requires attention from a specialist. A human support agent will review your case and follow up shortly."


def main() -> None:
    data_dir = ROOT_DIR / "data"
    ticket_dir = _find_ticket_dir(ROOT_DIR)
    input_path = ticket_dir / "support_tickets.csv"
    sample_path = ticket_dir / "sample_support_tickets.csv"
    output_path = ticket_dir / "output.csv"
    log_path = ticket_dir / "log.txt"

    print("[triage] Building corpus indexes...")
    indexer = CorpusIndexer(str(data_dir))
    indexer.build()
    print(
        f"[triage] Indexed {len(indexer.chunks)} chunks "
        f"across {sum(1 for chunks in indexer.domain_chunks.values() if chunks)} domains "
        f"(semantic={indexer.semantic_model_name})."
    )
    if indexer.used_csv_fallback:
        print("[triage] data/ corpus not found; using local sample CSV fallback for this workspace.")

    # Initialize structured logger
    structured_logger = StructuredLogger(log_path)
    structured_logger.log_event("run_start", {
        "chunks": len(indexer.chunks),
        "semantic_model": indexer.semantic_model_name,
        "csv_fallback": indexer.used_csv_fallback,
    })

    _log_event(log_path, "run_start", {
        "chunks": len(indexer.chunks),
        "semantic_model": indexer.semantic_model_name,
        "csv_fallback": indexer.used_csv_fallback,
    })

    if sample_path.exists():
        _validate_sample(sample_path, indexer, log_path)

    rows = list(_read_csv(input_path))
    print(f"[triage] Processing {len(rows)} tickets from {input_path}...")
    results = []
    for idx, row in enumerate(rows, start=1):
        result = process_ticket(row, idx, indexer, log_path, structured_logger)
        results.append(result)
        print(f"[{result['ticket_id']}] {result['product_area']} | {result['request_type']} | {result['action']}")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    structured_logger.log_event("run_complete", {
        "tickets": len(results),
        "reply": sum(1 for row in results if row["action"] == "reply"),
        "escalate": sum(1 for row in results if row["action"] == "escalate"),
        "output": str(output_path),
    })
    _log_event(log_path, "run_complete", {
        "tickets": len(results),
        "reply": sum(1 for row in results if row["action"] == "reply"),
        "escalate": sum(1 for row in results if row["action"] == "escalate"),
        "output": str(output_path),
    })
    print(f"[triage] Wrote {output_path}")
    print(f"[triage] Appended structured log to {log_path}")
    
    # Summary of request types
    request_type_counts = {}
    for result in results:
        req_type = result.get("request_type", "unknown")
        request_type_counts[req_type] = request_type_counts.get(req_type, 0) + 1
    
    print("\n[triage] Request Type Distribution:")
    for req_type in sorted(request_type_counts.keys()):
        count = request_type_counts[req_type]
        print(f"  {req_type:20} {count:3d}")
    
    other_count = request_type_counts.get("other", 0)
    if other_count > 0:
        print(f"\n[triage] ⚠ WARNING: {other_count} tickets classified as 'other' (target: < 3)")


def process_ticket(
    row: Dict[str, str],
    idx: int,
    indexer: CorpusIndexer,
    log_path: Path,
    structured_logger: StructuredLogger,
) -> Dict[str, str]:
    ticket_id = _first(row, ("ticket_id", "Ticket ID", "id", "ID")) or f"ticket-{idx:03d}"
    ticket_text = _ticket_text(row)

    # Innovation 1: Multi-Intent Decomposition
    sub_intents = decompose_ticket(ticket_text)
    
    # Innovation 5: Tone Detection
    tone = detect_tone(ticket_text)
    
    # Domain routing
    domain, domain_confidence, domain_reason = route_domain_with_confidence(ticket_text)
    
    # Request type classification
    request_type, request_confidence = classify_request_type_with_confidence(ticket_text)
    
    # Retrieval
    chunks = retrieve(ticket_text, domain, indexer)
    
    # Diagnostic output for first 4 tickets
    if idx <= 4:
        print(f"\n[DIAG] Ticket {ticket_id} retrieval scores:")
        for chunk in chunks[:3]:
            print(f"  - {chunk.get('source', 'unknown'):30} score={chunk.get('score', 0.0):.3f} domain={chunk.get('domain', '?')}")
        if not chunks:
            print(f"  - No chunks retrieved")
    
    # Innovation 2: Adversarial Safety Check
    safety_verdict, safety_reason = adversarial_check(ticket_text)
    safety_should_escalate = safety_verdict != "safe"
    
    # Basic escalation decision
    should_escalate, escalation_reason = escalation_decision(ticket_text, request_type, chunks)
    
    # Combine safety and escalation decisions
    if safety_should_escalate:
        should_escalate = True
        escalation_reason = f"adversarial_check:{safety_reason}"

    retrieval_confidence = chunks[0]["score"] if chunks else 0.0
    confidence = round((domain_confidence * 0.35) + (request_confidence * 0.25) + (retrieval_confidence * 0.40), 3)
    if confidence < 0.42 and not should_escalate:
        should_escalate = True
        escalation_reason = f"low_confidence:{confidence:.2f}"
    
    # Diagnostic output for first 4 tickets
    if idx <= 4:
        print(f"[DIAG] Confidence={confidence} (domain={domain_confidence}, request={request_confidence}, retrieval={retrieval_confidence})")
        print(f"[DIAG] Escalation={should_escalate} ({escalation_reason})")

    action = "escalate" if should_escalate else "reply"
    grounding_score = None
    grounding_verdict = "n/a"
    
    if should_escalate:
        response = ESCALATION_RESPONSE
        source_citation = _retrieved_docs(chunks[:3])
        if source_citation:
            response = f"{response} Source: {source_citation}."
    else:
        # Generate response with tone adaptation
        response = generate_response(ticket_text, chunks, tone)
        if response == UNCOVERED_RESPONSE:
            action = "escalate"
            escalation_reason = "response_generator_uncovered"
            response = ESCALATION_RESPONSE
            source_citation = _retrieved_docs(chunks[:3])
            if source_citation:
                response = f"{response} Source: {source_citation}."
        else:
            # Verify grounding (Innovation 3 is integrated into generate_response)
            grounding_score, grounding_verdict, _ = verify_grounding(response, chunks)

    retrieved_docs = _retrieved_docs(chunks)
    result = {
        "ticket_id": ticket_id,
        "request_type": request_type,
        "product_area": domain,
        "action": action,
        "retrieved_docs": retrieved_docs,
        "response": response,
    }

    # Extract domain signals for logging
    domain_signals = domain_reason.split("=")[-1] if "=" in domain_reason else []

    # Log structured decision (Innovation 6)
    structured_logger.log_ticket(
        ticket_id=ticket_id,
        ticket_text=ticket_text,
        domain=domain,
        domain_confidence=domain_confidence,
        domain_signals=domain_signals,
        sub_intents=sub_intents,
        tone=tone,
        retrieved_chunks=chunks[:3],
        safety_verdict=safety_verdict,
        safety_reason=safety_reason,
        grounding_score=grounding_score,
        grounding_verdict=grounding_verdict,
        request_type=request_type,
        action=action,
        escalation_reason=escalation_reason if should_escalate else None,
    )

    # Also log to JSON log for backward compatibility
    _log_event(log_path, "ticket_decision", {
        "ticket_id": ticket_id,
        "domain": domain,
        "domain_confidence": domain_confidence,
        "domain_reason": domain_reason,
        "request_type": request_type,
        "request_confidence": request_confidence,
        "tone": tone,
        "sub_intents": sub_intents,
        "safety_verdict": safety_verdict,
        "grounding_score": grounding_score,
        "retrieval": [
            {
                "source": chunk["source"],
                "score": chunk["score"],
                "bm25_score": chunk.get("bm25_score"),
                "semantic_score": chunk.get("semantic_score"),
            }
            for chunk in chunks
        ],
        "confidence": confidence,
        "action": action,
        "escalation_reason": escalation_reason,
        "response": response,
    })
    return result


def _validate_sample(sample_path: Path, indexer: CorpusIndexer, log_path: Path) -> None:
    rows = list(_read_csv(sample_path))
    if not rows:
        return

    domain_hits = 0
    domain_total = 0
    action_hits = 0
    comparable_actions = 0
    failures = []
    unlabeled = []
    
    for idx, row in enumerate(rows, start=1):
        ticket_text = _ticket_text(row)
        expected_domain = _normalize_domain(row.get("Company", ""))
        predicted_domain, _, _ = route_domain_with_confidence(ticket_text)
        if expected_domain:
            domain_total += 1
            if predicted_domain == expected_domain:
                domain_hits += 1
            else:
                failures.append({
                    "idx": idx,
                    "ticket_text": ticket_text[:100],
                    "expected_domain": expected_domain,
                    "predicted_domain": predicted_domain,
                    "expected_action": _expected_action(row),
                    "predicted_action": "",
                    "request_type": "",
                    "chunks": [],
                })
        else:
            unlabeled.append({
                "idx": idx,
                "ticket_text": ticket_text[:100],
                "predicted_domain": predicted_domain,
            })

        expected_action = _expected_action(row)
        if expected_action:
            comparable_actions += 1
            request_type, _ = classify_request_type_with_confidence(ticket_text)
            chunks = retrieve(ticket_text, predicted_domain, indexer)
            predicted_escalate, _ = escalation_decision(ticket_text, request_type, chunks)
            predicted_action = "escalate" if predicted_escalate else "reply"
            if predicted_action == expected_action:
                action_hits += 1
            else:
                # Track action failure
                failures.append({
                    "idx": idx,
                    "ticket_text": ticket_text[:100],
                    "expected_domain": expected_domain,
                    "predicted_domain": predicted_domain,
                    "expected_action": expected_action,
                    "predicted_action": predicted_action,
                    "request_type": request_type,
                    "chunks": chunks[:3],
                })

    metrics = {
        "sample_rows": len(rows),
        "domain_accuracy": round(domain_hits / domain_total, 3) if domain_total else None,
        "action_accuracy": round(action_hits / comparable_actions, 3) if comparable_actions else None,
    }
    print(f"[triage] Sample validation: {metrics}")
    _log_event(log_path, "sample_validation", metrics)
    
    # Print details of first 3 failures
    if failures:
        print(f"\n[triage] ⚠ {len(failures)} validation failures (showing first 3):")
        for failure in failures[:3]:
            print(f"\n[FAIL-{failure['idx']}] {failure['ticket_text']}...")
            if failure.get("expected_domain"):
                print(f"  Expected domain: {failure['expected_domain']}")
                print(f"  Predicted domain: {failure['predicted_domain']}")
            print(f"  Expected: {failure['expected_action']}")
            print(f"  Predicted: {failure['predicted_action']}")
            print(f"  Request Type: {failure['request_type']}")
            print(f"  Top 3 chunks:")
            for chunk in failure['chunks']:
                score = chunk.get('score', 0.0)
                source = chunk.get('source', 'unknown')
                print(f"    - {source:30} score={score:.3f}")
    elif unlabeled:
        print(f"\n[triage] Note: {len(unlabeled)} sample rows have no expected Company label and were excluded from domain accuracy.")
        for row in unlabeled[:3]:
            print(f"  [UNLABELED-{row['idx']}] predicted={row['predicted_domain']} | {row['ticket_text']}...")


def _find_ticket_dir(root: Path) -> Path:
    candidates = [
        root / "support_tickets",
        root / "support_tickets" / "support_tickets",
    ]
    for candidate in candidates:
        if (candidate / "support_tickets.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find support_tickets.csv under support_tickets/.")


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _ticket_text(row: Dict[str, str]) -> str:
    parts = [
        _first(row, ("issue", "Issue", "description", "Description", "body", "Body")),
        _first(row, ("subject", "Subject", "title", "Title")),
        _first(row, ("company", "Company", "product", "Product")),
    ]
    return "\n".join(part for part in parts if part).strip()


def _first(row: Dict[str, str], names: Tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return ""


def _retrieved_docs(chunks: List[Dict]) -> str:
    sources = []
    for chunk in chunks:
        source = chunk.get("source", "")
        if source and source not in sources:
            sources.append(source)
    return ";".join(sources)


def _normalize_domain(value: str) -> str:
    value = value.lower()
    if "hackerrank" in value or "hacker rank" in value:
        return "hackerrank"
    if "claude" in value or "anthropic" in value:
        return "claude"
    if "visa" in value:
        return "visa"
    return ""


def _expected_action(row: Dict[str, str]) -> str:
    status = (row.get("Status") or "").lower()
    if "escalat" in status:
        return "escalate"
    if "repl" in status or "resolved" in status:
        return "reply"
    return ""


def _log_event(log_path: Path, event_type: str, payload: Dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def run_test_mode() -> None:
    """Run 5 comprehensive tests of the triage system."""
    print("\n" + "=" * 70)
    print("MULTI-DOMAIN SUPPORT TRIAGE AGENT - TEST SUITE")
    print("=" * 70 + "\n")
    
    results = []
    
    # Initialize indexer
    data_dir = ROOT_DIR / "data"
    indexer = CorpusIndexer(str(data_dir))
    indexer.build()
    print(f"✓ Corpus indexed: {len(indexer.chunks)} chunks\n")
    
    # TEST 1: CORPUS CHECK
    print("TEST 1: CORPUS CHECK")
    print("-" * 70)
    test1_pass = True
    query = "how do I reset my password"
    for domain in ("hackerrank", "claude", "visa"):
        chunks = retrieve(query, domain, indexer)
        if chunks:
            top_chunk = chunks[0]
            score = top_chunk.get("score", 0.0)
            print(f"  {domain:12} → {top_chunk['source']:30} (score: {score:.3f})")
            if score <= 0.2:
                test1_pass = False
        else:
            print(f"  {domain:12} → NO CHUNKS RETRIEVED")
            test1_pass = False
    print(f"Result: {'PASS' if test1_pass else 'FAIL'}\n")
    results.append(("CORPUS CHECK", test1_pass))
    
    # TEST 2: DOMAIN ROUTING CHECK
    print("TEST 2: DOMAIN ROUTING CHECK")
    print("-" * 70)
    test2_tickets = [
        ("my visa card was declined at a merchant", "visa"),
        ("my HackerRank assessment timed out unfairly", "hackerrank"),
        ("I can't see my Claude conversation history", "claude"),
        ("I was charged twice on my account", "any"),
        ("how do I contact support", "any"),
    ]
    test2_pass = True
    for ticket_text, expected_domain in test2_tickets:
        predicted_domain, confidence, reason = route_domain_with_confidence(ticket_text)
        status = "✓" if expected_domain == "any" or predicted_domain == expected_domain else "✗"
        print(f"  {status} {predicted_domain:12} (expected: {expected_domain:12}) | {ticket_text[:40]}")
        if expected_domain != "any" and predicted_domain != expected_domain:
            test2_pass = False
    print(f"Result: {'PASS' if test2_pass else 'FAIL'}\n")
    results.append(("DOMAIN ROUTING CHECK", test2_pass))
    
    # TEST 3: ESCALATION CHECK
    print("TEST 3: ESCALATION CHECK")
    print("-" * 70)
    escalation_test_cases = [
        ("someone hacked my account and made unauthorized transactions", True),
        ("what are your business hours", False),
    ]
    test3_pass = True
    for ticket_text, should_escalate_expected in escalation_test_cases:
        request_type, _ = classify_request_type_with_confidence(ticket_text)
        domain, _, _ = route_domain_with_confidence(ticket_text)
        chunks = retrieve(ticket_text, domain, indexer)
        should_escalate, reason = escalation_decision(ticket_text, request_type, chunks)
        status = "✓" if should_escalate == should_escalate_expected else "✗"
        action = "ESCALATE" if should_escalate else "REPLY"
        expected_action = "ESCALATE" if should_escalate_expected else "REPLY"
        print(f"  {status} {action:8} (expected: {expected_action:8}) | {reason:40} | {ticket_text[:40]}")
        if should_escalate != should_escalate_expected:
            test3_pass = False
    print(f"Result: {'PASS' if test3_pass else 'FAIL'}\n")
    results.append(("ESCALATION CHECK", test3_pass))
    
    # TEST 4: GROUNDING CHECK
    print("TEST 4: GROUNDING CHECK")
    print("-" * 70)
    query_text = "how do I reset my password"
    domain, _, _ = route_domain_with_confidence(query_text)
    chunks = retrieve(query_text, domain, indexer)
    if chunks:
        response = generate_response(query_text, chunks, tone="neutral")
        grounding_score, grounding_verdict, unsupported = verify_grounding(response, chunks)
        print(f"  Query: {query_text}")
        print(f"  Domain: {domain}")
        print(f"  Response length: {len(response)} chars")
        print(f"  Grounding score: {grounding_score:.3f}")
        print(f"  Grounding verdict: {grounding_verdict}")
        test4_pass = grounding_score > 0.6
        if not test4_pass and unsupported:
            print(f"  Unsupported claims: {unsupported}")
    else:
        print("  NO CHUNKS RETRIEVED")
        test4_pass = False
    print(f"Result: {'PASS' if test4_pass else 'FAIL'}\n")
    results.append(("GROUNDING CHECK", test4_pass))
    
    # TEST 5: OUTPUT FORMAT CHECK
    print("TEST 5: OUTPUT FORMAT CHECK")
    print("-" * 70)
    sample_row = {
        "ticket_id": "test-001",
        "Issue": "I need help with my password reset",
        "Subject": "Password Reset Help",
        "Company": "HackerRank",
    }
    output = process_ticket(sample_row, 1, indexer, ROOT_DIR / "test.log", StructuredLogger(ROOT_DIR / "test.log"))
    required_keys = ["ticket_id", "request_type", "product_area", "action", "retrieved_docs", "response"]
    test5_pass = True
    missing_keys = []
    for key in required_keys:
        if key not in output:
            missing_keys.append(key)
            test5_pass = False
        elif not output[key]:
            print(f"  ✗ {key}: EMPTY")
            test5_pass = False
        else:
            print(f"  ✓ {key}: {str(output[key])[:50]}")
    if missing_keys:
        print(f"  Missing keys: {missing_keys}")
    print(f"Result: {'PASS' if test5_pass else 'FAIL'}\n")
    results.append(("OUTPUT FORMAT CHECK", test5_pass))
    
    # Summary
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    print(f"SUMMARY: {passed}/5 tests passed\n")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    print("=" * 70 + "\n")
    
    # Clean up test log
    try:
        (ROOT_DIR / "test.log").unlink()
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_mode()
    else:
        main()
