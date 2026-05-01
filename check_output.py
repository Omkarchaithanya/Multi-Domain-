import csv

with open('support_tickets/support_tickets/output.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# ── CHECK 1: Response quality (first 150 chars of each replied ticket)
print("=" * 70)
print("CHECK 1 — RESPONSE QUALITY (replied tickets)")
print("=" * 70)
for r in rows:
    if r['status'] == 'replied':
        preview = r['response'][:150].replace('\n', ' ')
        print(f"\n[{r['ticket_id']}] ({r['product_area']} | {r['request_type']})")
        print(f"  RESPONSE: {preview}...")

# ── CHECK 2: Justification clarity
print("\n" + "=" * 70)
print("CHECK 2 — JUSTIFICATION FIELD")
print("=" * 70)
for r in rows:
    print(f"\n[{r['ticket_id']}] {r['status'].upper()}")
    print(f"  JUSTIFICATION: {r['justification']}")

# ── CHECK 3: Escalated tickets — are they genuinely high risk?
print("\n" + "=" * 70)
print("CHECK 3 — ESCALATED TICKETS (should be high-risk)")
print("=" * 70)
for r in rows:
    if r['status'] == 'escalated':
        print(f"\n[{r['ticket_id']}] product_area={r['product_area']} | request_type={r['request_type']}")
        print(f"  JUSTIFICATION : {r['justification']}")
        print(f"  RESPONSE      : {r['response'][:120].replace(chr(10), ' ')}...")