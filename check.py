import csv
from collections import Counter

with open('support_tickets/support_tickets/output.csv') as f:
    rows = list(csv.DictReader(f))

print(f'Total rows: {len(rows)} (expected: 29)')

required = ['ticket_id','request_type','product_area','action','retrieved_docs','response']
errors = []
for r in rows:
    for col in required:
        if not r.get(col,'').strip():
            errors.append(f"EMPTY: {r['ticket_id']} -> {col}")

if errors:
    for e in errors: print(e)
else:
    print('All fields populated - no empty cells')

print()
for k,v in Counter(r['action'] for r in rows).items(): print(f'  {k}: {v}')
for k,v in Counter(r['product_area'] for r in rows).items(): print(f'  {k}: {v}')
print()
print('First 3 rows:')
for r in rows[:3]: print(f"  {r['ticket_id']} | {r['product_area']} | {r['request_type']} | {r['action']}")
