import csv
path=r"C:\Users\omkar\Downloads\Multi‑Domain\support_tickets\support_tickets\output.csv"
with open(path, encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
print('data_rows', len(rows))
empty = []
for row in rows:
    for k in ['ticket_id','request_type','product_area','action','retrieved_docs','response']:
        if (row.get(k) or '').strip() == '':
            empty.append((row.get('ticket_id'), k))
print('empty_count', len(empty))
if empty:
    print('examples_empty', empty[:5])
invalid_action = [r for r in rows if r.get('action') not in ('reply','escalate')]
print('invalid_action_count', len(invalid_action))
if invalid_action:
    print('invalid_action_examples', [(r.get('ticket_id'), r.get('action')) for r in invalid_action])
invalid_product = [r for r in rows if r.get('product_area') not in ('hackerrank','claude','visa')]
print('invalid_product_count', len(invalid_product))
if invalid_product:
    print('invalid_product_examples', [(r.get('ticket_id'), r.get('product_area')) for r in invalid_product])
