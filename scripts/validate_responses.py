import csv
import sys
import os

path = os.path.join('support_tickets','support_tickets','output.csv')
if not os.path.exists(path):
    print('ERROR: output.csv not found at', path)
    sys.exit(3)

problems = []
total = 0
replied = 0

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        status = (row.get('status') or '').strip().lower()
        if status != 'replied':
            continue
        replied += 1
        resp = (row.get('response') or '').strip()
        tid = row.get('ticket_id') or f'row-{total}'
        if len(resp) < 10:
            problems.append((tid, 'too_short', resp))
            continue
        first = resp[0]
        if not first.isupper():
            problems.append((tid, 'not_capitalized', resp[:80]))
        if not (resp.endswith('.') or resp.endswith('!') or resp.endswith('?')):
            problems.append((tid, 'no_sentence_ending_punct', resp[-80:]))
        closing = 'Let us know if you need further help.'
        if not resp.endswith(closing):
            problems.append((tid, 'missing_closing_phrase', resp[-200:]))

if not problems:
    print(f'OK: {replied} replied rows passed validation (total {total})')
    sys.exit(0)

print(f'FAIL: {len(problems)} problems found across {replied} replied rows (total {total})')
for tid, reason, sample in problems:
    print(tid, reason, ':', sample)
sys.exit(2)
