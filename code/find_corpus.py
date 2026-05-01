import os
from pathlib import Path

# Search common locations for the actual corpus files
search_roots = [
    r"C:\Users\omkar\Downloads",
    r"C:\Users\omkar\Desktop",
    r"C:\Users\omkar\Documents",
]

print("=== Searching for support corpus files ===")
for root in search_roots:
    for path in Path(root).rglob("*.txt"):
        if any(x in str(path).lower() for x in ["support", "corpus", "hackerrank", "claude", "visa", "help"]):
            print(f"  FOUND: {path} ({path.stat().st_size} bytes)")

data_dir = Path(r"C:\Users\omkar\Downloads\Multi-Domain\data")
# Also list EVERYTHING inside the data/ folder recursively
data_dir_candidates = [
    Path(r"C:\Users\omkar\Downloads\Multi-Domain\data"),
    Path(r"C:\Users\omkar\Downloads\Multi‑Domain\data"),
]

data_dir = next((path for path in data_dir_candidates if path.exists()), data_dir_candidates[0])
print(f"\n=== Full recursive listing of {data_dir} ===")
if data_dir.exists():
    for f in data_dir.rglob("*"):
        print(f"  {f} ({'DIR' if f.is_dir() else f.stat().st_size}b)")
else:
    print("  [data directory not found]")
