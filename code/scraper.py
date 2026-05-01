"""
Support-site crawler — scrapes HackerRank, Claude, and Visa help centres.
Writes JSON chunks to data/{domain}/ for use by CorpusIndexer.

Usage:
    python code/scraper.py          # from repo root
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Set
from urllib.parse import urljoin, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: run  pip install requests beautifulsoup4"
    ) from exc

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TARGETS = {
    "hackerrank": "https://support.hackerrank.com/hc/en-us",
    "claude":     "https://support.claude.com/en/",
    "visa":       "https://www.visa.co.in/support.html",
}

MAX_PAGES_PER_DOMAIN = 60
MAX_DEPTH            = 2
CHUNK_WORDS          = 400
OVERLAP_WORDS        = 50
REQUEST_DELAY        = 0.5   # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SupportTriageBot/1.0; "
        "+https://github.com/anthropics/support-triage)"
    )
}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(soup: BeautifulSoup) -> str:
    """Pull readable text from paragraph and list elements."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    parts: List[str] = []
    for el in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "td", "dt", "dd"]):
        text = el.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _page_title(soup: BeautifulSoup) -> str:
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(" ", strip=True)
        # Strip " | Site Name" suffix (e.g. "| HackerRank Knowledge Base")
        title = re.sub(r"\s*\|.*$", "", title).strip()
        return title
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap
    return chunks


# ---------------------------------------------------------------------------
# Link discovery
# ---------------------------------------------------------------------------

def _same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def _collect_links(soup: BeautifulSoup, current_url: str, base_url: str) -> List[str]:
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        full = urljoin(current_url, href)
        full = full.split("#")[0]  # strip fragment
        if _same_domain(full, base_url) and full not in links:
            links.append(full)
    return links


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
            return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        print(f"    [skip] {url}  ({exc})")
    return None


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl_domain(
    domain: str,
    start_url: str,
    out_dir: Path,
    max_pages: int = MAX_PAGES_PER_DOMAIN,
    max_depth: int = MAX_DEPTH,
) -> int:
    """Crawl start_url up to max_depth and write chunks to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    visited:  Set[str] = set()
    frontier: List[tuple[str, int]] = [(start_url, 0)]   # (url, depth)
    total_chunks = 0
    pages_scraped = 0

    while frontier and pages_scraped < max_pages:
        url, depth = frontier.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"  [{domain}] depth={depth} ({pages_scraped+1}/{max_pages})  {url}")
        soup = _get(url, session)
        time.sleep(REQUEST_DELAY)
        if soup is None:
            continue

        pages_scraped += 1

        # Always enqueue links before deciding whether to save text
        if depth < max_depth:
            for link in _collect_links(soup, url, start_url):
                if link not in visited:
                    frontier.append((link, depth + 1))

        title = _page_title(soup)
        text  = _extract_text(soup)
        if not text:
            continue

        full_text = (title + " " + text).strip() if title else text
        chunks = _chunk_text(full_text)

        # Derive a short filename slug from the URL path
        path_slug = re.sub(r"[^a-z0-9_-]", "_", urlparse(url).path.strip("/").lower())[:80] or "index"
        for chunk_idx, chunk in enumerate(chunks):
            fname = f"{path_slug}_{chunk_idx}.json"
            payload = {
                "source": fname,
                "domain": domain,
                "text":   chunk,
                "url":    url,
            }
            (out_dir / fname).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        total_chunks += len(chunks)

    return total_chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[scraper] Writing corpus to {DATA_DIR}\n")
    summary: dict[str, dict] = {}

    for domain, start_url in TARGETS.items():
        out_dir = DATA_DIR / domain
        print(f"[scraper] === {domain.upper()} ===  {start_url}")
        try:
            n_chunks = crawl_domain(domain, start_url, out_dir)
            existing = list(out_dir.glob("*.json"))
            summary[domain] = {"pages": len(existing), "chunks": n_chunks}
        except Exception as exc:
            print(f"[scraper] ERROR for {domain}: {exc}")
            summary[domain] = {"pages": 0, "chunks": 0, "error": str(exc)}
        print()

    print("[scraper] Summary:")
    for domain, stats in summary.items():
        if "error" in stats:
            print(f"  {domain:12}  ERROR: {stats['error']}")
        else:
            print(f"  {domain:12}  pages={stats['pages']:3}  chunks={stats['chunks']:4}")
    print("\n[scraper] Done. Re-run main.py to rebuild corpus indexes.")


if __name__ == "__main__":
    main()
