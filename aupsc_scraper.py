# au_scrape_psc.py
# AU PSC Communiqués scraper (exact base URL; table-only; language check; pagination; flexible stopping)
#
# Base URL (fixed): https://www.peaceau.org/en/resource/documents?idtype=7
# Only rows inside <table class="category"> are considered.
# Pagination is discovered via <div class="pagination"> (param p).
# Language filter: --lang en|fr|all (matches -en.pdf / -fr.pdf).
# Year filter: --year YYYY (client-side, read from row's date text).
# Stopping controls:
#   --start-page N       : start at page N (default 1)
#   --max-pages N        : scrape at most N pages from start (0 = no manual cap; use site max)
#   --empty-stop N       : stop after N consecutive empty pages (0 = never stop early; 1 = stop on first empty) [default 1]
# Manifest:
#   overwrites by default; add --append to append and dedup by URL.

import os, re, time, csv, argparse
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Optional, Tuple, List, Dict
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.peaceau.org/en/resource/documents?idtype=7"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

def get(url: str, **kw):
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r

def page_url(page: int) -> str:
    return BASE_URL if page <= 1 else f"{BASE_URL}&p={page}"

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "", name)
    return name[:180]

def want_lang(pdf_url: str, lang: str) -> bool:
    """
    Accept -en.pdf / -fr.pdf OR -en-*.pdf / -fr-*.pdf at the end of the filename.
    Examples that match for en:
      .../1234.comm-en.pdf
      .../1234.comm-en-annex.pdf
      .../1234.comm-en-v2.pdf
    """
    u = (pdf_url or "").lower()
    if lang == "en":
        return bool(re.search(r"-en(?:-[^/]+)?\.pdf$", u))
    if lang == "fr":
        return bool(re.search(r"-fr(?:-[^/]+)?\.pdf$", u))
    return True  # 'all'


def parse_row_year(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract year (and dd/mm/yyyy if present) from a table row's text."""
    if not text:
        return None, None
    m = re.search(r"\b(3[01]|[12]?\d)/(1[0-2]|0?\d)/((19|20)\d{2})\b", text)  # dd/mm/yyyy
    if m:
        return int(m.group(3)), m.group(0)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return (int(m.group(0)), None) if m else (None, None)

def discover_max_page(soup: BeautifulSoup) -> int:
    """Read <div class='pagination'> and return the highest page number listed."""
    max_p = 1
    pager = soup.find("div", class_="pagination")
    if not pager:
        return max_p
    for a in pager.find_all("a", href=True):
        try:
            qs = parse_qs(urlparse(a["href"]).query)
            p = int(qs.get("p", [None])[0]) if "p" in qs else None
            if p and p > max_p:
                max_p = p
        except Exception:
            continue
    return max_p

def scrape_table_rows(html: str, lang: str, year: Optional[int], page_num: int) -> Tuple[List[Dict], int]:
    """
    Parse a single page and return (rows, max_page).
    Only considers <table class='category'> rows; applies language and (optional) year filter here.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="category")
    max_page = discover_max_page(soup)
    if not table:
        print(f"[WARN] Page {page_num}: no <table class='category'> found.")
        return [], max_page

    rows, seen = [], set()
    for tr in table.find_all("tr"):
        anchors = tr.find_all("a", href=True)
        pdf_as = [a for a in anchors if a["href"].lower().endswith(".pdf")]
        if not pdf_as:
            continue

        row_text = tr.get_text(" ", strip=True)
        yr, date_str = parse_row_year(row_text)
        # client-side year filter (only if we could read year in that row)
        if year is not None and yr is not None and yr != year:
            continue

        for a in pdf_as:
            pdf_url = urljoin(BASE_URL, a["href"])
            if not want_lang(pdf_url, lang):
                continue
            if pdf_url in seen:
                continue
            seen.add(pdf_url)
            title = a.get_text(" ", strip=True) or "PSC Communiqué"
            rows.append({
                "page": page_num,
                "title": title,
                "url": pdf_url,
                "date": date_str or "",
                "year": yr or (year if year is not None else ""),
                "row_text": row_text
            })
    return rows, max_page

def download_pdf(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = safe_filename(os.path.basename(url.split("?")[0]))
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    with get(url, stream=True) as r, open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    return out_path

def write_manifest(rows: List[Dict], append: bool = False) -> str:
    manifest_path = os.path.join("data", "manifest.csv")
    fieldnames = ["page", "year", "date", "title", "url", "filename", "path", "size_bytes", "row_text"]

    existing_urls = set()
    write_header = True
    mode = "w"

    if append and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("url"):
                        existing_urls.add(row["url"])
            mode = "a"
            write_header = False
        except Exception:
            mode = "w"; write_header = True

    # de-dupe by URL
    out_rows = []
    for r in rows:
        for k in fieldnames:
            r.setdefault(k, "")
        if append and r["url"] in existing_urls:
            continue
        out_rows.append(r)

    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in out_rows:
            w.writerow(r)

    verb = "Appended" if mode == "a" else "Saved"
    print(f"[DONE] {verb} {len(out_rows)} entrie(s) to {manifest_path}")
    return manifest_path

def run(lang: str = "en",
        year: Optional[int] = None,
        pause: float = 0.7,
        start_page: int = 1,
        max_pages: int = 0,
        empty_stop: int = 1,
        append: bool = False):
    os.makedirs("data", exist_ok=True)
    pdf_dir = os.path.join("data", "pdfs")
    manifest_rows: List[Dict] = []

    # First page: discover site max and compute last page to scan.
    html = get(page_url(start_page)).text
    page_rows, site_max = scrape_table_rows(html, lang, year, start_page)

    if max_pages and max_pages > 0:
        last_page = min(site_max, start_page + max_pages - 1)
    else:
        last_page = max(site_max, start_page)

    print(f"[INFO] Site shows up to page {site_max}. Will scan pages {start_page}..{last_page}.")

    # empty-page stopping control
    empty_run = 0

    # Process current (start) page
    if not page_rows:
        empty_run += 1
        print(f"[INFO] Page {start_page}: no matching PDFs. (empty_run={empty_run})")
        if empty_stop > 0 and empty_run >= empty_stop:
            print(f"[INFO] Stopping (empty_stop={empty_stop}).")
            return write_manifest(manifest_rows, append=append)
    else:
        print(f"[INFO] Page {start_page}: {len(page_rows)} matching PDF(s).")
        for r in page_rows:
            try:
                path = download_pdf(r["url"], pdf_dir)
                r["filename"] = os.path.basename(path)
                r["path"] = path.replace("\\", "/")
                r["size_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
                manifest_rows.append(r)
                time.sleep(pause)
            except Exception as e:
                print(f"[WARN] {r['url']}: {e}")

    # Loop remaining pages
    for p in range(start_page + 1, last_page + 1):
        html = get(page_url(p)).text
        page_rows, _ = scrape_table_rows(html, lang, year, p)

        if not page_rows:
            empty_run += 1
            print(f"[INFO] Page {p}: no matching PDFs. (empty_run={empty_run})")
            if empty_stop > 0 and empty_run >= empty_stop:
                print(f"[INFO] Stopping (empty_stop={empty_stop}).")
                break
            # else continue scanning
            continue
        else:
            empty_run = 0
            print(f"[INFO] Page {p}: {len(page_rows)} matching PDF(s).")

        for r in page_rows:
            try:
                path = download_pdf(r["url"], pdf_dir)
                r["filename"] = os.path.basename(path)
                r["path"] = path.replace("\\", "/")
                r["size_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
                manifest_rows.append(r)
                time.sleep(pause)
            except Exception as e:
                print(f"[WARN] {r['url']}: {e}")

    write_manifest(manifest_rows, append=append)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape AU PSC Communiqués from <table class='category'> with flexible stopping.")
    ap.add_argument("--lang", choices=["en","fr","all"], default="en", help="Download only -en.pdf, -fr.pdf, or all PDFs")
    ap.add_argument("--year", type=int, default=None, help="Keep only rows whose Date shows this year (client-side filter)")
    ap.add_argument("--pause", type=float, default=0.7, help="Pause (seconds) between downloads")
    ap.add_argument("--start-page", type=int, default=1, help="Page number to start from (1-based)")
    ap.add_argument("--max-pages", type=int, default=0, help="Max pages to scan from start (0 = no manual cap)")
    ap.add_argument("--empty-stop", type=int, default=1,
                    help="Stop after N consecutive empty pages (0 = never stop early; 1 = stop on first empty)")
    ap.add_argument("--append", action="store_true", help="Append to data/manifest.csv and de-duplicate by URL")
    args = ap.parse_args()
    run(args.lang, args.year, args.pause, args.start_page, args.max_pages, args.empty_stop, args.append)
