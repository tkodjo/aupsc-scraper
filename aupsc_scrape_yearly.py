# au_scrape.py — AUPSC Communiqués scraper with YEAR, pager detection, and auto-stop.
import os, re, time, csv, argparse, unicodedata
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

DEFAULT_BASE_LIST_URL = "https://www.peaceau.org/en/resource/documents?idtype=7"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

# ----------------- Utilities -----------------
def get(url, **kw):
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r

def normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(ch)).lower()

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "", name)
    return name[:180]

DATE_RE_DMY = re.compile(r"\b(0?[1-9]|[12]\d|3[01])[./-](0?[1-9]|1[0-2])[./-]([12]\d{3})\b")
DATE_RE_Y   = re.compile(r"\b(19|20)\d{2}\b")

def parse_date_any(text: str):
    """
    Parse a date from nearby text (dd/mm/yyyy etc.). Returns (year, iso_date_str).
    """
    if not isinstance(text, str): return (None, "")
    t = text.strip()

    m = DATE_RE_DMY.search(t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(y, mo, d).date()
            return (dt.year, dt.isoformat())
        except ValueError:
            pass

    m = DATE_RE_Y.search(t)
    if m:
        y = int(m.group(0))
        return (y, f"{y}-01-01")

    return (None, "")

def is_lang_ok(pdf_url: str, lang: str) -> bool:
    if lang == "all": return True
    if lang == "en": return bool(re.search(r"[-_.]en\.pdf$", pdf_url, re.IGNORECASE))
    if lang == "fr": return bool(re.search(r"[-_.]fr\.pdf$", pdf_url, re.IGNORECASE))
    return True

def is_communique(pdf_url: str, title_norm: str, container_norm: str) -> bool:
    """
    Heuristics:
    - Filenames often like ####.comm-en.pdf / ####.comm-fr.pdf
    - Otherwise detect 'communique/communiqué' in title or container text
    """
    fname = os.path.basename(pdf_url).lower()
    if re.search(r"\bcomm\b", fname):  # '...comm-en.pdf'
        return True
    if "communique" in title_norm or "communique" in container_norm:
        return True
    return False

def mentions_psc(title_norm: str, container_norm: str, page_text_norm: str = "") -> bool:
    needles = [r"\bpeace and security council\b", r"\bpsc\b"]
    haystacks = [title_norm, container_norm, page_text_norm]
    for pat in needles:
        rx = re.compile(pat, re.IGNORECASE)
        if any(rx.search(h or "") for h in haystacks):
            return True
    return False

# ----------------- Paging helpers -----------------
def list_page_url(base_url: str, page: int) -> str:
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}p={page}"

def detect_last_page(html: str, base_url: str) -> int:
    """
    Read pagination anchors and find the max ?p= number.
    If none found, return 1.
    """
    soup = BeautifulSoup(html, "lxml")
    max_p = 1
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "p=" not in href:  # quick check
            continue
        full = urljoin(base_url, href)
        try:
            q = parse_qs(urlparse(full).query)
            if "p" in q:
                pval = int(q["p"][0])
                if pval > max_p:
                    max_p = pval
        except Exception:
            continue
    return max_p

# ----------------- Page parsing -----------------
def extract_items_from_page(html: str, base_url: str):
    """
    Returns list of dicts:
      - pdf_url
      - title
      - article_url
      - container_text
      - date_text
    """
    soup = BeautifulSoup(html, "lxml")
    items = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        pdf_url = urljoin(base_url, href)

        # Find surrounding container to gather context strings (title, date, etc.)
        container = a
        for _ in range(6):
            if not container: break
            if container.name in ("tr", "li", "div", "article", "section"):
                break
            container = container.parent

        container_text = container.get_text(" ", strip=True) if container else ""
        date_text = container_text

        # Try to find a non-PDF anchor in the container to serve as the human title
        title = ""
        article_url = None
        if container:
            title_a = None
            for cand in container.find_all("a", href=True):
                if cand["href"].lower().endswith(".pdf"):
                    continue
                title_a = cand
                break
            if title_a:
                title = title_a.get_text(" ", strip=True)
                article_url = urljoin(base_url, title_a["href"])
            else:
                title = a.get_text(" ", strip=True) or container_text

        items.append({
            "pdf_url": pdf_url,
            "title": title,
            "article_url": article_url,
            "container_text": container_text,
            "date_text": date_text
        })

    # De-duplicate by pdf_url
    seen, uniq = set(), []
    for it in items:
        if it["pdf_url"] in seen: continue
        uniq.append(it); seen.add(it["pdf_url"])
    return uniq

# ----------------- Downloader -----------------
def download_pdf(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = safe_filename(os.path.basename(url.split("?")[0]))
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    with get(url, stream=True) as r:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
    return out_path

# ----------------- Main -----------------
def run(base_url: str,
        year: int,
        start_page: int,
        end_page: int,
        pause: float,
        lang: str,
        out_dir: str,
        strict_psc: bool,
        follow_article: bool,
        auto_stop: bool,
        grace_pages: int,
        detect_last: bool):
    os.makedirs("data", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    seen_pdf = set()
    found_year_hits = 0
    nohit_streak = 0

    # Optional: detect the real last page
    if detect_last:
        first_html = get(list_page_url(base_url, start_page)).text
        detected_last = detect_last_page(first_html, base_url)
        if detected_last and detected_last < end_page:
            print(f"[INFO] Detected last page = {detected_last}. Capping end_page.")
            end_page = detected_last
        # Process page 1 content too, so stash it
        preloaded = {start_page: first_html}
    else:
        preloaded = {}

    for p in range(start_page, end_page + 1):
        if p in preloaded:
            html = preloaded[p]
        else:
            try:
                html = get(list_page_url(base_url, p)).text
            except Exception as e:
                print(f"[WARN] Fetch page {p}: {e}")
                continue

        items = extract_items_from_page(html, base_url)
        if not items:
            print(f"[INFO] No PDFs on page {p}.")
            if auto_stop and found_year_hits > 0:
                nohit_streak += 1
                if nohit_streak >= grace_pages:
                    print(f"[INFO] Auto-stop: no {year} items on the last {grace_pages} pages.")
                    break
            continue

        year_hits_this_page = 0
        print(f"[INFO] Page {p}: found {len(items)} PDFs (pre-filter).")

        for it in tqdm(items, desc=f"Page {p}"):
            pdf_url = it["pdf_url"]
            if pdf_url in seen_pdf:
                continue

            # Language filter
            if not is_lang_ok(pdf_url, lang):
                continue

            title_norm = normalize(it.get("title", ""))
            cont_norm  = normalize(it.get("container_text", ""))

            # Must be a communiqué
            if not is_communique(pdf_url, title_norm, cont_norm):
                continue

            # PSC context (strict by default)
            page_text_norm = ""
            if strict_psc and not mentions_psc(title_norm, cont_norm):
                if follow_article and it.get("article_url"):
                    try:
                        art_html = get(it["article_url"]).text
                        page_text_norm = normalize(BeautifulSoup(art_html, "lxml").get_text(" ", strip=True))
                    except Exception:
                        page_text_norm = ""
                if not mentions_psc(title_norm, cont_norm, page_text_norm):
                    continue

            # YEAR filter: try nearby date; else fallbacks
            y_found, iso_date = parse_date_any(it.get("date_text", ""))
            if y_found is None:
                ty, _ = parse_date_any(it.get("title", ""))
                cy, _ = parse_date_any(it.get("container_text", ""))
                file_years = re.findall(r"(19|20)\d{2}", os.path.basename(pdf_url))
                y_found = ty or cy or (int(file_years[0]) if file_years else None)

            if y_found != year:
                continue

            # Passed all filters
            try:
                path = download_pdf(pdf_url, out_dir)
            except Exception as e:
                print(f"[WARN] Download failed: {pdf_url} -> {e}")
                continue

            seen_pdf.add(pdf_url)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            rows.append({
                "page": p,
                "year": y_found or "",
                "date_iso": iso_date,
                "title": it.get("title", ""),
                "pdf_url": pdf_url,
                "article_url": it.get("article_url", "") or "",
                "filename": os.path.basename(path),
                "path": path.replace("\\", "/"),
                "size_bytes": size,
                "lang": lang,
                "psc_flag": True
            })
            year_hits_this_page += 1
            time.sleep(pause)

        # ---- auto-stop logic
        if auto_stop:
            if year_hits_this_page == 0:
                if found_year_hits > 0:
                    nohit_streak += 1
            else:
                found_year_hits += year_hits_this_page
                nohit_streak = 0

            if found_year_hits > 0 and nohit_streak >= grace_pages:
                print(f"[INFO] Auto-stop: no {year} items on the last {grace_pages} page(s). Stopping at page {p}.")
                break

    # Save manifest
    manifest_path = os.path.join(out_dir, f"psc_communiques_{year}_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "page","year","date_iso","title","pdf_url","article_url",
            "filename","path","size_bytes","lang","psc_flag"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] Saved {len(rows)} communiqué(s) to {out_dir}")
    print(f"[MANIFEST] {manifest_path}")

# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape AUPSC Communiqués by year (with pager detection & auto-stop)")
    ap.add_argument("--base", default=DEFAULT_BASE_LIST_URL, help="Listing base URL")
    ap.add_argument("--year", type=int, required=True, help="Target year, e.g., 2025")
    ap.add_argument("--start-page", type=int, default=1, help="First page to scan")
    ap.add_argument("--end-page",   type=int, default=50, help="Last page to scan (safety cap)")
    ap.add_argument("--pause", type=float, default=0.8, help="Seconds to sleep between downloads")
    ap.add_argument("--lang", choices=["en","fr","all"], default="en", help="Language filter for PDFs")
    ap.add_argument("--out", default="data/psc_year", help="Output directory")
    ap.add_argument("--no-strict-psc", action="store_true",
                    help="Disable strict PSC detection (by default, require PSC context)")
    ap.add_argument("--no-follow-article", action="store_true",
                    help="Do not open article pages to confirm PSC when needed")
    ap.add_argument("--auto-stop", action="store_true",
                    help="Stop automatically after N pages without hits post-year-start")
    ap.add_argument("--grace-pages", type=int, default=2,
                    help="Consecutive pages with no YEAR hits before auto-stop")
    ap.add_argument("--detect-last-page", action="store_true",
                    help="Auto-detect the real last page number from pagination and cap end_page")

    args = ap.parse_args()

    run(
        base_url=args.base,
        year=args.year,
        start_page=args.start_page,
        end_page=args.end_page,
        pause=args.pause,
        lang=args.lang,
        out_dir=args.out,
        strict_psc=(not args.no_strict_psc),
        follow_article=(not args.no_follow_article),
        auto_stop=args.auto_stop,
        grace_pages=args.grace_pages,
        detect_last=args.detect_last_page
    )
