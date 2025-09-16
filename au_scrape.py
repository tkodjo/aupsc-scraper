# au_scrape.py
import os, re, time, csv, argparse
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_LIST_URL = "https://www.peaceau.org/en/resource/documents?idtype=7"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

def get(url, **kw):
    resp = requests.get(url, headers=HEADERS, timeout=30, **kw)
    resp.raise_for_status()
    return resp

def list_page_url(page:int) -> str:
    # Explicit &p= works for page 1 as well
    return f"{BASE_LIST_URL}&p={page}"

def find_pdf_links(html:str, lang="en"):
    """
    Extract direct PDF links from listing pages.
    If lang=='en', keep only *-en.pdf; if 'fr', only *-fr.pdf; 'all' keeps all PDFs.
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full = urljoin(BASE_LIST_URL, href)
            if lang == "en":
                if re.search(r"-en\.pdf$", full, re.IGNORECASE):
                    links.append((a.get_text(strip=True), full))
            elif lang == "fr":
                if re.search(r"-fr\.pdf$", full, re.IGNORECASE):
                    links.append((a.get_text(strip=True), full))
            else:
                links.append((a.get_text(strip=True), full))
    # de-dupe
    seen, unique = set(), []
    for title, url in links:
        if url not in seen:
            unique.append((title, url))
            seen.add(url)
    return unique

def safe_filename(name:str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "", name)
    return name[:180]

def download_pdf(url:str, out_dir:str) -> str:
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

def run(start_page:int, end_page:int, pause:float, lang:str):
    all_rows = []
    pdf_dir = os.path.join("data", "pdfs")
    os.makedirs("data", exist_ok=True)

    for p in range(start_page, end_page + 1):
        url = list_page_url(p)
        try:
            html = get(url).text
        except Exception as e:
            print(f"[WARN] Fetch page {p}: {e}")
            continue

        links = find_pdf_links(html, lang=lang)
        if not links:
            print(f"[INFO] No PDFs on page {p}.")
            continue

        print(f"[INFO] Page {p}: {len(links)} PDF links.")
        for title, pdf_url in tqdm(links, desc=f"Page {p}"):
            try:
                path = download_pdf(pdf_url, pdf_dir)
                size = os.path.getsize(path) if os.path.exists(path) else 0
                all_rows.append({
                    "page": p,
                    "title": title,
                    "url": pdf_url,
                    "filename": os.path.basename(path),
                    "path": path.replace("\\", "/"),
                    "size_bytes": size
                })
                time.sleep(pause)
            except Exception as e:
                print(f"[WARN] {pdf_url}: {e}")

    manifest_path = os.path.join("data", "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "title", "url", "filename", "path", "size_bytes"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"[DONE] Saved {len(all_rows)} entries to {manifest_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page",   type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.8)
    ap.add_argument("--lang", choices=["en","fr","all"], default="en")
    args = ap.parse_args()
    run(args.start_page, args.end_page, args.pause, args.lang)
