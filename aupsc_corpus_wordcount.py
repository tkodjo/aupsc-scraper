# au_corpus_wordcount.py
import os, re, argparse, pandas as pd

# ---------- PDF text extraction ----------
def extract_text_pages(path: str) -> list[str]:
    # Try PyMuPDF
    try:
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(path) as doc:
            for pg in doc:
                pages.append(pg.get_text())
        return pages if pages else [""]
    except Exception:
        pass
    # Fallback: pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        full = extract_text(path) or ""
        parts = [p for p in re.split(r"\f+", full) if p.strip()]
        return parts if parts else [full]
    except Exception:
        return [""]

def word_count(text: str) -> int:
    # Simple tokenization by word boundaries
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))

def main(manifest="data/manifest.csv",
         out_csv="data/wordcounts.csv",
         year=None,
         lang=None,               # "en" | "fr" | "all"/None
         psc_only=False,
         skip_first_page=False):
    if not os.path.exists(manifest):
        raise SystemExit(f"Manifest not found: {manifest}")

    dfm = pd.read_csv(manifest)
    if dfm.empty:
        raise SystemExit("Manifest is empty.")

    rows = []
    total_words = 0
    docs_counted = 0

    for _, r in dfm.iterrows():
        # Optional filters
        if year and str(r.get("year", "")).strip():
            try:
                if int(r["year"]) != int(year):
                    continue
            except Exception:
                pass
        if lang and lang.lower() in ("en", "fr"):
            # prefer explicit lang column, else infer from URL/filename suffix
            lang_col = str(r.get("lang", "")).lower()
            if lang_col not in ("en", "fr"):
                u = (str(r.get("url","")) or str(r.get("filename",""))).lower()
                if   re.search(r"-en(?:-[^/]+)?\.pdf$", u): lang_col = "en"
                elif re.search(r"-fr(?:-[^/]+)?\.pdf$", u): lang_col = "fr"
                else: lang_col = "unknown"
            if lang_col != lang.lower():
                continue

        # Get file path
        path = r.get("path") or os.path.join("data","pdfs", r.get("filename",""))
        if not path or not os.path.exists(path):
            continue

        # Extract text
        pages = extract_text_pages(path)
        if skip_first_page and len(pages) > 1:
            pages = pages[1:]
        txt = "\n\n".join(pages)

        # Optional PSC-only filter using a light heuristic
        if psc_only:
            title = str(r.get("title",""))
            fn = str(r.get("filename",""))
            blob = f"{title} {fn} {txt}".lower()
            is_comm = bool(re.search(r"\bcommuniqu[eé]|communique\b", blob) or re.search(r"\.comm[-_.](en|fr)\.pdf$", fn.lower()))
            is_psc = ("peace and security council" in blob) or (re.search(r"\bpsc\b", blob) is not None)
            if not (is_comm and is_psc):
                continue

        wc = word_count(txt)
        total_words += wc
        docs_counted += 1

        rows.append({
            "title": r.get("title",""),
            "filename": r.get("filename",""),
            "url": r.get("url",""),
            "path": str(path).replace("\\","/"),
            "year": r.get("year",""),
            "lang": r.get("lang",""),
            "words": wc
        })

    # Output
    if rows:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Documents counted: {docs_counted}")
    print(f"TOTAL WORDS: {total_words}")
    if rows:
        print(f"Per-document breakdown saved to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Count words across the AU PSC PDF corpus")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-csv", default="data/wordcounts.csv")
    ap.add_argument("--year", type=int, default=None, help="Filter a specific year")
    ap.add_argument("--lang", choices=["en","fr","all"], default=None, help="Language filter (by filename suffix or manifest lang)")
    ap.add_argument("--psc-only", action="store_true", help="Only count PSC communiqués")
    ap.add_argument("--skip-first-page", action="store_true", help="Skip page 1 of each PDF (venue/city page)")
    args = ap.parse_args()
    main(args.manifest, args.out_csv, args.year, None if args.lang=="all" else args.lang,
         args.psc_only, args.skip_first_page)
