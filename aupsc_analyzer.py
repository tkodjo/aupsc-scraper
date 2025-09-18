# au_analyze.py
# Build PSC charts:
#   1) Diplomatic Intent (DITS) distribution (pie)
#   2) Diplomatic Intent (DITS) monthly trend (line) — x-axis uses month abbreviations (Jan, Feb, …)
#   3) Most Frequent Discursive Verbs (bar) — bars colored by the verb’s DITS category
#
# Reads:  data/manifest.csv  (from aupsc_scraper.py)
# Writes: data/analysis.csv, output/charts/*.png
#
# Notes:
# - DITS = 4-level Diplomatic Intent/Tone Scale used here:
#     Assertive / Binding | Advisory / Urging | Supportive / Commendatory | Procedural / Neutral
# - We identify PSC communiqués via doc_type + text hit (“peace and security council” / “PSC”).
# - Y-axis on the intent trend is number of PSC communiqués per month for that DITS category.
# - Ties between DITS scores break by precedence order (see INTENT_ORDER).
# - Matplotlib uses a headless backend to avoid GUI issues.

import os, re, json, argparse, datetime, unicodedata

# Use non-GUI backend (safe on servers/CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- FS helpers --------------------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

# -------------------- Colors --------------------
def load_colors(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {"palette": [], "intent": {}, "verbs": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("palette", [])
        cfg.setdefault("intent", {})
        cfg.setdefault("verbs", {})
        return cfg
    except Exception as e:
        print(f"[WARN] failed to load colors ({e}); using defaults.")
        return {"palette": [], "intent": {}, "verbs": {}}

def color_seq(labels, cmap: dict | None, palette: list[str] | None):
    cmap = cmap or {}
    palette = palette or []
    out = []
    for i, lbl in enumerate(labels or []):
        c = cmap.get(lbl)
        if not c and palette:
            c = palette[i % len(palette)]
        out.append(c)
    return out if any(out) else None

# -------------------- PDF text extraction --------------------
def extract_text_pages(path: str) -> list[str]:
    """Return list of page texts. Prefer PyMuPDF; fallback to pdfminer (split on form-feed)."""
    # PyMuPDF
    try:
        import fitz  # type: ignore
        pages = []
        with fitz.open(path) as doc:
            for pg in doc:
                pages.append(pg.get_text())
        return pages if pages else [""]
    except Exception:
        pass
    # pdfminer
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        full = extract_text(path) or ""
        parts = [p.strip() for p in re.split(r"\f+", full) if p.strip()]
        return parts if parts else [full]
    except Exception:
        return [""]

# -------------------- Normalization & date parsing --------------------
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s:
        return None
    t = s
    # YYYY-MM-DD / YYYY/MM/DD
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])[-/](3[01]|[12]?\d)\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime.date(y, mo, d)
    # D Month YYYY
    m = re.search(r"\b(3[01]|[12]?\d)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
                  r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
                  r"nov(?:ember)?)\s+(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        d = int(m.group(1)); mon = m.group(2).lower(); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Month D, YYYY
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                  r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?)\s+"
                  r"(3[01]|[12]?\d),\s*(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        mon = m.group(1).lower(); d = int(m.group(2)); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Year only
    m = re.search(r"\b(20\d{2}|19\d{2})\b", t)
    if m:
        return datetime.date(int(m.group(1)), 1, 1)
    return None

# -------------------- Doc type & PSC flag --------------------
def guess_doc_type(title: str, filename: str, text: str):
    blob = _norm(" ".join([title or "", filename or "", text or ""]))
    if re.search(r"\bcommuniqu[eé]\b", blob) or re.search(r"\.comm[-_.](en|fr)\.pdf$", (filename or "").lower()):
        return "Communiqué"
    if re.search(r"\bpress (statement|release)\b", blob):
        return "Press Statement"
    if re.search(r"\breport\b", blob):
        return "Report"
    if re.search(r"\bdecision|declaration\b", blob):
        return "Decision/Declaration"
    if re.search(r"\bspeech|remarks|address\b", blob):
        return "Speech"
    return "Other"

def is_psc_communique(doc_type: str, title: str, text: str) -> bool:
    if (doc_type or "").lower() != "communiqué":
        return False
    blob = _norm(f"{title or ''} {text or ''}")
    return ("peace and security council" in blob) or (re.search(r"\bpsc\b", blob) is not None)

# -------------------- Sentiment (optional, light) --------------------
_SIA = None
def _get_sia():
    global _SIA
    if _SIA is None:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        except Exception:
            import nltk
            nltk.download("vader_lexicon")
        from nltk.sentiment import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
    return _SIA

def sentiment_scores(text: str):
    try:
        sia = _get_sia()
        t = text or ""
        head = t[:4000]; tail = t[-3000:] if len(t) > 7000 else ""
        s = sia.polarity_scores(head + "\n" + tail)
        comp = s.get("compound", 0.0)
        if comp >= 0.05: label = "Positive"
        elif comp <= -0.05: label = "Negative"
        else: label = "Neutral"
        return comp, label
    except Exception:
        return 0.0, "Neutral"

# -------------------- DITS (4-level) --------------------
INTENT_LEXICON = {
    "Assertive / Binding": [
        r"\bdecide(s|d|ing)?\b", r"\bauthori[sz]e(s|d|ing)?\b", r"\bcondemn(s|ed|ing)?\b",
        r"\bdirect(s|ed|ing)?\b", r"\bmandate(s|d|ing)?\b", r"\bimpos(e|es|ed|ing)\b.*\bsanction(s)?\b",
        r"\bsanction(s|ed|ing)?\b", r"\bsuspend(s|ed|ing)?\b", r"\breject(s|ed|ing)?\b",
        r"\bdemand(s|ed|ing)?\b", r"\border(s|ed|ing)?\b", r"\badopt(s|ed|ing)?\b",
        r"\bendorse(s|d|ing)?\b", r"\bapprove(s|d|ing)?\b",
    ],
    "Advisory / Urging": [
        r"\bcall(s|ed|ing)?\s+(upon|on|for)\b", r"\burge(s|d|ing)?\b", r"\bencourage(s|d|ing)?\b",
        r"\brequest(s|ed|ing)?\b", r"\bappeal(s|ed|ing)?\s+(to|for)\b", r"\binvite(s|d|ing)?\b",
        r"\brecommend(s|ed|ing)?\b", r"\bshould\b", r"\bpress(es|ed|ing)?\b",
    ],
    "Supportive / Commendatory": [
        r"\bwelcom(e|es|ed|ing)\b", r"\bcommend(s|ed|ing)?\b", r"\bappreciate(s|d|ing)?\b", r"\bsupport(s|ed|ing)?\b",
        r"\bapplaud(s|ed|ing)?\b", r"\bpraise(s|d|ing)?\b", r"\bcongratulate(s|d|ing)?\b",
        r"\brecognis(e|es|ed|ing)\b|\brecogniz(e|es|ed|ing)\b", r"\bnote(s|d)?\s+with\s+appreciation\b",
    ],
    "Procedural / Neutral": [
        r"\brecall(s|ed|ing)?\b", r"\breaffirm(s|ed|ing)?\b", r"\breiterate(s|d|ing)?\b",
        r"\bnote(s|d|ing)?\b", r"\btake(s|n)?\s+note\b", r"\bconsider(s|ed|ing)?\b", r"\bconsidering\b",
        r"\b(decide(s|d)?\s+to\s+)?remain(s|ed|ing)?\s+seized(\s+of\s+the\s+matter)?\b",
        r"\bpursuant\s+to\b", r"\bin\s+accordance\s+with\b", r"\bguided\s+by\b", r"\bmindful\s+of\b",
        r"\bunderline(s|d|ing)?\b", r"\bemphasise?s?\b", r"\baffirm(s|ed|ing)?\b",
    ],
}
INTENT_ORDER = [
    "Assertive / Binding",
    "Advisory / Urging",
    "Supportive / Commendatory",
    "Procedural / Neutral",
    "Other/General",
]
PROCEDURAL_OVERRIDE = re.compile(r"\b(decide(s|d)?\s+to\s+)?remain(s|ed|ing)?\s+seized(\s+of\s+the\s+matter)?\b", re.I)

def intent_from_text(text: str) -> str:
    """Classify one document to a single DITS label. Ties break by INTENT_ORDER precedence."""
    if not isinstance(text, str) or not text.strip():
        return "Other/General"
    t = text.lower()
    if PROCEDURAL_OVERRIDE.search(t):
        return "Procedural / Neutral"
    scores = {cat: 0 for cat in INTENT_LEXICON}
    for cat, pats in INTENT_LEXICON.items():
        scores[cat] = sum(len(re.findall(p, t)) for p in pats)
    if not any(scores.values()):
        return "Other/General"
    # tie-break by precedence order
    best = max(scores.items(), key=lambda kv: (kv[1], -INTENT_ORDER.index(kv[0])))
    return best[0]

# -------------------- Discursive verbs + DITS mapping --------------------
VERB_PATTERNS = {
    "adopt":      r"\badopt(s|ed|ing)?\b",
    "support":    r"\bsupport(s|ed|ing)?\b",
    "encourage":  r"\bencourag(e|es|ed|ing)\b",
    "request":    r"\brequest(s|ed|ing)?\b",
    "call":       r"\bcall(s|ed|ing)?\b",
    "recall":     r"\brecall(s|ed|ing)?\b",
    "note":       r"\bnote(s|d|ing)?\b",
    "reiterate":  r"\breiterate(s|d|ing)?\b",
    "commend":    r"\bcommend(s|ed|ing)?\b",
    "express":    r"\bexpres(s|ses|sed|sing)\b",
    "underline":  r"\bunderlin(e|es|ed|ing)\b",
    "warn":       r"\bwarn(s|ed|ing)?\b",
    "decide":     r"\bdecide(s|d|ing)?\b",
    "welcome":    r"\bwelcom(e|es|ed|ing)\b",
    "mandate":    r"\bmandate(s|d|ing)?\b",
    "reaffirm":   r"\breaffirm(s|ed|ing)?\b",
    "urge":       r"\burge(s|d|ing)?\b",
    "direct":     r"\bdirect(s|ed|ing)?\b",
    "deploy":     r"\bdeploy(s|ed|ing)?\b",
    "endorse":    r"\bendorse(s|d|ing)?\b",
}
VERB_TO_DITS = {
    # Assertive / Binding
    "adopt": "Assertive / Binding",
    "decide": "Assertive / Binding",
    "mandate": "Assertive / Binding",
    "direct": "Assertive / Binding",
    "deploy": "Assertive / Binding",
    "endorse": "Assertive / Binding",
    # Advisory / Urging
    "call": "Advisory / Urging",
    "urge": "Advisory / Urging",
    "encourage": "Advisory / Urging",
    "request": "Advisory / Urging",
    "warn": "Advisory / Urging",
    # Supportive / Commendatory
    "support": "Supportive / Commendatory",
    "welcome": "Supportive / Commendatory",
    "commend": "Supportive / Commendatory",
    "express": "Supportive / Commendatory",
    # Procedural / Neutral
    "recall": "Procedural / Neutral",
    "note": "Procedural / Neutral",
    "reiterate": "Procedural / Neutral",
    "reaffirm": "Procedural / Neutral",
    "underline": "Procedural / Neutral",
}
def count_verbs_in_text(text: str) -> dict[str, int]:
    if not isinstance(text, str) or not text:
        return {k: 0 for k in VERB_PATTERNS}
    t = text.lower()
    return {v: len(re.findall(pat, t)) for v, pat in VERB_PATTERNS.items()}

# -------------------- Plotters --------------------
from matplotlib.patches import Patch

def pie_save(series: pd.Series, title: str, out_path: str, colors_cfg: dict | None = None, cmap_key: str | None = None):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    labels = series.index.tolist()
    colors = color_seq(labels, (colors_cfg or {}).get(cmap_key or "", {}), (colors_cfg or {}).get("palette", []))
    plt.pie(series.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def line_save_monthabbr(df_wide: pd.DataFrame, title: str, out_path: str,
                        xlabel="Month", ylabel="Count", colors_cfg: dict | None = None, cmap_key: str | None = None):
    """Line plot with x labels = month abbreviations (%b). Works with PeriodIndex('M')."""
    ensure_dir(os.path.dirname(out_path))
    plt.figure()

    cols = list(df_wide.columns)
    colors = color_seq(cols, (colors_cfg or {}).get(cmap_key or "", {}), (colors_cfg or {}).get("palette", []))

    # Build month-abbr x labels
    try:
        ts = df_wide.index.to_timestamp()
    except Exception:
        ts = pd.to_datetime(df_wide.index)
    x_labels = [getattr(x, "strftime", lambda fmt=None, _x=x: str(_x))("%b") for x in ts]
    x_pos = list(range(len(x_labels)))

    for i, col in enumerate(cols):
        y = df_wide[col].values
        kw = {"marker": "o"}
        if colors and i < len(colors) and colors[i]:
            kw["color"] = colors[i]
        plt.plot(x_pos, y, label=col, **kw)

    plt.xticks(ticks=x_pos, labels=x_labels)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def bar_save_verbs_colored(series: pd.Series, title: str, out_path: str,
                           verb_to_dits: dict, colors_cfg: dict | None):
    """Bar chart of verbs, bars colored by the verb's DITS category using colors_cfg['intent']."""
    ensure_dir(os.path.dirname(out_path))
    plt.figure()

    labels = series.index.tolist()
    values = series.values.tolist()
    intent_colors = (colors_cfg or {}).get("intent", {})
    palette = (colors_cfg or {}).get("palette", [])

    bar_colors = []
    used_cats = []
    for i, v in enumerate(labels):
        cat = verb_to_dits.get(v, "Other/General")
        c = intent_colors.get(cat) or (palette[i % len(palette)] if palette else None)
        bar_colors.append(c)
        used_cats.append(cat)

    x = range(len(labels))
    plt.bar(x, values, color=bar_colors)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title); plt.xlabel("Verb"); plt.ylabel("Occurrences")

    # Legend (categories actually used), keep INTENT_ORDER order
    cats_unique = [c for c in INTENT_ORDER if c in used_cats and c != "Other/General"]
    if any(c == "Other/General" for c in used_cats):
        cats_unique.append("Other/General")
    handles = []
    for idx, cat in enumerate(cats_unique):
        col = intent_colors.get(cat) or (palette[idx % len(palette)] if palette else None)
        handles.append(Patch(facecolor=col, label=cat))
    if handles:
        plt.legend(handles=handles, title="DITS category", frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# -------------------- Main --------------------
def main(manifest="data/manifest.csv",
         analysis_csv="data/analysis.csv",
         outdir="output/charts",
         colors_file="aupsc_chartcolor.json",
         limit=0,
         top_verbs=20,
         store_text=False):
    colors_cfg = load_colors(colors_file)
    ensure_dir(outdir)

    if not os.path.exists(manifest):
        raise SystemExit(f"Manifest not found: {manifest}")

    man = pd.read_csv(manifest)
    if limit and limit > 0:
        man = man.head(limit)

    rows = []
    psc_rows = []
    verb_totals = {v: 0 for v in VERB_PATTERNS}

    for _, r in man.iterrows():
        path = r.get("path") or os.path.join("data", "pdfs", r.get("filename", ""))
        title = str(r.get("title", "")) if pd.notna(r.get("title", "")) else ""
        filename = str(r.get("filename", "")) if pd.notna(r.get("filename", "")) else ""
        url = str(r.get("url", "")) if pd.notna(r.get("url", "")) else ""

        # Extract text
        pages = extract_text_pages(path)
        full_text = "\n\n".join(pages).strip()

        # Meta
        doc_type = guess_doc_type(title, filename, full_text)
        psc_flag = is_psc_communique(doc_type, title, full_text)
        intent4 = intent_from_text(full_text)
        sent_comp, sent_label = sentiment_scores(full_text)

        # Date: try title -> filename -> text
        date_val = None
        for field in (title, filename, full_text):
            date_val = _parse_date_from_text(field)
            if date_val: break

        # Persist row (small)
        row = {
            "title": title,
            "filename": filename,
            "url": url,
            "path": str(path).replace("\\","/"),
            "doc_type": doc_type,
            "psc_flag": bool(psc_flag),
            "intent4": intent4,
            "sent_compound": sent_comp,
            "sent_label": sent_label,
            "date": date_val.isoformat() if date_val else "",
            "size_bytes": int(r.get("size_bytes", 0)) if pd.notna(r.get("size_bytes", 0)) else 0,
            "lang": r.get("lang", "") if "lang" in r else "",
        }
        if store_text:
            row["text"] = full_text
        rows.append(row)

        # PSC-only aggregations
        if psc_flag:
            psc_rows.append({"date": row["date"], "intent4": intent4})
            # Verb totals (PSC only)
            counts = count_verbs_in_text(full_text)
            for v, n in counts.items():
                verb_totals[v] += n

    # Save analysis CSV
    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(analysis_csv) or ".")
    df.to_csv(analysis_csv, index=False, encoding="utf-8")
    print(f"[DATA] Wrote {analysis_csv} ({len(df)} rows)")

    # ---- Build PSC charts ----
    df_psc = pd.DataFrame(psc_rows)
    if df_psc.empty:
        print("[INFO] No PSC communiqués found; skipping PSC charts.")
        return

    # (1) Pie — DITS distribution
    dits_order = ["Assertive / Binding","Advisory / Urging","Supportive / Commendatory","Procedural / Neutral","Other/General"]
    pie_series = df_psc["intent4"].value_counts().reindex(dits_order, fill_value=0)
    # drop trailing zeros so the legend is clean
    pie_series = pie_series[pie_series > 0]
    if not pie_series.empty:
        pie_save(pie_series, "PSC Communiqués: Diplomatic Intent (4 levels)",
                 os.path.join(outdir, "psc_pie_intent4.png"),
                 colors_cfg=colors_cfg, cmap_key="intent")

    # (2) Line — DITS monthly trend (x axis = month abbreviations)
    df_psc["date_dt"] = pd.to_datetime(df_psc["date"], errors="coerce")
    df_psc = df_psc.dropna(subset=["date_dt"])
    if not df_psc.empty:
        df_psc["month"] = df_psc["date_dt"].dt.to_period("M")
        trend = (df_psc.pivot_table(index="month", columns="intent4", values="date", aggfunc="count")
                           .fillna(0)
                           .reindex(columns=dits_order, fill_value=0))
        if not trend.empty:
            full_index = pd.period_range(trend.index.min(), trend.index.max(), freq="M")
            trend = trend.reindex(full_index, fill_value=0)
            line_save_monthabbr(trend,
                                "PSC Communiqués: Diplomatic Intent Trend (Monthly)",
                                os.path.join(outdir, "psc_line_intent_trend.png"),
                                xlabel="Month", colors_cfg=colors_cfg, cmap_key="intent")

    # (3) Bar — Most frequent discursive verbs (Top N), colored by DITS
    verb_s = pd.Series(verb_totals).sort_values(ascending=False).head(int(top_verbs))
    verb_s = verb_s[verb_s > 0]
    if not verb_s.empty:
        bar_save_verbs_colored(verb_s,
                               f"Most Frequent Discursive Verbs in PSC Communiqués (Top {int(top_verbs)})",
                               os.path.join(outdir, "psc_bar_discursive_verbs.png"),
                               VERB_TO_DITS, colors_cfg)

    print(f"[DONE] Charts saved to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze AU PSC communiqués and build DITS + verbs charts")
    ap.add_argument("--manifest", default="data/manifest.csv", help="Input manifest from scraper")
    ap.add_argument("--analysis-csv", default="data/analysis.csv", help="Output CSV path")
    ap.add_argument("--outdir", default="output/charts", help="Charts output directory")
    ap.add_argument("--colors-file", default="aupsc_chartcolor.json", help="Colors JSON (intent map + palette)")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N manifest rows")
    ap.add_argument("--top-verbs", type=int, default=20, help="How many verbs to show in the bar")
    ap.add_argument("--store-text", action="store_true", help="Include full text in analysis.csv (large)")
    args = ap.parse_args()
    main(args.manifest, args.analysis_csv, args.outdir, args.colors_file, args.limit, args.top_verbs, args.store_text)
