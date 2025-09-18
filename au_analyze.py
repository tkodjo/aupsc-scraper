# analyze.py — AU/PSC document analysis pipeline
# - Reads data/manifest.csv (from au_scrape.py)
# - Extracts PDF text (per-page via PyMuPDF, pdfminer fallback)
# - Heuristic doc_type
# - Multi-country extraction (ignores PDF page 1 by default) + region mapping
# - Topic tagging (country preferred; else thematic/organs/instruments/RECs)
# - Sentiment (VADER)
# - 4-level Diplomatic Intent
# - Saves data/analysis.csv + charts in output/charts/
# - Optional colors via chart_colors.json

import os, re, argparse, json, unicodedata, datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- FS helpers ----------------
def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

# ---------------- Colors ----------------
def load_colors(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in ("tone","intent","verbs","regions","doc_types","topics","crises"):
            if k in cfg and not isinstance(cfg[k], dict):
                cfg[k] = {}
        if "palette" not in cfg or not isinstance(cfg["palette"], list):
            cfg["palette"] = []
        return cfg
    except Exception as e:
        print(f"[WARN] Colors not loaded ({e}).")
        return {"palette": []}

def _color_list(labels, cmap, palette):
    if not labels:
        return None
    out = []
    for i, lbl in enumerate(labels):
        c = (cmap or {}).get(lbl)
        if not c and palette:
            c = palette[i % len(palette)]
        out.append(c)
    return out if any(out) else None

# ---------------- PDF text extraction ----------------
def extract_text_pages(path):
    """Return list of page texts; prefer PyMuPDF; fallback to pdfminer (split by form feed)."""
    # Primary: PyMuPDF
    try:
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pages.append(page.get_text())
        return pages if pages else [""]
    except Exception:
        pass
    # Fallback: pdfminer.six
    try:
        from pdfminer.high_level import extract_text
        full = extract_text(path) or ""
        parts = [p.strip() for p in re.split(r"\f+", full) if p.strip()]
        return parts if parts else [full]
    except Exception:
        return [""]

# ---------------- Normalization ----------------
def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

# ---------------- Date parsing ----------------
_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s:
        return None
    t = s
    # YYYY-MM-DD or YYYY/MM/DD
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

def derive_date(title, filename, text):
    for field in (title, filename, text):
        d = _parse_date_from_text(str(field or ""))
        if d:
            return d
    return None

# ---------------- Country & Region mapping ----------------
COUNTRIES = [
    # North Africa (6 + SADR)
    "Algeria","Egypt","Libya","Mauritania","Morocco","Tunisia",
    "Sahrawi Arab Democratic Republic (SADR)",
    # West Africa (15)
    "Benin","Burkina Faso","Cabo Verde","Côte d’Ivoire","Gambia","Ghana","Guinea",
    "Guinea-Bissau","Liberia","Mali","Niger","Nigeria","Senegal","Sierra Leone","Togo",
    # Central Africa (9)
    "Burundi","Cameroon","Central African Republic","Chad","Congo (Republic)",
    "Democratic Republic of Congo (DRC)","Equatorial Guinea","Gabon","São Tomé and Príncipe",
    # East Africa (14)
    "Comoros","Djibouti","Eritrea","Ethiopia","Kenya","Madagascar","Mauritius","Rwanda",
    "Seychelles","Somalia","South Sudan","Sudan","Tanzania","Uganda",
    # Southern Africa (11)
    "Angola","Botswana","Eswatini","Lesotho","Malawi","Mozambique","Namibia",
    "South Africa","Zambia","Zimbabwe",
]
REGION_OF = {
    # North
    "Algeria":"North Africa","Egypt":"North Africa","Libya":"North Africa","Mauritania":"North Africa",
    "Morocco":"North Africa","Tunisia":"North Africa","Sahrawi Arab Democratic Republic (SADR)":"North Africa",
    # West
    "Benin":"West Africa","Burkina Faso":"West Africa","Cabo Verde":"West Africa","Côte d’Ivoire":"West Africa",
    "Gambia":"West Africa","Ghana":"West Africa","Guinea":"West Africa","Guinea-Bissau":"West Africa",
    "Liberia":"West Africa","Mali":"West Africa","Niger":"West Africa","Nigeria":"West Africa",
    "Senegal":"West Africa","Sierra Leone":"West Africa","Togo":"West Africa",
    # Central
    "Burundi":"Central Africa","Cameroon":"Central Africa","Central African Republic":"Central Africa",
    "Chad":"Central Africa","Congo (Republic)":"Central Africa","Democratic Republic of Congo (DRC)":"Central Africa",
    "Equatorial Guinea":"Central Africa","Gabon":"Central Africa","São Tomé and Príncipe":"Central Africa",
    # East
    "Comoros":"East Africa","Djibouti":"East Africa","Eritrea":"East Africa","Ethiopia":"East Africa",
    "Kenya":"East Africa","Madagascar":"East Africa","Mauritius":"East Africa","Rwanda":"East Africa",
    "Seychelles":"East Africa","Somalia":"East Africa","South Sudan":"East Africa","Sudan":"East Africa",
    "Tanzania":"East Africa","Uganda":"East Africa",
    # Southern
    "Angola":"Southern Africa","Botswana":"Southern Africa","Eswatini":"Southern Africa","Lesotho":"Southern Africa",
    "Malawi":"Southern Africa","Mozambique":"Southern Africa","Namibia":"Southern Africa",
    "South Africa":"Southern Africa","Zambia":"Southern Africa","Zimbabwe":"Southern Africa",
}
ALIASES = {
    "Côte d’Ivoire": ["cote d’ivoire","cote d'ivoire","ivory coast"],
    "Democratic Republic of Congo (DRC)": [  # English variants
        "democratic republic of congo",
        "democratic republic of the congo",
        "democratic republic of congo (drc)",
        "democratic republic of the congo (drc)",
        "dr congo",
        "dr of congo",
        "congo-kinshasa",
        "drc",
        # French variants
        "republique democratique du congo",
        "république démocratique du congo",
        "rep. dem. du congo",
        "rdc"],
    "Congo (Republic)": ["republic of the congo","congo-brazzaville"],
    "Eswatini": ["swaziland"],
    "São Tomé and Príncipe": ["sao tome and principe","sao tome","são tomé","sao tomé","sao tome & principe"],
    "Sahrawi Arab Democratic Republic (SADR)": ["sadr","sahrawi","western sahara"],
    "Gambia": ["the gambia"],
}

def _alias_to_regex(alias: str) -> re.Pattern:
    a = _norm(alias)
    a = re.sub(r"\s+", r"\\s+", re.escape(a))   # collapse whitespace
    a = a.replace("\\-", "[-\\s]?")             # hyphen or space
    return re.compile(rf"(?<!\w){a}(?!\w)", re.IGNORECASE)

COUNTRY_PATTERNS = {}
for c in COUNTRIES:
    pats = [_alias_to_regex(c)]
    for alt in ALIASES.get(c, []):
        pats.append(_alias_to_regex(alt))
    COUNTRY_PATTERNS[c] = pats

def extract_countries_from_text(title: str, text: str):
    blob = _norm(f"{title or ''} {text or ''}")
    hits = set()
    for country, pats in COUNTRY_PATTERNS.items():
        if any(p.search(blob) for p in pats):
            hits.add(country)
    return sorted(hits)

def region_from_countries(countries):
    if not countries:
        return "Non-country/Other"
    regs = {REGION_OF.get(c) for c in countries if REGION_OF.get(c)}
    if not regs:
        return "Non-country/Other"
    return regs.pop() if len(regs) == 1 else "Multi-Region"

# ---------------- Topic dictionaries ----------------
THEMATIC = {
    "African Standby Force (ASF)": [r"\bafrican standby force\b", r"\basf\b"],
    "APSA / Peace Architecture": [r"\bapsa\b", r"\bpeace and security architecture\b"],
    "Disarmament / DDR / SSR": [r"\bddr\b", r"\bdisarmament\b", r"\bsecurity sector reform\b", r"\bssr\b"],
    "Early Warning / CEWS": [r"\bearly warning\b", r"\bcews\b"],
    "Women/Children in Armed Conflict": [r"\bwomen\b.*\barme?d conflict\b", r"\bchildren in armed conflict\b"],
    "Youth, Peace & Security (YPS)": [r"\byouth(,|\s) peace(,|\s) and security\b", r"\byps\b"],
    "Humanitarian / Disaster Response": [r"\bhumanitarian\b", r"\bdisaster response\b"],
    "Maritime Security": [r"\bmaritime security\b", r"\bpiracy\b"],
    "Cybersecurity / Digital": [r"\bcyber(security| ?crime| ?attack)s?\b", r"\bdigital\b"],
    "Climate & Security": [r"\bclimate (change|security)\b", r"\benvironmental security\b"],
    "Health Security": [r"\bpandemic(s)?\b", r"\bepidemic(s)?\b", r"\bbiosecurity\b"],
    "Terrorism / TOC": [r"\bterroris(m|t)s?\b", r"\btransnational organized crime\b", r"\bviolent extremis(m|t)s?\b"],
    "PCRD": [r"\bpost-?conflict reconstruction\b", r"\bpcrd\b"],
    "PoC / Protection of Civilians": [r"\bprotection of civilians\b", r"\bpoc\b"],
    "UCG / Coups": [r"\bunconstitutional changes of government\b", r"\bucg\b", r"\bcoup(s)?\b"],
}
ORGANS = {
    "Peace and Security Council (PSC)": [r"\bpeace and security council\b", r"\bpsc\b"],
    "AU Commission (AUC)": [r"\bau commission\b", r"\bauc\b"],
    "Panel of the Wise": [r"\bpanel of the wise\b"],
    "ACSRT (Algiers)": [r"\bacsrt\b", r"\bafrican centre for the study and research on terrorism\b"],
    "CEWS": [r"\bcews\b", r"\bcontinental early warning system\b"],
}
INSTRUMENTS = {
    "PSC Protocol": [r"\bprotocol relating to the establishment of the peace and security council\b", r"\bpsc protocol\b"],
    "ACDEG (2007)": [r"\bcharter on democracy, elections and governance\b", r"\bacdeg\b"],
    "Lomé Declaration (2000)": [r"\blom[eé] declaration\b"],
    "Pelindaba Treaty": [r"\bafrican nuclear-weapon-free zone\b", r"\bpelindaba\b"],
    "Kampala Convention (2009)": [r"\bkampala convention\b"],
}
RECS = {
    "ECOWAS": [r"\becowas\b"],
    "SADC": [r"\bsadc\b"],
    "IGAD": [r"\bigad\b"],
    "ECCAS": [r"\beccas\b"],
    "COMESA": [r"\bcomesa\b"],
    "CEN-SAD": [r"\bcen-?sad\b"],
    "UMA / AMU": [r"\b(uma|amu|arab maghreb union)\b"],
}

def _match_any(dct, blob):
    hits = []
    for label, patterns in dct.items():
        for p in patterns:
            if re.search(p, blob):
                hits.append(label); break
    return hits

def detect_topics(title: str, text: str, countries):
    # Prefer country topics when present
    blob = _norm(f"{title or ''} {text or ''}")
    if countries:
        if len(countries) == 1:
            return countries[0], "Country"
        return "Multi-country", "Country"
    # Else fall back to other taxonomies
    for dct, src in ((THEMATIC,"Thematic"), (ORGANS,"Organ"), (INSTRUMENTS,"Instrument"), (RECS,"REC")):
        hits = _match_any(dct, blob)
        if hits:
            return hits[0], src
    return "Other/General", "Other"

# ---------------- Doc type heuristics ----------------
def guess_doc_type(title: str, filename: str, text: str):
    blob = _norm(" ".join([title or "", filename or "", text or ""]))
    if re.search(r"\bcommuniqu[eé]\b", blob) or re.search(r"\b\.comm[-_.](en|fr)\.pdf$", (filename or "").lower()):
        return "Communiqué"
    if re.search(r"\bpress (statement|release)\b", blob):
        return "Press Statement"
    if re.search(r"\breport\b", blob):
        return "Report"
    if re.search(r"\bspeech|remarks|address\b", blob):
        return "Speech"
    if re.search(r"\bdecision|declaration\b", blob):
        return "Decision/Declaration"
    return "Other"

# ---------------- Sentiment (VADER) ----------------
_SIA = None
def _get_sia():
    global _SIA
    if _SIA is None:
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # noqa
        except Exception:
            import nltk
            nltk.download("vader_lexicon")
        from nltk.sentiment import SentimentIntensityAnalyzer
        _SIA = SentimentIntensityAnalyzer()
    return _SIA

def sentiment_scores(text: str):
    sia = _get_sia()
    t = text or ""
    head = t[:4000]; tail = t[-3000:] if len(t) > 7000 else ""
    s = sia.polarity_scores(head + "\n" + tail)
    comp = s.get("compound", 0.0)
    if comp >= 0.05: label = "Positive"
    elif comp <= -0.05: label = "Negative"
    else: label = "Neutral"
    return comp, label

# ---------------- 4-level Diplomatic Intent ----------------
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
    "Assertive / Binding","Advisory / Urging","Supportive / Commendatory","Procedural / Neutral","Other/General"
]
PROCEDURAL_OVERRIDE = re.compile(
    r"\b(decide(s|d)?\s+to\s+)?remain(s|ed|ing)?\s+seized(\s+of\s+the\s+matter)?\b", re.IGNORECASE
)

def intent_from_text(text: str) -> str:
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
    best = max(scores.items(), key=lambda kv: (kv[1], -INTENT_ORDER.index(kv[0])))
    return best[0]

# ---------------- PSC flag ----------------
def is_psc_communique(doc_type: str, title: str, text: str):
    if (doc_type or "").lower() != "communiqué":
        return False
    blob = _norm(f"{title or ''} {text or ''}")
    return ("peace and security council" in blob) or (re.search(r"\bpsc\b", blob) is not None)

# ---------------- Chart helpers ----------------
def bar_save(series, title, out_path, xlabel="", ylabel="Count", colors_cfg=None, cmap_key=None):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    colors = _color_list(series.index.tolist(), (colors_cfg or {}).get(cmap_key or "", {}), (colors_cfg or {}).get("palette", []))
    series.plot(kind="bar", color=colors)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def stacked_save(df_wide, title, out_path, xlabel="", ylabel="Count", colors_cfg=None, cmap_key=None):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    cols = df_wide.columns.tolist()
    colors = _color_list(cols, (colors_cfg or {}).get(cmap_key or "", {}), (colors_cfg or {}).get("palette", []))
    df_wide.plot(kind="bar", stacked=True, color=colors)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def hist_save(series, title, out_path, bins=30, xlabel="Score"):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ---------------- Main ----------------
def main(manifest="data/manifest.csv", analysis_csv="data/analysis.csv", outdir="output/charts",
         limit=0, colors_file=None, count_first_page=False):
    colors_cfg = load_colors(colors_file)

    if not os.path.exists(manifest):
        raise SystemExit(f"Manifest not found: {manifest}")
    man = pd.read_csv(manifest)
    if limit and limit > 0:
        man = man.head(limit)

    rows = []
    for _, r in man.iterrows():
        path = r.get("path") or os.path.join("data","pdfs", r.get("filename",""))
        title = str(r.get("title",""))
        filename = str(r.get("filename",""))
        url = str(r.get("url","") or r.get("pdf_url",""))

        pages = extract_text_pages(path)
        full_text = "\n\n".join(pages).strip()
        text_no_p1 = "\n\n".join(pages[1:]).strip() if len(pages) > 1 else ""
        country_scope_text = full_text if count_first_page else text_no_p1

        doc_type = guess_doc_type(title, filename, full_text)
        countries = extract_countries_from_text(title, country_scope_text)
        region = region_from_countries(countries)
        topic, topic_src = detect_topics(title, full_text, countries)
        sent_comp, sent_label = sentiment_scores(full_text)
        intent4 = intent_from_text(full_text)
        date_val = derive_date(title, filename, full_text)
        psc_flag = is_psc_communique(doc_type, title, full_text)

        rows.append({
            "title": title,
            "filename": filename,
            "url": url,
            "path": str(path).replace("\\","/"),
            "doc_type": doc_type,
            "text": full_text,
            "topic": topic,
            "topic_source": topic_src,
            "countries_str": "; ".join(countries),
            "region": region,
            "sent_compound": sent_comp,
            "sent_label": sent_label,
            "intent4": intent4,
            "date": date_val.isoformat() if date_val else "",
            "psc_flag": bool(psc_flag),
            "size_bytes": int(r.get("size_bytes", 0)),
            "pages_count": len(pages),
            "countries_count_scope": "full" if count_first_page else "no_first_page"
        })

    # Save analysis CSV
    ensure_dir(os.path.dirname(analysis_csv) or ".")
    df = pd.DataFrame(rows)
    df.to_csv(analysis_csv, index=False, encoding="utf-8")
    print(f"[DATA] Wrote {analysis_csv} ({len(df)} rows)")

    # ---- Charts ----
    ensure_dir(outdir)

    # 1) Doc types
    if "doc_type" in df.columns:
        c = df["doc_type"].value_counts()
        bar_save(c, "Documents by Type", os.path.join(outdir, "doc_types.png"),
                 colors_cfg=colors_cfg, cmap_key="doc_types")

    # 2) Sentiment labels
    if "sent_label" in df.columns:
        c = df["sent_label"].value_counts()
        bar_save(c, "Sentiment Labels", os.path.join(outdir, "sentiment_labels.png"),
                 colors_cfg=colors_cfg, cmap_key="tone")

    # 3) Sentiment histogram
    if "sent_compound" in df.columns:
        hist_save(df["sent_compound"], "Sentiment (VADER compound)",
                  os.path.join(outdir, "sentiment_hist.png"))

    # 4) Topics (Top 20)
    if "topic" in df.columns:
        top = df["topic"].value_counts().head(20)
        bar_save(top, "Top 20 Topics", os.path.join(outdir, "topics_top20.png"),
                 colors_cfg=colors_cfg, cmap_key="topics")

    # 5) Region counts
    if "region" in df.columns:
        reg = df["region"].value_counts()
        bar_save(reg, "Documents by Region", os.path.join(outdir, "regions.png"),
                 colors_cfg=colors_cfg, cmap_key="regions")

    # 6) Region × Sentiment (stacked)
    if {"region","sent_label"}.issubset(df.columns):
        pivot = (df.pivot_table(index="region", columns="sent_label", values="title", aggfunc="count")
                   .fillna(0)
                   .loc[lambda x: x.sum(axis=1) > 0])
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        stacked_save(pivot, "Documents by Region × Sentiment",
                     os.path.join(outdir, "regions_sentiment_stacked.png"),
                     xlabel="Region", colors_cfg=colors_cfg, cmap_key="tone")

    # 7) PSC-only quick counts (intent 4-level)
    df_psc = df[df.get("psc_flag", False) == True]
    if not df_psc.empty:
        order = ["Assertive / Binding","Advisory / Urging","Supportive / Commendatory","Procedural / Neutral","Other/General"]
        c = df_psc["intent4"].value_counts().reindex(order, fill_value=0)
        bar_save(c, "PSC Communiqués — Diplomatic Intent (4 levels)",
                 os.path.join(outdir, "psc_intent4_counts.png"),
                 colors_cfg=colors_cfg, cmap_key="intent")

    print(f"[DONE] Charts saved to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze AU/PSC PDFs -> CSV + charts")
    ap.add_argument("--manifest", default="data/manifest.csv", help="Input manifest (CSV) from au_scrape.py")
    ap.add_argument("--analysis-csv", default="data/analysis.csv", help="Output analysis CSV path")
    ap.add_argument("--outdir", default="output/charts", help="Charts output directory")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N manifest rows (testing)")
    ap.add_argument("--colors-file", default="chart_colors.json", help="Optional colors config")
    ap.add_argument("--count-first-page", action="store_true",
                    help="Include page 1 when detecting countries (default: exclude)")
    args = ap.parse_args()
    main(args.manifest, args.analysis_csv, args.outdir, args.limit, args.colors_file, args.count_first_page)
