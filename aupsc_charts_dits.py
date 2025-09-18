# au_psc_charts.py — PSC charts using a 4-level diplomatic intent model + color themes
import os, re, argparse, collections, datetime, json
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- I/O + color helpers ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_colors(path):
    if not path: return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # normalize expected shapes
        if "tone" not in cfg: cfg["tone"] = {}
        if "intent" not in cfg: cfg["intent"] = {}
        if "verbs" not in cfg: cfg["verbs"] = {}
        if "palette" not in cfg or not isinstance(cfg["palette"], list): cfg["palette"] = []
        if "verbs_palette_strong" not in cfg or not isinstance(cfg["verbs_palette_strong"], list):
            cfg["verbs_palette_strong"] = cfg["palette"]
        return cfg
    except Exception as e:
        print(f"[WARN] Could not load colors file {path}: {e}")
        return {"tone":{}, "intent":{}, "verbs":{}, "palette":[], "verbs_palette_strong":[]}

def map_colors(labels, color_map=None, palette=None):
    """Return a color list for labels using color_map or cycling palette."""
    if not labels: return None
    out = []
    for i, lbl in enumerate(labels):
        c = (color_map or {}).get(lbl)
        if not c and palette:
            c = palette[i % len(palette)] if palette else None
        out.append(c)
    return out if any(out) else None

def save_pie(series, title, out_path, color_map=None, palette=None):
    s = series[series > 0]
    if s.empty:
        print(f"[WARN] Nothing to plot for {title}")
        return
    labels = s.index.tolist()
    colors = map_colors(labels, color_map, palette)
    plt.figure()
    plt.pie(s.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_line(df_wide, title, xlabel, ylabel, out_path, color_map=None, palette=None):
    if df_wide.empty:
        print(f"[WARN] Nothing to plot for {title}"); return
    # x-axis
    x = df_wide.index
    if hasattr(x, "to_timestamp"): x = x.to_timestamp()
    plt.figure()
    cols = list(df_wide.columns)
    for i, col in enumerate(cols):
        color = (color_map or {}).get(col)
        if not color and palette:
            color = palette[i % len(palette)]
        plt.plot(x, df_wide[col], marker="o", label=col, color=color)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_bar(series_or_df, title, xlabel, ylabel, out_path, color_map=None, palette=None):
    plt.figure()
    if isinstance(series_or_df, pd.Series):
        labels = series_or_df.index.tolist()
        colors = map_colors(labels, color_map, palette)
        series_or_df.plot(kind="bar", color=colors)
    else:
        # DataFrame bars colored by columns
        cols = series_or_df.columns.tolist()
        colors = map_colors(cols, color_map, palette)
        series_or_df.plot(kind="bar", color=colors)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# ---------------- Date parsing ----------------
_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s: return None
    t = s
    # YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])[-/](3[01]|[12]?\d)\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime.date(y, mo, d)
    # D Month YYYY
    m = re.search(r"\b(3[01]|[12]?\d)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
                  r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
                  r"nov(?:ember)?|dec(?:ember)?)\s+(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        d = int(m.group(1)); mon = m.group(2).lower(); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Month D, YYYY
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                  r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
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

def derive_date(row):
    for field in ("title", "filename", "text"):
        d = _parse_date_from_text(str(row.get(field, "")))
        if d: return d
    return None

# ---------------- 4-level Diplomatic Intent ----------------
INTENT_LEXICON = {
    # 1) Assertive / Binding — strongest, claims authority & demands compliance
    "Assertive / Binding": [
        r"\bdecide(s|d|ing)?\b",
        r"\bauthori[sz]e(s|d|ing)?\b",
        r"\bcondemn(s|ed|ing)?\b",
        r"\bdirect(s|ed|ing)?\b",
        r"\bmandate(s|d|ing)?\b",
        r"\bimpos(e|es|ed|ing)\b.*\bsanction(s)?\b",
        r"\bsanction(s|ed|ing)?\b",
        r"\bsuspend(s|ed|ing)?\b",
        r"\breject(s|ed|ing)?\b",
        r"\bdemand(s|ed|ing)?\b",
        r"\border(s|ed|ing)?\b",
        r"\badopt(s|ed|ing)?\b",
        r"\bendorse(s|d|ing)?\b",
        r"\bapprove(s|d|ing)?\b",
    ],
    # 2) Advisory / Urging — normative pressure, not direct command
    "Advisory / Urging": [
        r"\bcall(s|ed|ing)?\s+(upon|on|for)\b",
        r"\burge(s|d|ing)?\b",
        r"\bencourage(s|d|ing)?\b",
        r"\brequest(s|ed|ing)?\b",
        r"\bappeal(s|ed|ing)?\s+(to|for)\b",
        r"\binvite(s|d|ing)?\b",
        r"\brecommend(s|ed|ing)?\b",
        r"\bshould\b",
        r"\bpress(es|ed|ing)?\b",
    ],
    # 3) Supportive / Commendatory — positive reinforcement
    "Supportive / Commendatory": [
        r"\bwelcom(e|es|ed|ing)\b",
        r"\bcommend(s|ed|ing)?\b",
        r"\bappreciate(s|d|ing)?\b",
        r"\bsupport(s|ed|ing)?\b",
        r"\bapplaud(s|ed|ing)?\b",
        r"\bpraise(s|d|ing)?\b",
        r"\bcongratulate(s|d|ing)?\b",
        r"\brecognis(e|es|ed|ing)\b|\brecogniz(e|es|ed|ing)\b",
        r"\bnote(s|d)?\s+with\s+appreciation\b",
    ],
    # 4) Procedural / Neutral — institutional memory & legal precision
    "Procedural / Neutral": [
        r"\brecall(s|ed|ing)?\b",
        r"\breaffirm(s|ed|ing)?\b",
        r"\breiterate(s|d|ing)?\b",
        r"\bnote(s|d|ing)?\b",
        r"\btake(s|n)?\s+note\b",
        r"\bconsider(s|ed|ing)?\b",
        r"\bconsidering\b",
        r"\b(decide(s|d)?\s+to\s+)?remain(s|ed|ing)?\s+seized(\s+of\s+the\s+matter)?\b",
        r"\bpursuant\s+to\b",
        r"\bin\s+accordance\s+with\b",
        r"\bguided\s+by\b",
        r"\bmindful\s+of\b",
        r"\bunderline(s|d|ing)?\b",
        r"\bemphasise?s?\b",
        r"\baffirm(s|ed|ing)?\b",
    ],
}
INTENT_ORDER = [
    "Assertive / Binding",
    "Advisory / Urging",
    "Supportive / Commendatory",
    "Procedural / Neutral",
    "Other/General",
]
PROCEDURAL_OVERRIDE = re.compile(
    r"\b(decide(s|d)?\s+to\s+)?remain(s|ed|ing)?\s+seized(\s+of\s+the\s+matter)?\b",
    re.IGNORECASE,
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
    # max by score; break ties by order
    best = max(scores.items(), key=lambda kv: (kv[1], -INTENT_ORDER.index(kv[0])))
    return best[0]

# ---------------- Discursive verbs ----------------
DISCursive_VERBS = {
    "condemn": r"\bcondemn(s|ed|ing)?\b",
    "denounce": r"\bdenounc(e|es|ed|ing)\b",
    "deplore": r"\bdeplor(e|es|ed|ing)\b",
    "sanction": r"\bsanction(s|ed|ing)?\b",
    "suspend": r"\bsuspend(s|ed|ing)?\b",
    "reject": r"\breject(s|ed|ing)?\b",
    "warn": r"\bwarn(s|ed|ing)?\b",
    "demand": r"\bdemand(s|ed|ing)?\b",

    "call": r"\bcall(s|ed|ing)?\b",
    "urge": r"\burge(s|d|ing)?\b",
    "request": r"\brequest(s|ed|ing)?\b",
    "appeal": r"\bappeal(s|ed|ing)?\b",
    "invite": r"\binvite(s|d|ing)?\b",
    "encourage": r"\bencourage(s|d|ing)?\b",
    "press": r"\bpress(es|ed|ing)?\b",

    "decide": r"\bdecide(s|d|ing)?\b",
    "authorize": r"\bauthori[sz]e(s|d|ing)?\b",
    "adopt": r"\badopt(s|ed|ing)?\b",
    "approve": r"\bapprove(s|d|ing)?\b",
    "endorse": r"\bendorse(s|d|ing)?\b",
    "direct": r"\bdirect(s|ed|ing)?\b",
    "mandate": r"\bmandate(s|d|ing)?\b",

    "deploy": r"\bdeploy(s|ed|ing)?\b",
    "support": r"\bsupport(s|ed|ing)?\b",
    "assist": r"\bassist(s|ed|ing)?\b",

    "welcome": r"\bwelcom(e|es|ed|ing)\b",
    "commend": r"\bcommend(s|ed|ing)?\b",
    "express": r"\bexpress(es|ed|ing)?\b",
    "reiterate": r"\breiterate(s|d|ing)?\b",
    "reaffirm": r"\breaffirm(s|ed|ing)?\b",
    "recall": r"\brecall(s|ed|ing)?\b",
    "emphasize": r"\bemphasise?s?\b",
    "underline": r"\bunderline(s|d|ing)?\b",
    "affirm": r"\baffirm(s|ed|ing)?\b",
    "congratulate": r"\bcongratulate(s|d|ing)?\b",
    "recognize": r"\brecognis(e|es|ed|ing)\b|\brecogniz(e|es|ed|ing)\b",
    "praise": r"\bpraise(s|d|ing)?\b",
    "applaud": r"\bapplaud(s|ed|ing)?\b",
    "note": r"\bnote(s|d|ing)?\b",
    "take_note": r"\btake(s|n)?\s+note\b",
}

def count_discursive_verbs(texts):
    counts = collections.Counter()
    for t in texts:
        if not isinstance(t, str): continue
        s = t.lower()
        for lemma, pat in DISCursive_VERBS.items():
            hits = len(re.findall(pat, s))
            if hits: counts[lemma] += hits
    return counts

# ---------------- PSC filter ----------------
def is_psc(row) -> bool:
    title = str(row.get("title", "")).lower()
    topic = str(row.get("topic", "")).lower()
    text = str(row.get("text", "")).lower()
    is_comm = str(row.get("doc_type", "")).lower() == "communiqué"
    psc_hit = ("peace and security council" in title or "psc" in title or
               "peace and security council" in topic or "psc" == topic or
               " peace and security council" in text or " psc " in text)
    return is_comm and psc_hit

# ---------------- Main ----------------
def main(manifest="data/analysis.csv", outdir="output/charts",
         topn_verbs=20, colors_file="chart_colors_dits.json"):
    ensure_dir(outdir)
    colors = load_colors(colors_file)
    df = pd.read_csv(manifest)

    needed = {"doc_type","title","topic","text","sent_label","filename"}
    if not needed.issubset(df.columns):
        raise SystemExit("analysis.csv must include: doc_type,title,topic,text,sent_label,filename")

    # filter PSC communiqués
    df_psc = df[df.apply(is_psc, axis=1)].copy()
    if df_psc.empty:
        raise SystemExit("No PSC communiqués found.")

    # dates
    df_psc["date"] = df_psc.apply(derive_date, axis=1)
    df_psc["month"] = pd.to_datetime(df_psc["date"], errors="coerce").dt.to_period("M")
    
     # 4-level intent
    df_psc["intent4"] = df_psc["text"].apply(intent_from_text)
    
    df_trend = df_psc.dropna(subset=["month"]).copy()

   

    # --- PIE: intent + tone
    intent_counts = df_psc["intent4"].value_counts().reindex(INTENT_ORDER, fill_value=0)
    save_pie(intent_counts,
             "PSC Communiqués: Diplomatic Intent (4 levels)",
             os.path.join(outdir, "psc_pie_intent4.png"),
             color_map=colors.get("intent"),
             palette=colors.get("palette"))

    tone_counts = df_psc["sent_label"].value_counts()
    save_pie(tone_counts,
             "PSC Communiqués: Tone (Sentiment)",
             os.path.join(outdir, "psc_pie_tone.png"),
             color_map=colors.get("tone"),
             palette=colors.get("palette"))

    # --- LINE: trends (monthly)
    tone_trend = (df_trend.groupby(["month", "sent_label"]).size()
                  .unstack(fill_value=0).sort_index())
    save_line(tone_trend,
              "PSC Communiqués: Tone Trend (Monthly)",
              "Month", "Count",
              os.path.join(outdir, "psc_line_tone_trend.png"),
              color_map=colors.get("tone"),
              palette=colors.get("palette"))

    intent_trend = (df_trend.groupby(["month", "intent4"]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=[c for c in INTENT_ORDER if c in df_trend["intent4"].unique()], fill_value=0)
                    .sort_index())
    save_line(intent_trend,
              "PSC Communiqués: Diplomatic Intent Trend (Monthly)",
              "Month", "Count",
              os.path.join(outdir, "psc_line_intent_trend.png"),
              color_map=colors.get("intent"),
              palette=colors.get("palette"))

    # --- BAR: top discursive verbs (use stronger palette fallback)
    verb_counts = count_discursive_verbs(df_psc["text"].tolist())
    if verb_counts:
        top_verbs = collections.Counter(verb_counts).most_common(topn_verbs)
        verbs, counts = zip(*top_verbs)
        series = pd.Series(counts, index=verbs)
        save_bar(series,
                 f"Most Frequent Discursive Verbs in PSC Communiqués (Top {topn_verbs})",
                 "Verb", "Occurrences",
                 os.path.join(outdir, "psc_bar_discursive_verbs.png"),
                 color_map=colors.get("verbs"),
                 palette=colors.get("verbs_palette_strong") or colors.get("palette"))

    print(f"[DONE] Charts saved to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/analysis.csv")
    ap.add_argument("--outdir", default="output/charts")
    ap.add_argument("--topn-verbs", type=int, default=20)
    ap.add_argument("--colors-file", default="chart_colors.json")
    args = ap.parse_args()
    main(args.manifest, args.outdir, args.topn_verbs, args.colors_file)
