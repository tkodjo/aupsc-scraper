# au_psc_charts.py — PSC charts with color support from chart_colors.json
import os, re, argparse, math, collections, datetime, json
import pandas as pd
import matplotlib.pyplot as plt

# ---------- I/O helpers ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_colors(path):
    if not path: return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            colors = json.load(f)
        # tiny sanity: normalize keys we expect to be dicts
        for k in ("tone", "intent", "verbs"):
            if k in colors and not isinstance(colors[k], dict):
                colors[k] = {}
        if "palette" in colors and not isinstance(colors["palette"], list):
            colors["palette"] = []
        print(f"[INFO] Loaded colors from {path}")
        return colors
    except Exception as e:
        print(f"[WARN] Could not load colors file {path}: {e}")
        return {}

# Small helper: map labels -> colors (fall back to palette if provided)
def map_colors(labels, color_map=None, palette=None):
    if color_map:
        colors = [color_map.get(lbl) for lbl in labels]
        # if any is None and palette exists, fill with palette cycle
        if palette:
            for i, c in enumerate(colors):
                if c is None:
                    colors[i] = palette[i % len(palette)] if palette else None
        return colors
    if palette:
        return [palette[i % len(palette)] for i in range(len(labels))]
    return None

def save_pie(series, title, out_path, color_map=None, palette=None):
    plt.figure()
    s = series[series > 0]
    if s.empty:
        print(f"[WARN] Nothing to plot for {title}")
        plt.close(); return
    labels = s.index.tolist()
    colors = map_colors(labels, color_map, palette)
    plt.pie(s.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_line(df_wide, title, xlabel, ylabel, out_path, color_map=None, palette=None):
    if df_wide.empty:
        print(f"[WARN] Nothing to plot for {title}")
        return
    plt.figure()
    # convert PeriodIndex to Timestamp for nicer x-axis
    x = df_wide.index
    if hasattr(x, "to_timestamp"):
        x = x.to_timestamp()
    for i, col in enumerate(df_wide.columns):
        color = None
        if color_map and col in color_map:
            color = color_map[col]
        elif palette:
            color = palette[i % len(palette)]
        plt.plot(x, df_wide[col], marker="o", label=col, color=color)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_bar(series_or_df, title, xlabel, ylabel, out_path, color_map=None, palette=None):
    plt.figure()
    if isinstance(series_or_df, pd.Series):
        labels = series_or_df.index.tolist()
        colors = map_colors(labels, color_map, palette)
        series_or_df.plot(kind="bar", color=colors)
    else:
        # DataFrame: try to color by columns (stacked handled by caller)
        if color_map or palette:
            cols = series_or_df.columns.tolist()
            colors = map_colors(cols, color_map, palette)
            series_or_df.plot(kind="bar", color=colors)
        else:
            series_or_df.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------- Date parsing ----------
_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s:
        return None
    t = s
    # ISO: YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])[-/](3[01]|[12]?\d)\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime.date(y, mo, d)
    # D Month YYYY
    m = re.search(r"\b(3[01]|[12]?\d)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                  r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2}|19\d{2})\b",
                  t, flags=re.IGNORECASE)
    if m:
        d = int(m.group(1)); mon = m.group(2).lower(); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Month D, YYYY
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                  r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+"
                  r"(3[01]|[12]?\d),\s*(20\d{2}|19\d{2})\b", t, flags=re.IGNORECASE)
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

# ---------- Diplomatic intent & verbs ----------
INTENT_LEXICON = {
    "Condemnation": [r"\bcondemn(s|ed|ing)?\b", r"\bdeplor(e|es|ed|ing)\b", r"\bdenounc(e|es|ed|ing)\b"],
    "Concern/Regret": [r"\b(express(es|ed|ing)? )?(deep )?concern(s)?\b", r"\bregret(s|ted|ting)?\b"],
    "Call/Urge/Request": [r"\burge(s|d|ing)?\b", r"\bcall(s|ed|ing)? (upon|on)\b", r"\brequest(s|ed|ing)?\b",
                          r"\bappeal(s|ed|ing)? (to|for)\b", r"\binvite(s|d|ing)?\b", r"\bencourage(s|d|ing)?\b"],
    "Welcome/Commend/Appreciate": [r"\bwelcom(e|es|ed|ing)\b", r"\bcommend(s|ed|ing)?\b", r"\bappreciate(s|d|ing)?\b", r"\bapplaud(s|ed|ing)?\b"],
    "Decide/Authorize/Adopt": [r"\bdecide(s|d|ing)?\b", r"\bauthori[sz]e(s|d|ing)?\b", r"\badopt(s|ed|ing)?\b",
                               r"\bapprove(s|d|ing)?\b", r"\bendorse(s|d|ing)?\b"],
    "Reiterate/Affirm/Recall": [r"\breiterate(s|d|ing)?\b", r"\breaffirm(s|ed|ing)?\b", r"\brecall(s|ed|ing)?\b",
                                r"\bemphasise?s?\b", r"\bunderline(s|d|ing)?\b", r"\baffirm(s|ed|ing)?\b"],
    "Sanctions/Suspension": [r"\bsanction(s|ed|ing)?\b", r"\bsuspend(s|ed|ing)?\b", r"\breject(s|ed|ing)?\b"],
    "Mandate/Extend/Renew": [r"\bmandate(s|d|ing)?\b", r"\bextend(s|ed|ing)?\b", r"\brenew(s|ed|ing)?\b"],
    "Deploy/Support/Assist": [r"\bdeploy(s|ed|ing)?\b", r"\bsupport(s|ed|ing)?\b", r"\bassist(s|ed|ing)?\b"],
    "Note/Take Note": [r"\bnote(s|d|ing)?\b", r"\btake(s|n)? note\b"],
}
INTENT_ORDER = list(INTENT_LEXICON.keys()) + ["Other/General"]

DISCursive_VERBS = {
    "condemn": r"\bcondemn(s|ed|ing)?\b",
    "urge": r"\burge(s|d|ing)?\b",
    "call": r"\bcall(s|ed|ing)?\b",
    "welcome": r"\bwelcom(e|es|ed|ing)\b",
    "commend": r"\bcommend(s|ed|ing)?\b",
    "express": r"\bexpress(es|ed|ing)?\b",
    "reiterate": r"\breiterate(s|d|ing)?\b",
    "reaffirm": r"\breaffirm(s|ed|ing)?\b",
    "recall": r"\brecall(s|ed|ing)?\b",
    "decide": r"\bdecide(s|d|ing)?\b",
    "authorize": r"\bauthori[sz]e(s|d|ing)?\b",
    "adopt": r"\badopt(s|ed|ing)?\b",
    "approve": r"\bapprove(s|d|ing)?\b",
    "endorse": r"\bendorse(s|d|ing)?\b",
    "sanction": r"\bsanction(s|ed|ing)?\b",
    "suspend": r"\bsuspend(s|ed|ing)?\b",
    "reject": r"\breject(s|ed|ing)?\b",
    "mandate": r"\bmandate(s|d|ing)?\b",
    "extend": r"\bextend(s|ed|ing)?\b",
    "renew": r"\brenew(s|ed|ing)?\b",
    "deploy": r"\bdeploy(s|ed|ing)?\b",
    "support": r"\bsupport(s|ed|ing)?\b",
    "assist": r"\bassist(s|ed|ing)?\b",
    "note": r"\bnote(s|d|ing)?\b",
    "appeal": r"\bappeal(s|ed|ing)?\b",
    "invite": r"\binvite(s|d|ing)?\b",
    "encourage": r"\bencourage(s|d|ing)?\b",
    "denounce": r"\bdenounc(e|es|ed|ing)\b",
    "deplore": r"\bdeplor(e|es|ed|ing)\b",
    "emphasize": r"\bemphasise?s?\b",
    "underline": r"\bunderline(s|d|ing)?\b",
    "affirm": r"\baffirm(s|ed|ing)?\b",
    "warn": r"\bwarn(s|ed|ing)?\b",
    "demand": r"\bdemand(s|ed|ing)?\b",
    "direct": r"\bdirect(s|ed|ing)?\b",
    "appreciate": r"\bappreciate(s|d|ing)?\b",
    "applaud": r"\bapplaud(s|ed|ing)?\b",
    "request": r"\brequest(s|ed|ing)?\b"
}

def intent_from_text(text: str) -> str:
    if not isinstance(text, str): return "Other/General"
    t = text.lower()
    best_cat, best_hits = "Other/General", 0
    for cat in INTENT_LEXICON:
        hits = sum(len(re.findall(pat, t)) for pat in INTENT_LEXICON[cat])
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat

def count_discursive_verbs(texts):
    counts = collections.Counter()
    for t in texts:
        if not isinstance(t, str): continue
        s = t.lower()
        for lemma, pat in DISCursive_VERBS.items():
            hits = len(re.findall(pat, s))
            if hits:
                counts[lemma] += hits
    return counts

# ---------- Filters ----------
def is_psc(row) -> bool:
    title = str(row.get("title", "")).lower()
    topic = str(row.get("topic", "")).lower()
    text = str(row.get("text", "")).lower()
    cond_type = str(row.get("doc_type", "")).lower() == "communiqué"
    psc_hit = ("peace and security council" in title or "psc" in title or
               "peace and security council" in topic or "psc" == topic or
               " peace and security council" in text or " psc " in text)
    return cond_type and psc_hit

# ---------- Main ----------
def main(manifest="data/analysis.csv", outdir="output/charts", topn_verbs=20, colors_file=None):
    ensure_dir(outdir)
    colors = load_colors(colors_file)
    palette = colors.get("palette", [])

    df = pd.read_csv(manifest)
    needed = {"doc_type","title","topic","text","sent_label","filename"}
    if not needed.issubset(df.columns):
        raise SystemExit("analysis.csv must have columns: doc_type,title,topic,text,sent_label,filename")

    df_psc = df[df.apply(is_psc, axis=1)].copy()
    if df_psc.empty:
        raise SystemExit("No PSC communiqués found. Check your filters or ensure 'doc_type' and 'topic' are populated.")

    # Dates (monthly)
    df_psc["date"] = df_psc.apply(derive_date, axis=1)
    df_psc["month"] = pd.to_datetime(df_psc["date"], errors="coerce").dt.to_period("M")
    
     # Diplomatic Intent
    df_psc["intent"] = df_psc["text"].apply(intent_from_text)
    
    df_trend = df_psc.dropna(subset=["month"]).copy()

   

    # --- PIE ---
    intent_counts = df_psc["intent"].value_counts().reindex(INTENT_ORDER, fill_value=0)
    save_pie(intent_counts, "PSC Communiqués: Diplomatic Intent Distribution",
             os.path.join(outdir, "psc_pie_intent.png"),
             color_map=colors.get("intent"), palette=palette)

    tone_counts = df_psc["sent_label"].value_counts()
    save_pie(tone_counts, "PSC Communiqués: Tone (Sentiment) Distribution",
             os.path.join(outdir, "psc_pie_tone.png"),
             color_map=colors.get("tone"), palette=palette)

    # --- LINE (monthly trends) ---
    tone_trend = (df_trend
                  .groupby(["month", "sent_label"])
                  .size()
                  .unstack(fill_value=0)
                  .sort_index())
    save_line(tone_trend, "PSC Communiqués: Tone Trend (Monthly)", "Month", "Count",
              os.path.join(outdir, "psc_line_tone_trend.png"),
              color_map=colors.get("tone"), palette=palette)

    top_intents = df_psc["intent"].value_counts().head(6).index.tolist()
    intent_trend = (df_trend.assign(intent=lambda x: x["intent"].where(x["intent"].isin(top_intents), "Other/General"))
                    .groupby(["month", "intent"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index())
    save_line(intent_trend, "PSC Communiqués: Diplomatic Intent Trend (Monthly)", "Month", "Count",
              os.path.join(outdir, "psc_line_intent_trend.png"),
              color_map=colors.get("intent"), palette=palette)

    # --- BAR (verbs) ---
    verb_counts = count_discursive_verbs(df_psc["text"].tolist())
    if verb_counts:
        top_verbs = collections.Counter(verb_counts).most_common(topn_verbs)
        verbs, counts = zip(*top_verbs)
        series = pd.Series(counts, index=verbs)
        save_bar(series, f"Most Frequent Discursive Verbs in PSC Communiqués (Top {topn_verbs})",
                 "Verb", "Occurrences",
                 os.path.join(outdir, "psc_bar_discursive_verbs.png"),
                 color_map=colors.get("verbs"), palette=palette)

    print(f"[DONE] Charts saved to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/analysis.csv")
    ap.add_argument("--outdir", default="output/charts")
    ap.add_argument("--topn-verbs", type=int, default=20)
    ap.add_argument("--colors-file", default="chart_colors.json")
    args = ap.parse_args()
    main(args.manifest, args.outdir, args.topn_verbs, args.colors_file)
