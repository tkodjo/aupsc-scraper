# au_psc_crises_stacked_hist.py
# Stacked histogram of Top-3 country crises per month in AUPSC communiqués.

import os, re, argparse, datetime, json, collections
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- helpers --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s:
        return None
    t = s
    # ISO-ish: YYYY-MM-DD or YYYY/MM/DD
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

def derive_date_from_row(row):
    for field in ("title", "filename", "text"):
        d = _parse_date_from_text(str(row.get(field, "")))
        if d: return d
    return None

def is_psc_communique(row) -> bool:
    title = str(row.get("title", "")).lower()
    topic = str(row.get("topic", "")).lower()
    text = str(row.get("text", "")).lower()
    is_comm = str(row.get("doc_type", "")).lower() == "communiqué"
    psc_hit = ("peace and security council" in title or "psc" in title or
               "peace and security council" in topic or topic.strip() == "psc" or
               " peace and security council" in text or " psc " in text)
    return is_comm and psc_hit

# AU-55 + SADR (must match your au_analyze.py topic labels)
COUNTRY_TOPICS = {
    # North Africa
    "Algeria","Egypt","Libya","Mauritania","Morocco","Tunisia",
    "Sahrawi Arab Democratic Republic (SADR)",
    # West Africa
    "Benin","Burkina Faso","Cabo Verde","Côte d’Ivoire","Gambia","Ghana","Guinea",
    "Guinea-Bissau","Liberia","Mali","Niger","Nigeria","Senegal","Sierra Leone","Togo",
    # Central Africa
    "Burundi","Cameroon","Central African Republic","Chad","Congo (Republic)",
    "Democratic Republic of Congo (DRC)","Equatorial Guinea","Gabon","São Tomé and Príncipe",
    # East Africa
    "Comoros","Djibouti","Eritrea","Ethiopia","Kenya","Madagascar","Mauritius","Rwanda",
    "Seychelles","Somalia","South Sudan","Sudan","Tanzania","Uganda",
    # Southern Africa
    "Angola","Botswana","Eswatini","Lesotho","Malawi","Mozambique","Namibia",
    "South Africa","Zambia","Zimbabwe",
}

# -------------- colors --------------
def load_colors(path):
    if not path: return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "palette" not in cfg or not isinstance(cfg["palette"], list):
            cfg["palette"] = []
        if "crises" not in cfg or not isinstance(cfg["crises"], dict):
            cfg["crises"] = {}
        return cfg
    except Exception as e:
        print(f"[WARN] Couldn't load colors file {path}: {e}")
        return {"palette": [], "crises": {}}

def color_for(label, idx, colors_cfg):
    cmap = colors_cfg.get("crises", {})
    if label in cmap and cmap[label]:
        return cmap[label]
    pal = colors_cfg.get("palette", [])
    return pal[idx % len(pal)] if pal else None  # None -> Matplotlib default

# -------------- plotting --------------
def plot_stacked_hist(pivot_df, month_labels, crisis_order, out_path, colors_cfg, normalize=False):
    """
    pivot_df: DataFrame indexed by month (string), columns=crisis, values=count (only Top-3 per month retained)
    """
    # prepare bottoms
    x = range(len(month_labels))
    bottoms = [0] * len(month_labels)

    plt.figure()
    for idx, crisis in enumerate(crisis_order):
        heights = [pivot_df.loc[m, crisis] if crisis in pivot_df.columns else 0 for m in month_labels]
        if normalize:
            # convert each bar to percentage of that month total (handle zero)
            totals = pivot_df.loc[month_labels].sum(axis=1).replace(0, 1)
            heights = [h / t * 100.0 for h, t in zip(heights, totals)]
            ylabel = "Share of monthly items (%)"
        else:
            ylabel = "Count"

        color = color_for(crisis, idx, colors_cfg)
        plt.bar(x, heights, bottom=bottoms, label=crisis, color=color)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    plt.xticks(list(x), month_labels, rotation=45, ha="right")
    plt.title("Top-3 crises per month in AUPSC communiqués (stacked)")
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.legend(title="Crisis (country)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# -------------- main --------------
def main(manifest="data/analysis.csv", outdir="output/charts", colors_file=None, normalize=False):
    ensure_dir(outdir)
    df = pd.read_csv(manifest)

    required = {"doc_type","title","topic","text","filename"}
    if not required.issubset(df.columns):
        raise SystemExit("analysis.csv must have: doc_type,title,topic,text,filename")

    # Filter to PSC communiqués
    df_psc = df[df.apply(is_psc_communique, axis=1)].copy()
    if df_psc.empty:
        raise SystemExit("No PSC communiqués found.")

    # Derive month
    df_psc["date"] = df_psc.apply(derive_date_from_row, axis=1)
    df_psc = df_psc.dropna(subset=["date"]).copy()
    df_psc["month"] = pd.to_datetime(df_psc["date"]).dt.to_period("M").astype(str)

    # Keep only country topics (crises)
    df_psc["crisis"] = df_psc["topic"].where(df_psc["topic"].isin(COUNTRY_TOPICS))
    df_psc = df_psc.dropna(subset=["crisis"]).copy()

    # Count per month x crisis
    counts = (df_psc.groupby(["month","crisis"])
                    .size()
                    .rename("count")
                    .reset_index())

    if counts.empty:
        raise SystemExit("No country crises found among PSC communiqués.")

    # For each month, keep only Top-3 crises; others -> 0
    month_labels = sorted(counts["month"].unique())
    keep_pairs = set()
    for m in month_labels:
        sub = counts[counts["month"] == m].sort_values(["count","crisis"], ascending=[False, True]).head(3)
        for _, r in sub.iterrows():
            keep_pairs.add((m, r["crisis"]))

    # Pivot with only kept pairs
    counts["keep"] = counts.apply(lambda r: (r["month"], r["crisis"]) in keep_pairs, axis=1)
    counts_top3 = counts[counts["keep"]].copy()
    pivot = (counts_top3.pivot_table(index="month", columns="crisis", values="count", aggfunc="sum")
                        .reindex(index=month_labels, fill_value=0))

    # Decide column (crisis) plotting order: by total across months, desc
    crisis_order = list(pivot.sum(axis=0).sort_values(ascending=False).index)

    # Colors
    colors_cfg = load_colors(colors_file)

    # Plot stacked histogram
    out_png = os.path.join(outdir, "psc_top3_crises_stacked.png" if not normalize else "psc_top3_crises_stacked_pct.png")
    plot_stacked_hist(pivot, month_labels, crisis_order, out_png, colors_cfg, normalize=normalize)

    # Save a CSV summary (for reference)
    summary_csv = os.path.join(outdir, "psc_top3_crises_monthly_counts.csv")
    pivot.to_csv(summary_csv, encoding="utf-8")

    print(f"[DONE] Chart: {out_png}")
    print(f"[DONE] Summary CSV: {summary_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/analysis.csv")
    ap.add_argument("--outdir", default="output/charts")
    ap.add_argument("--colors-file", default="chart_colors.json")
    ap.add_argument("--normalize", action="store_true", help="Plot percentages instead of counts")
    args = ap.parse_args()
    main(args.manifest, args.outdir, args.colors_file, args.normalize)
