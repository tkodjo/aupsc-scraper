# au_psc_top3_crises.py
# Build monthly charts of the Top 3 crisis (country topics) in AUPSC communiqués.

import os, re, argparse, datetime, collections
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- helpers --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s:
        return None
    t = s
    # ISO-like: YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])[-/](3[01]|[12]?\d)\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime.date(y, mo, d)
    # D Month YYYY
    m = re.search(r"\b(3[01]|[12]?\d)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                  r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        d = int(m.group(1)); mon = m.group(2).lower(); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Month D, YYYY
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                  r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(3[01]|[12]?\d),\s*(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        mon = m.group(1).lower(); d = int(m.group(2)); y = int(m.group(3))
        mo = [i+1 for i, name in enumerate(_MONTHS) if name.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    # Year only -> set to Jan 1
    m = re.search(r"\b(20\d{2}|19\d{2})\b", t)
    if m:
        return datetime.date(int(m.group(1)), 1, 1)
    return None

def derive_date_from_row(row):
    # Try title, filename, then text (cheap & robust)
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

# Full AU Member States + SADR (must match your topic labels from au_analyze.py)
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

def month_key(dt: datetime.date):
    return pd.Period(pd.Timestamp(dt), freq="M")

def plot_top3_bar(month_label, series, out_path):
    # series: counts indexed by crisis (country)
    top = series.sort_values(ascending=True)  # for horizontal bar, smallest on top
    plt.figure()
    top.plot(kind="barh")
    plt.title(f"Top 3 crises in PSC Communiqués — {month_label}")
    plt.xlabel("Count"); plt.ylabel("Crisis (country)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# -------------------- main --------------------
def main(manifest="data/analysis.csv", outdir="output/charts/psc_top3_crises"):
    ensure_dir(outdir)
    df = pd.read_csv(manifest)

    required = {"doc_type","title","topic","text","filename"}
    if not required.issubset(df.columns):
        raise SystemExit("analysis.csv must have: doc_type,title,topic,text,filename (from au_analyze.py)")

    # 1) Filter to PSC communiqués only
    df_psc = df[df.apply(is_psc_communique, axis=1)].copy()
    if df_psc.empty:
        raise SystemExit("No PSC communiqués found in analysis.csv.")

    # 2) Derive month
    df_psc["date"]  = df_psc.apply(derive_date_from_row, axis=1)
    df_psc = df_psc.dropna(subset=["date"]).copy()
    if df_psc.empty:
        raise SystemExit("No dates could be derived for PSC communiqués.")
    df_psc["month"] = df_psc["date"].apply(month_key)

    # 3) Restrict to country topics (treat those as 'crises')
    df_psc["crisis"] = df_psc["topic"].where(df_psc["topic"].isin(COUNTRY_TOPICS), None)
    df_psc = df_psc.dropna(subset=["crisis"]).copy()
    if df_psc.empty:
        raise SystemExit("No country topics found among PSC communiqués (check your TOPIC_KEYWORDS).")

    # 4) Count per month x crisis, pick Top 3 per month
    grouped = (df_psc.groupby(["month","crisis"])
                      .size()
                      .rename("count")
                      .reset_index())

    # Prepare summary CSV rows
    summary_rows = []
    months = sorted(grouped["month"].unique())
    for m in months:
        sub = grouped[grouped["month"] == m].sort_values(["count","crisis"], ascending=[False, True])
        top3 = sub.head(3)
        if top3.empty:
            continue
        # Save per-month chart
        s = top3.set_index("crisis")["count"]
        out_png = os.path.join(outdir, f"psc_top3_crises_{str(m)}.png")
        plot_top3_bar(str(m), s, out_png)

        # Add to CSV summary
        for rank, (_, row) in enumerate(top3.iterrows(), start=1):
            summary_rows.append({
                "month": str(m),
                "rank": rank,
                "crisis": row["crisis"],
                "count": int(row["count"])
            })

    # 5) Write summary CSV
    if summary_rows:
        out_csv = os.path.join(outdir, "psc_top3_crises_by_month.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[DONE] Saved {len(summary_rows)} rows to {out_csv}")
        print(f"[DONE] Per-month charts in: {outdir}")
    else:
        print("[INFO] No Top-3 rows computed (insufficient data?).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/analysis.csv")
    ap.add_argument("--outdir", default="output/charts/psc_top3_crises")
    args = ap.parse_args()
    main(args.manifest, args.outdir)
