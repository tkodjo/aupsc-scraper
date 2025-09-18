# au_psc_crises_stacked_counts.py
# Stacked bar (counts) of Top-K country crises per month in AUPSC communiqués.

import os, re, argparse, datetime
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

_MONTHS = ("january february march april may june july august september october november december").split()

def _parse_date_from_text(s: str):
    if not isinstance(s, str) or not s: return None
    t = s
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](1[0-2]|0?[1-9])[-/](3[01]|[12]?\d)\b", t)
    if m:
        y, mo, d = map(int, m.groups()); return datetime.date(y, mo, d)
    m = re.search(r"\b(3[01]|[12]?\d)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
                  r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?)\s+"
                  r"(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        d = int(m.group(1)); mon = m.group(2).lower(); y = int(m.group(3))
        mo = [i+1 for i, n in enumerate(_MONTHS) if n.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                  r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?)\s+"
                  r"(3[01]|[12]?\d),\s*(20\d{2}|19\d{2})\b", t, re.I)
    if m:
        mon = m.group(1).lower(); d = int(m.group(2)); y = int(m.group(3))
        mo = [i+1 for i, n in enumerate(_MONTHS) if n.startswith(mon[:3])][0]
        return datetime.date(y, mo, d)
    m = re.search(r"\b(20\d{2}|19\d{2})\b", t)
    if m: return datetime.date(int(m.group(1)), 1, 1)
    return None

def derive_date(row):
    for field in ("title","filename","text"):
        d = _parse_date_from_text(str(row.get(field,"")))
        if d: return d
    return None

def is_psc_communique(row) -> bool:
    title = str(row.get("title","")).lower()
    topic = str(row.get("topic","")).lower()
    text  = str(row.get("text","")).lower()
    is_comm = str(row.get("doc_type","")).lower() == "communiqué"
    psc_hit = ("peace and security council" in title or "psc" in title or
               "peace and security council" in topic or topic.strip() == "psc" or
               " peace and security council" in text or " psc " in text)
    return is_comm and psc_hit

# Country list used to validate labels (must match your analyze.py output labels)
COUNTRIES = set([
    # North
    "Algeria","Egypt","Libya","Mauritania","Morocco","Tunisia",
    "Sahrawi Arab Democratic Republic (SADR)",
    # West
    "Benin","Burkina Faso","Cabo Verde","Côte d’Ivoire","Gambia","Ghana","Guinea",
    "Guinea-Bissau","Liberia","Mali","Niger","Nigeria","Senegal","Sierra Leone","Togo",
    # Central
    "Burundi","Cameroon","Central African Republic","Chad","Congo (Republic)",
    "Democratic Republic of Congo (DRC)","Equatorial Guinea","Gabon","São Tomé and Príncipe",
    # East
    "Comoros","Djibouti","Eritrea","Ethiopia","Kenya","Madagascar","Mauritius","Rwanda",
    "Seychelles","Somalia","South Sudan","Sudan","Tanzania","Uganda",
    # Southern
    "Angola","Botswana","Eswatini","Lesotho","Malawi","Mozambique","Namibia",
    "South Africa","Zambia","Zimbabwe",
])

def row_countries(row):
    # Prefer parsed list if present
    if "countries" in row and isinstance(row["countries"], list):
        return [c for c in row["countries"] if c in COUNTRIES]
    # CSV-friendly string
    if isinstance(row.get("countries_str"), str) and row["countries_str"].strip():
        parts = [p.strip() for p in row["countries_str"].split(";")]
        return [c for c in parts if c in COUNTRIES]
    # Fallback: single-country topic
    if row.get("topic") in COUNTRIES:
        return [row["topic"]]
    return []

def main(manifest="data/analysis.csv",
         out_png="output/charts/psc_topK_crises_stacked_counts.png",
         topk=3):
    topk = max(1, int(topk))
    ensure_dir(os.path.dirname(out_png) or ".")

    df = pd.read_csv(manifest)
    needed = {"doc_type","title","topic","text","filename"}
    if not needed.issubset(df.columns):
        raise SystemExit("analysis.csv must include: doc_type,title,topic,text,filename")

    # Filter to PSC communiqués
    df_psc = df[df.apply(is_psc_communique, axis=1)].copy()
    if df_psc.empty:
        raise SystemExit("No PSC communiqués found.")

    # Derive month
    df_psc["date"] = df_psc.apply(derive_date, axis=1)
    df_psc = df_psc.dropna(subset=["date"]).copy()
    if df_psc.empty:
        raise SystemExit("No dates could be derived for PSC communiqués.")
    df_psc["month"] = pd.to_datetime(df_psc["date"]).dt.to_period("M")

    # Build (month, country) occurrence rows (unique per doc)
    occ_rows = []
    for _, r in df_psc.iterrows():
        cs = sorted(set(row_countries(r)))
        for c in cs:
            occ_rows.append({"month": r["month"], "crisis": c})

    if not occ_rows:
        raise SystemExit("No country matches found. Ensure analyze.py writes countries/countries_str.")

    df_occ = pd.DataFrame(occ_rows)

    # Crosstab: month × crisis (counts)
    ct = pd.crosstab(df_occ["month"], df_occ["crisis"])
    if ct.empty:
        raise SystemExit("Crosstab is empty; nothing to plot.")

    # Keep only Top-K per month (others -> 0), with deterministic tie-break by name
    def keep_topk(row):
        # coerce numeric (robustness)
        counts = pd.to_numeric(row, errors="coerce").fillna(0).astype(int)
        if counts.sum() == 0:
            return counts
        tmp = pd.DataFrame({"val": counts, "name": counts.index})
        tmp = tmp.sort_values(["val","name"], ascending=[False, True])
        top_names = tmp.head(topk)["name"].tolist()
        return counts.where(counts.index.isin(top_names), other=0)

    ct_topk = ct.apply(keep_topk, axis=1)

    # Drop all-zero columns and enforce int dtype
    ct_topk = ct_topk.loc[:, ct_topk.sum(axis=0) > 0]
    if ct_topk.empty or ct_topk.shape[1] == 0:
        raise SystemExit("After Top-K filtering, there are no crises to plot.")

    ct_topk = ct_topk.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Order columns by overall prominence
    ct_topk = ct_topk[ct_topk.sum(axis=0).sort_values(ascending=False).index]

    # Plot stacked bar (counts)
    plt.figure()
    ct_topk.index = ct_topk.index.astype(str)  # "YYYY-MM"
    ax = ct_topk.plot(kind="bar", stacked=True)
    ax.set_title(f"Top-{topk} crises per month in AUPSC communiqués (counts)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Occurrences")
    plt.tight_layout()
    # Make sure filename reflects K
    out_png_final = out_png.replace("topK", f"top{topk}")
    ensure_dir(os.path.dirname(out_png_final) or ".")
    plt.savefig(out_png_final, dpi=160)
    plt.close()

    # Export the exact data used
    out_csv = os.path.splitext(out_png_final)[0] + ".csv"
    ct_topk.to_csv(out_csv, encoding="utf-8")

    print(f"[DONE] {out_png_final}")
    print(f"[DATA] {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/analysis.csv")
    ap.add_argument("--out-png", default="output/charts/psc_topK_crises_stacked_counts.png")
    ap.add_argument("--topk", type=int, default=3, help="Top-K crises per month (e.g., 3, 5, 10)")
    args = ap.parse_args()
    main(args.manifest, args.out_png, args.topk)
