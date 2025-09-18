# au_psc_crises_stacked_counts.py
# Stacked monthly histogram of Top-K country mentions in AUPSC communiqués.
# - Always skips the first PDF page for counting (venue/city page).
# - X axis: month abbreviations (Jan, Feb, ...).
# - Colors: uses au_countries_colors.json; avoids near-white and too-similar hues.
# - Order control: --order desc|asc (stack bottom->top).

import os, re, json, argparse, unicodedata, datetime, colorsys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IO ----------
def ensure_dir(p: str):
    if p: os.makedirs(p, exist_ok=True)

# ---------- Colors ----------
def load_country_colors(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("countries", {}) or cfg  # accept {"countries": {...}} or plain map

def _is_near_white(hexstr: str, thresh: float = 0.90) -> bool:
    try:
        h = hexstr.strip().lstrip("#")
        if len(h) == 3: h = "".join(ch*2 for ch in h)
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        def srgb2lin(c):
            x = c/255.0
            return x/12.92 if x <= 0.04045 else ((x+0.055)/1.055)**2.4
        R,G,B = srgb2lin(r), srgb2lin(g), srgb2lin(b)
        Y = 0.2126*R + 0.7152*G + 0.0722*B
        return Y >= thresh
    except Exception:
        return True

def _hex_to_hsl(hexstr: str):
    h = hexstr.strip().lstrip("#")
    if len(h) == 3: h = "".join(ch*2 for ch in h)
    r,g,b = int(h[0:2],16)/255.0, int(h[2:4],16)/255.0, int(h[4:6],16)/255.0
    hh, ll, ss = colorsys.rgb_to_hls(r,g,b)
    return (hh, ss, ll)  # h in [0,1)

def _hsl_to_hex(h: float, s: float, l: float) -> str:
    r,g,b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(round(r*255)), int(round(g*255)), int(round(b*255)))

def _min_hue_distance(hex_list, default_min=1.0):
    hues = []
    for hx in hex_list:
        try:
            h, s, l = _hex_to_hsl(hx); hues.append(h)
        except Exception:
            pass
    if len(hues) < 2: return default_min
    mind = 1.0
    for i in range(len(hues)):
        for j in range(i+1, len(hues)):
            d = abs(hues[i]-hues[j]); d = min(d, 1.0-d)
            if d < mind: mind = d
    return mind

def _generate_distinct_pastels(n: int, s: float=0.58, l: float=0.62, offset: float=0.07):
    return [_hsl_to_hex((offset + i/max(1,n)) % 1.0, s, l) for i in range(n)]

def pick_colors_for_countries(cols, cmap, hue_threshold=0.08):
    """
    Return colors in the SAME order as cols.
    If provided colors are near-white or too-similar in hue, generate distinct pastels.
    """
    provided = []
    ok = True
    for c in cols:
        col = cmap.get(c)
        if not col or _is_near_white(col):
            ok = False
        provided.append(col)
    if ok and _min_hue_distance(provided) >= hue_threshold:
        return provided
    return _generate_distinct_pastels(len(cols), s=0.58, l=0.62, offset=0.07)

# ---------- PDF text extraction ----------
def extract_text_pages(path: str) -> list[str]:
    try:
        import fitz  # PyMuPDF
        pages = []
        with fitz.open(path) as doc:
            for pg in doc:
                pages.append(pg.get_text())
        return pages if pages else [""]
    except Exception:
        pass
    try:
        from pdfminer.high_level import extract_text
        full = extract_text(path) or ""
        parts = [p.strip() for p in re.split(r"\f+", full) if p.strip()]
        return parts if parts else [full]
    except Exception:
        return [""]

# ---------- Normalization ----------
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = _strip_accents(s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- PSC detection ----------
def guess_doc_type(title: str, filename: str, text: str) -> str:
    blob = f"{title or ''} {filename or ''} {text or ''}".lower()
    if re.search(r"\bcommuniqu[eé]|communique\b", blob) or re.search(r"\.comm[-_.](en|fr)\.pdf$", (filename or "").lower()):
        return "Communiqué"
    if re.search(r"\bpress (statement|release)\b", blob): return "Press Statement"
    if re.search(r"\breport\b", blob): return "Report"
    return "Other"

def is_psc_communique(doc_type: str, title: str, text: str) -> bool:
    if (doc_type or "").lower() != "communiqué": return False
    blob = f"{title or ''} {text or ''}".lower()
    return ("peace and security council" in blob) or (re.search(r"\bpsc\b", blob) is not None)

# ---------- Date helpers ----------
def parse_row_date(s: str):
    if not isinstance(s, str): return None
    m = re.search(r"\b(3[01]|[12]?\d)/(1[0-2]|0?\d)/((19|20)\d{2})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return datetime.date(y, mo, d)
        except ValueError: pass
    m = re.search(r"\b(20\d{2}|19\d{2})-(1[0-2]|0?\d)-(3[01]|[12]?\d)\b", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return datetime.date(y, mo, d)
        except ValueError: pass
    m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
    if m: return datetime.date(int(m.group(1)), 1, 1)
    return None

# ---------- Country aliases ----------
def build_aliases() -> dict:
    A = {
        "Algeria": ["algeria"],
        "Angola": ["angola"],
        "Benin": ["benin"],
        "Botswana": ["botswana"],
        "Burkina Faso": ["burkina faso"],
        "Burundi": ["burundi"],
        "Cabo Verde": ["cabo verde", "cape verde"],
        "Cameroon": ["cameroon"],
        "Central African Republic": ["central african republic", "car (central african republic)"],
        "Chad": ["chad"],
        "Comoros": ["comoros", "union of the comoros"],
        "Congo (Republic of the Congo)": ["republic of the congo", "congo brazzaville", "congo-brazzaville"],
        "Côte d’Ivoire": ["cote d ivoire", "cote d'ivoire", "ivory coast"],
        "Democratic Republic of Congo (DRC)": [
            "democratic republic of the congo", "democratic republic of congo", "drc", "dr congo", "congo kinshasa"
        ],
        "Djibouti": ["djibouti"],
        "Egypt": ["egypt"],
        "Equatorial Guinea": ["equatorial guinea"],
        "Eritrea": ["eritrea"],
        "Eswatini (Swaziland)": ["eswatini", "swaziland"],
        "Ethiopia": ["ethiopia"],
        "Gabon": ["gabon"],
        "Gambia": ["gambia", "the gambia"],
        "Ghana": ["ghana"],
        "Guinea": ["republic of guinea", "guinea"],
        "Guinea-Bissau": ["guinea bissau", "guinea-bissau"],
        "Kenya": ["kenya"],
        "Lesotho": ["lesotho"],
        "Liberia": ["liberia"],
        "Libya": ["libya"],
        "Madagascar": ["madagascar"],
        "Malawi": ["malawi"],
        "Mali": ["mali"],
        "Mauritania": ["mauritania"],
        "Mauritius": ["mauritius"],
        "Morocco": ["morocco"],
        "Mozambique": ["mozambique"],
        "Namibia": ["namibia"],
        "Niger": ["niger"],
        "Nigeria": ["nigeria"],
        "Rwanda": ["rwanda"],
        "Sahrawi Arab Democratic Republic": ["sahrawi arab democratic republic", "sadr", "western sahara"],
        "São Tomé and Príncipe": ["sao tome and principe", "sao tome", "são tome", "principe"],
        "Senegal": ["senegal"],
        "Seychelles": ["seychelles"],
        "Sierra Leone": ["sierra leone"],
        "Somalia": ["somalia"],
        "South Africa": ["south africa"],
        "South Sudan": ["south sudan"],
        "Sudan": ["sudan"],
        "Tanzania": ["tanzania", "united republic of tanzania"],
        "Togo": ["togo"],
        "Tunisia": ["tunisia"],
        "Uganda": ["uganda"],
        "Zambia": ["zambia"],
        "Zimbabwe": ["zimbabwe"]
    }
    return {k: [norm_text(x) for x in v] for k, v in A.items()}

def count_countries(norm_text_blob: str, alias_map: dict) -> dict:
    out = {k: 0 for k in alias_map}
    for country, aliases in alias_map.items():
        n = 0
        for a in aliases:
            if not a: continue
            pat = r"\b" + re.escape(a) + r"\b"
            n += len(re.findall(pat, norm_text_blob))
        out[country] = n
    return out

# ---------- Plot (manual stacking to honor order) ----------
def plot_stacked_manual(df_counts: pd.DataFrame, country_colors: dict, out_png: str, title: str):
    ensure_dir(os.path.dirname(out_png))
    cols = list(df_counts.columns)           # in desired stacking order (left->right == bottom->top)
    colors = pick_colors_for_countries(cols, country_colors)

    fig, ax = plt.subplots(figsize=(11,7))
    x_pos = range(len(df_counts.index))
    bottoms = [0]*len(df_counts.index)

    # draw in column order so first column is at the bottom of each stack
    for i, col in enumerate(cols):
        vals = df_counts[col].values
        ax.bar(x_pos, vals, bottom=bottoms, label=col, color=colors[i])
        bottoms = [b+v for b, v in zip(bottoms, vals)]

    # Month abbreviations on x-axis
    xlabs = []
    for p in df_counts.index:
        try: ts = p.to_timestamp()
        except Exception: ts = pd.to_datetime(str(p))
        xlabs.append(ts.strftime("%b"))
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(xlabs, rotation=0)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Occurrences")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------- Main ----------
def main(manifest="data/manifest.csv",
         out_png="output/charts/psc_top10_crises_stacked_counts.png",
         country_colors_file="au_countries_colors.json",
         topk=10,
         year=None,
         order="desc"):
    if not os.path.exists(manifest):
        raise SystemExit(f"Manifest not found: {manifest}")
    ensure_dir(os.path.dirname(out_png))

    dfm = pd.read_csv(manifest)
    if dfm.empty:
        raise SystemExit("Manifest is empty.")

    country_colors = load_country_colors(country_colors_file)
    aliases = build_aliases()
    month_map = {}

    for _, row in dfm.iterrows():
        # Parse a date (manifest 'date', then row_text/title/filename)
        d = None
        for key in ("date", "row_text", "title", "filename"):
            if key in row and pd.notna(row[key]) and row[key]:
                d = parse_row_date(str(row[key])); 
                if d: break
        if not d or (year and d.year != int(year)):
            continue

        # Extract text and SKIP FIRST PAGE
        path = row.get("path") or os.path.join("data", "pdfs", row.get("filename", ""))
        pages = extract_text_pages(path)
        pages_use = pages[1:] if len(pages) > 1 else []
        text = "\n\n".join(pages_use)
        norm_blob = norm_text(text)

        # Only PSC communiqués
        doc_type = guess_doc_type(str(row.get("title","")), str(row.get("filename","")), text)
        if not is_psc_communique(doc_type, str(row.get("title","")), text):
            continue

        counts = count_countries(norm_blob, aliases)
        month = pd.Period(d, freq="M")
        if month not in month_map:
            month_map[month] = {k: 0 for k in aliases}
        for c, n in counts.items():
            month_map[month][c] += int(n)

    if not month_map:
        raise SystemExit("No counts found (check year filter and PSC detection).")

    df = pd.DataFrame.from_dict(month_map, orient="index").sort_index().fillna(0).astype(int)

    # Top-K by total occurrences
    totals = df.sum(axis=0)
    top_list = totals.sort_values(ascending=False).head(int(topk)).index.tolist()

    # Stack order (columns)
    if str(order).lower() == "asc":
        stack_cols = df[top_list].sum(axis=0).sort_values(ascending=True).index.tolist()   # least at bottom
    else:
        stack_cols = df[top_list].sum(axis=0).sort_values(ascending=False).index.tolist()  # most at bottom (default)

    df_top = df.reindex(columns=stack_cols)
    df_top = df_top[(df_top.sum(axis=1) > 0)]
    if df_top.empty:
        raise SystemExit("No non-zero counts to plot after filtering.")

    plot_stacked_manual(
        df_top,
        country_colors,
        out_png,
        title=f"Top {int(topk)} crises per month in AUPSC communiqués (counts)"
    )
    print(f"[DONE] Saved {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stacked monthly histogram of Top-K country mentions in AUPSC communiqués")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-png", default="output/charts/psc_top10_crises_stacked_counts.png")
    ap.add_argument("--country-colors-file", default="au_countries_colors.json")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--year", type=int, default=None, help="Filter to a specific year (e.g., 2025)")
    ap.add_argument("--order", choices=["desc","asc"], default="desc",
                    help="Stacking order (bottom->top): 'desc' = most to least; 'asc' = least to most")
    args = ap.parse_args()
    main(args.manifest, args.out_png, args.country_colors_file, args.topk, args.year, args.order)
