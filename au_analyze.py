# au_analyze.py
import os, re, argparse
import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# One-time VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# ---------- Utilities ----------
def pdf_to_text(path:str) -> str:
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def bar_save(series_or_df, title, xlabel, ylabel, out_path, stacked=False):
    plt.figure()
    if hasattr(series_or_df, "plot"):
        series_or_df.plot(kind="bar", stacked=stacked)
    else:
        plt.bar(range(len(series_or_df)), series_or_df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def hist_save(series, bins, title, xlabel, out_path):
    plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------- Classification helpers ----------
def classify_type(title:str, filename:str, text:str) -> str:
    t = f"{title} {filename} {text[:4000]}".lower()
    if "press statement" in t or "press release" in t:
        return "Press Statement"
    if "communiqué" in t or "communique" in t or ".comm" in filename.lower():
        return "Communiqué"
    if "report of the chairperson" in t or re.search(r"\breport\b", t):
        return "Report"
    if "decision" in t or "declaration" in t or "resolution" in t:
        return "Decision/Declaration"
    if "speech" in t or "address by" in t or "remarks by" in t:
        return "Speech"
    return "Other"

def sentiment_scores(sia, text:str):
    # VADER prefers shorter text; sample head and tail for long docs
    chunk = (text[:6000] + "\n" + text[-4000:]) if len(text) > 12000 else text
    return sia.polarity_scores(chunk)

def label_from_compound(c: float) -> str:
    if c >= 0.05: return "Positive"
    if c <= -0.05: return "Negative"
    return "Neutral"

# ---------- Expanded Topics (I–V) ----------
TOPIC_KEYWORDS = {
    # I. Thematic Areas (Expanded)
    "African Standby Force (ASF)": ["african standby force", "asf"],
    "AU Border Programme (AUBP)": ["au border programme", "aubp"],
    "African Peace and Security Architecture (APSA)": ["apsa", "african peace and security architecture"],
    "Disarmament": ["disarmament"],
    "Early Warning / Conflict Prevention": ["early warning", "conflict prevention", "cews"],
    "Women and Children in Armed Conflicts": ["women in conflict", "children in armed conflict", "acerwc"],
    "Youth, Peace and Security (YPS)": ["yps", "youth peace and security"],
    "Humanitarian Assistance and Disaster Response": ["humanitarian assistance", "disaster response", "relief aid"],
    "Maritime Security and Safety": ["maritime security", "naval safety", "piracy"],
    "Cybersecurity / Digital Threats": ["cybersecurity", "digital threats", "cyber attack", "cyber crime"],
    "Climate Change and Security": ["climate change", "environmental security", "climate security"],
    "Health Security": ["health security", "pandemic", "epidemic", "biosecurity", "covid"],
    "Partnerships": ["united nations", "european union", "eu", "ecowas", "sadc", "bilateral", "civil society", "csos"],
    "Peace Agreements / Mediation Support": ["peace agreement", "ceasefire", "mediation", "negotiation"],
    "Peacekeeping Operations & Missions": ["peacekeeping", "mission", "operation", "amisom", "atmis", "mnjtf"],
    "Protection of Civilians (PoC)": ["protection of civilians", "poc"],
    "Post-Conflict Reconstruction and Development (PCRD)": ["post-conflict reconstruction", "pcrd"],
    "Security Sector Reform (SSR) / DDR": ["security sector reform", "ssr", "disarmament demobilisation reintegration", "ddr"],
    "Terrorism and Transnational Organized Crime": ["terrorism", "violent extremism", "organized crime", "boko haram", "al shabaab", "isis"],
    "Piracy and Illicit Trafficking": ["illicit trafficking", "arms smuggling", "drug trafficking", "human trafficking", "wildlife trafficking"],
    "Unconstitutional Changes of Government (UCG)": ["unconstitutional change of government", "coup d'etat", "coup d’état", "military takeover"],
    "Year of Peace / Make Peace Happen Campaign": ["year of peace", "make peace happen"],
    "African Union High Implementation Panel (AUHIP)": ["auhip", "high implementation panel"],
    "Democracy / Governance / Elections": ["election", "elections", "governance", "democracy", "referendum"],
    "Refugees, Returnees & Internally Displaced Persons (IDPs)": ["refugee", "refugees", "internally displaced", "idp", "returnees"],
    "Apartheid / Decolonization / Self-Determination Issues": ["apartheid", "self-determination", "colonialism", "decolonization", "decolonisation"],
    "Common African Defence and Security Policy (CADSP)": ["cadsp", "common african defence and security policy"],
    "Small Arms and Light Weapons (SALW) Control": ["small arms", "light weapons", "salw"],
    "Nuclear Non-Proliferation and Arms Control": ["nuclear non-proliferation", "arms control", "pelindaba treaty"],
    "Human Rights and Rule of Law in Conflict Situations": ["human rights", "rule of law", "violations in conflict"],
    "Cross-Border and Inter-communal Conflicts": ["cross-border conflict", "inter-communal violence", "intercommunal"],

    # II. AU Organs & Structures (Expanded)
    "Assembly – Heads of State and Government": ["assembly of heads of state", "au assembly"],
    "Executive Council – Ministers": ["executive council"],
    "AU Commission (AUC)": ["au commission", "auc"],
    "Peace and Security Council (PSC)": ["peace and security council", "psc"],
    "Panel of the Wise": ["panel of the wise"],
    "CISSA": ["committee of intelligence and security services of africa", "cissa"],
    "African Court on Human and Peoples’ Rights (AfCHPR)": ["african court on human and peoples’ rights", "afchpr"],
    "African Commission on Human and Peoples’ Rights (ACHPR)": ["achpr", "african commission on human and peoples’ rights"],
    "ACERWC": ["acerwc", "committee of experts on the rights and welfare of the child"],
    "APRM": ["african peer review mechanism", "aprm"],
    "ECOSOCC": ["economic, social and cultural council", "ecosocc"],
    "Permanent Representatives’ Committee (PRC)": ["permanent representatives committee", "prc"],
    "Specialized Technical Committees (STCs)": ["specialized technical committee", "stc"],
    "African Union Development Agency (AUDA-NEPAD)": ["auda-nepad", "nepad"],
    "ACSRT (Algiers)": ["african centre for the study and research on terrorism", "acsrt"],
    "Continental Early Warning System (CEWS)": ["continental early warning system", "cews"],
    "ASF Regional Brigades": ["east standby brigade", "southern standby brigade", "western standby brigade", "central standby brigade", "north standby brigade"],

    # III. Legal Instruments (Expanded)
    "Constitutive Act of the AU": ["constitutive act of the african union"],
    "PSC Protocol": ["protocol relating to the establishment of the peace and security council", "psc protocol"],
    "Pelindaba Treaty": ["pelindaba treaty", "african nuclear-weapon-free zone treaty"],
    "Lomé Declaration (2000) on UCG": ["lome declaration", "2000 declaration on unconstitutional changes of government"],
    "ACDEG (2007)": ["african charter on democracy elections and governance", "acdeg"],
    "OAU/AU Terrorism Convention (1999) + 2004 Protocol": ["oau convention on the prevention and combating of terrorism", "terrorism protocol"],
    "AU Non-Aggression and Common Defence Pact (2005)": ["non-aggression and common defence pact"],
    "OAU Mercenarism Convention (1977)": ["convention for the elimination of mercenarism in africa", "mercenarism convention"],
    "African Charter on Human and Peoples’ Rights (1981)": ["african charter on human and peoples’ rights"],
    "African Charter on the Rights and Welfare of the Child (1990)": ["african charter on the rights and welfare of the child"],
    "AU Refugee Convention (1969)": ["au convention governing specific aspects of refugee problems in africa", "oau refugee convention"],
    "Kampala Convention (2009) on IDPs": ["kampala convention", "idp convention"],
    "PCRD Policy Framework (2006)": ["policy framework on post-conflict reconstruction and development", "pcrd framework"],
    "CADSP (2004)": ["common african defence and security policy"],

    # IV. Regional Economic Communities (RECs)
    "CEN-SAD": ["cen-sad", "community of sahel-saharan states"],
    "ECCAS": ["eccas", "economic community of central african states"],
    "COMESA": ["comesa", "common market for eastern and southern africa"],
    "ECOWAS": ["ecowas", "economic community of west african states"],
    "IGAD": ["igad", "intergovernmental authority on development"],
    "SADC": ["sadc", "southern african development community"],
   "UMA": ["uma", "arab maghreb union"],
}  # <-- Close the dictionary here

# IV. Regional Economic Communities (RECs)
TOPIC_KEYWORDS.update({
    "CEN-SAD": ["cen-sad", "community of sahel-saharan states"],
    "ECCAS": ["eccas", "economic community of central african states"],
    "COMESA": ["comesa", "common market for eastern and southern africa"],
    "ECOWAS": ["ecowas", "economic community of west african states"],
    "IGAD": ["igad", "intergovernmental authority on development"],
    "SADC": ["sadc", "southern african development community"],
    "UMA": ["uma", "arab maghreb union"],
})

# V. AU Member States (55) + SADR
TOPIC_KEYWORDS.update({
    # North Africa (6) + SADR
    "Algeria": ["algeria", "algiers"],
    "Egypt": ["egypt", "cairo"],
    "Libya": ["libya", "tripoli", "benghazi"],
    "Mauritania": ["mauritania", "nouakchott"],
    "Morocco": ["morocco", "rabat", "western sahara"],
    "Tunisia": ["tunisia", "tunis"],
    "Sahrawi Arab Democratic Republic (SADR)": ["sadr", "sahrawi", "western sahara"],

    # West Africa (15)
    "Benin": ["benin", "porto-novo", "cotonou"],
    "Burkina Faso": ["burkina faso", "ouagadougou"],
    "Cabo Verde": ["cabo verde", "cape verde", "praia"],
    "Côte d’Ivoire": ["cote d’ivoire", "ivory coast", "abidjan", "yamoussoukro", "yamoussukro"],
    "Gambia": ["gambia", "banjul"],
    "Ghana": ["ghana", "accra"],
    "Guinea": ["guinea", "conakry"],
    "Guinea-Bissau": ["guinea-bissau", "bissau"],
    "Liberia": ["liberia", "monrovia"],
    "Mali": ["mali", "bamako"],
    "Niger": ["niger", "niamey"],
    "Nigeria": ["nigeria", "abuja", "lagos"],
    "Senegal": ["senegal", "dakar"],
    "Sierra Leone": ["sierra leone", "freetown"],
    "Togo": ["togo", "lomé", "lome"],

    # Central Africa (9)
    "Burundi": ["burundi", "bujumbura", "gitega"],
    "Cameroon": ["cameroon", "yaoundé", "yaounde", "douala"],
    "Central African Republic": ["central african republic", "car", "bangui"],
    "Chad": ["chad", "ndjamena", "n’djamena", "n'djamena"],
    "Congo (Republic)": ["republic of the congo", "congo-brazzaville", "brazzaville"],
    "Democratic Republic of Congo (DRC)": ["democratic republic of congo", "drc", "kinshasa", "goma"],
    "Equatorial Guinea": ["equatorial guinea", "malabo"],
    "Gabon": ["gabon", "libreville"],
    "São Tomé and Príncipe": ["sao tome", "são tomé", "principe", "sao tome and principe", "são tomé and príncipe"],

    # East Africa (14)
    "Comoros": ["comoros", "moroni"],
    "Djibouti": ["djibouti"],
    "Eritrea": ["eritrea", "asmara"],
    "Ethiopia": ["ethiopia", "addis ababa"],
    "Kenya": ["kenya", "nairobi", "mombasa"],
    "Madagascar": ["madagascar", "antananarivo"],
    "Mauritius": ["mauritius", "port louis"],
    "Rwanda": ["rwanda", "kigali"],
    "Seychelles": ["seychelles", "victoria"],
    "Somalia": ["somalia", "mogadishu"],
    "South Sudan": ["south sudan", "juba"],
    "Sudan": ["sudan", "khartoum", "darfur"],
    "Tanzania": ["tanzania", "dodoma", "dar es salaam"],
    "Uganda": ["uganda", "kampala"],

    # Southern Africa (11)
    "Angola": ["angola", "luanda"],
    "Botswana": ["botswana", "gaborone"],
    "Eswatini": ["eswatini", "swaziland", "mbabane", "lobamba"],
    "Lesotho": ["lesotho", "maseru"],
    "Malawi": ["malawi", "lilongwe", "blantyre"],
    "Mozambique": ["mozambique", "maputo"],
    "Namibia": ["namibia", "windhoek"],
    "South Africa": ["south africa", "pretoria", "johannesburg", "cape town", "durban"],
    "Zambia": ["zambia", "lusaka"],
    "Zimbabwe": ["zimbabwe", "harare", "bulawayo"],
})

# ---------------------------
# Regions for AU Member States
# ---------------------------
COUNTRY_TO_REGION = {
    # North Africa (6)
    "Algeria": "North Africa",
    "Egypt": "North Africa",
    "Libya": "North Africa",
    "Mauritania": "North Africa",
    "Morocco": "North Africa",
    "Tunisia": "North Africa",
    # Western Sahara / SADR
    "Sahrawi Arab Democratic Republic (SADR)": "North Africa",

    # West Africa (15)
    "Benin": "West Africa",
    "Burkina Faso": "West Africa",
    "Cabo Verde": "West Africa",
    "Côte d’Ivoire": "West Africa",
    "Gambia": "West Africa",
    "Ghana": "West Africa",
    "Guinea": "West Africa",
    "Guinea-Bissau": "West Africa",
    "Liberia": "West Africa",
    "Mali": "West Africa",
    "Niger": "West Africa",
    "Nigeria": "West Africa",
    "Senegal": "West Africa",
    "Sierra Leone": "West Africa",
    "Togo": "West Africa",

    # Central Africa (9)
    "Burundi": "Central Africa",
    "Cameroon": "Central Africa",
    "Central African Republic": "Central Africa",
    "Chad": "Central Africa",
    "Congo (Republic)": "Central Africa",
    "Democratic Republic of Congo (DRC)": "Central Africa",
    "Equatorial Guinea": "Central Africa",
    "Gabon": "Central Africa",
    "São Tomé and Príncipe": "Central Africa",

    # East Africa (14)
    "Comoros": "East Africa",
    "Djibouti": "East Africa",
    "Eritrea": "East Africa",
    "Ethiopia": "East Africa",
    "Kenya": "East Africa",
    "Madagascar": "East Africa",
    "Mauritius": "East Africa",
    "Rwanda": "East Africa",
    "Seychelles": "East Africa",
    "Somalia": "East Africa",
    "South Sudan": "East Africa",
    "Sudan": "East Africa",
    "Tanzania": "East Africa",
    "Uganda": "East Africa",

    # Southern Africa (11)
    "Angola": "Southern Africa",
    "Botswana": "Southern Africa",
    "Eswatini": "Southern Africa",
    "Lesotho": "Southern Africa",
    "Malawi": "Southern Africa",
    "Mozambique": "Southern Africa",
    "Namibia": "Southern Africa",
    "South Africa": "Southern Africa",
    "Zambia": "Southern Africa",
    "Zimbabwe": "Southern Africa",
}


def detect_topic(text:str) -> str:
    t = text.lower()
    best_topic, best_hits = "General", 0
    for topic, keys in TOPIC_KEYWORDS.items():
        hits = sum(t.count(k) for k in keys)
        if hits > best_hits:
            best_topic, best_hits = topic, hits
    return best_topic

def topic_to_region(topic: str) -> str:
    return COUNTRY_TO_REGION.get(topic, "Non-country/Other")

# ---------- Main pipeline ----------
def run(manifest_path:str, limit:int=None):
    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit).copy()

    texts, types, topics, sent_compound, sent_label = [], [], [], [], []
    sia = SentimentIntensityAnalyzer()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyze PDFs"):
        path = row["path"]
        try:
            text = pdf_to_text(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            text = ""

        doc_type = classify_type(row.get("title",""), row.get("filename",""), text)
        topic = detect_topic(text)
        scores = sentiment_scores(sia, text)
        compound = scores["compound"]
        label = label_from_compound(compound)

        texts.append(text)
        types.append(doc_type)
        topics.append(topic)
        sent_compound.append(compound)
        sent_label.append(label)

    df["text"] = texts
    df["doc_type"] = types
    df["topic"] = topics
    df["sent_compound"] = sent_compound
    df["sent_label"] = sent_label

    # Region from topic (if the topic is a country)
    df["region"] = df["topic"].apply(topic_to_region)

    ensure_dir("data")
    out_csv = "data/analysis.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DONE] Analysis saved to {out_csv}")

    # ----- Charts -----
    ensure_dir(os.path.join("output", "charts"))

    type_counts = df["doc_type"].value_counts()
    bar_save(type_counts, "Documents by Type", "Type", "Count",
             os.path.join("output", "charts", "doc_types.png"))

    label_counts = df["sent_label"].value_counts()
    bar_save(label_counts, "Sentiment Labels", "Label", "Count",
             os.path.join("output", "charts", "sentiment_labels.png"))

    hist_save(df["sent_compound"], bins=21,
              title="Sentiment (VADER Compound)", xlabel="Compound Score",
              out_path=os.path.join("output", "charts", "sentiment_hist.png"))

    # Topics (top 20)
    topic_counts = df["topic"].value_counts().head(20)
    bar_save(topic_counts, "Top 20 Topics", "Topic", "Count",
             os.path.join("output", "charts", "topics_top20.png"))

    # Regions
    region_counts = df["region"].value_counts()
    bar_save(region_counts, "Documents by Region", "Region", "Count",
             os.path.join("output", "charts", "regions.png"))

    # Region × Sentiment (stacked)
    pivot_rs = pd.crosstab(df["region"], df["sent_label"]).reindex(
        ["North Africa", "West Africa", "Central Africa", "East Africa", "Southern Africa", "Non-country/Other"],
        fill_value=0
    )
    bar_save(pivot_rs, "Documents by Region and Sentiment", "Region", "Count",
             os.path.join("output", "charts", "regions_sentiment_stacked.png"),
             stacked=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/psc_2025_en/psc_communiques_2025_manifest.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.manifest, args.limit)
