"""SmartJob Morocco — moteur de recherche d'offres d'emploi (BM25 + embeddings + RRF) et assistant RAG.

Lancement :  streamlit run app.py
Artefacts attendus dans ./smartjob_final_artifacts : offers_clean.parquet, chunks.parquet, config.json
et (optionnel, pour la recherche sémantique) embeddings.npy.
"""
from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SmartJob Morocco", page_icon="💼", layout="wide",
                   initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("SMARTJOB_ARTIFACT_DIR", BASE_DIR / "smartjob_final_artifacts"))
OFFERS_FILE = ARTIFACT_DIR / "offers_clean.parquet"
CHUNKS_FILE = ARTIFACT_DIR / "chunks.parquet"
EMBEDDINGS_FILE = ARTIFACT_DIR / "embeddings.npy"
CONFIG_FILE = ARTIFACT_DIR / "config.json"

DATA_ROLES = ["Data Scientist", "Data Engineer", "Data Analyst", "BI / Décisionnel", "Business Analyst"]
EXAMPLES = [
    ("Data Engineer + Spark", "Data Engineer Spark à Casablanca"),
    ("Stage Power BI", "stage Power BI"),
    ("Analyste débutant", "analyste de données pour débutant"),
    ("Combien d'offres Data ?", "Combien d'offres Data Analyst ?"),
]

st.markdown("""
<style>
.block-container{max-width:1300px;padding-top:1.6rem;padding-bottom:3rem}
header[data-testid="stHeader"]{background:transparent}
h1,h2,h3{letter-spacing:-.01em}
.hero{padding:.2rem 0 1rem;margin-bottom:.4rem;border-bottom:1px solid #D7E0E9}
.hero .eyebrow{font-size:.75rem;font-weight:700;color:#3560A8;text-transform:uppercase;letter-spacing:.09em}
.hero h1{margin:.1rem 0 .2rem;font-size:1.9rem;color:#174F7A}
.hero p{margin:0;color:#687481;font-size:.95rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #D7E0E9;border-radius:8px;padding:.7rem 1rem}
[data-testid="stMetricValue"]{color:#205B50;font-weight:750}
.card{background:#fff;border:1px solid #D7E0E9;border-left:4px solid #3560A8;border-radius:8px;
      padding:1rem 1.2rem;margin:.6rem 0}
.card .rank{color:#3560A8;font-weight:750;margin-right:.35rem}
.card .title{font-size:1.05rem;font-weight:700;color:#174F7A}
.card .meta{color:#687481;font-size:.88rem;margin:.15rem 0 .5rem}
.card .excerpt{color:#252B33;font-size:.92rem;line-height:1.45;margin-top:.5rem}
.card .link{margin-top:.55rem;font-size:.88rem}
.card .link a{color:#3560A8;font-weight:650;text-decoration:none}
.pill{display:inline-block;padding:.12rem .5rem;margin:.1rem .25rem .1rem 0;border:1px solid #D5E0EB;
      border-radius:999px;background:#F1F5F9;color:#315477;font-size:.76rem}
.pill.tech{background:#E8F3EF;border-color:#C9E2D9;color:#205B50}
.pill.out{background:#FFF4E5;border-color:#F3D9AE;color:#8A5A12}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def fold(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", fold(value))


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


@st.cache_resource(show_spinner="Chargement de l'index de recherche…")
def load_data():
    missing = [p.name for p in (OFFERS_FILE, CHUNKS_FILE, CONFIG_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Artefacts manquants : " + ", ".join(missing) + ". Exécutez le notebook Kaggle, "
            "téléchargez le dossier « smartjob_final_artifacts » et placez-le à côté de app.py."
        )
    offers = pd.read_parquet(OFFERS_FILE)
    chunks = pd.read_parquet(CHUNKS_FILE)
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    embeddings = None
    if EMBEDDINGS_FILE.exists():
        embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
        if len(embeddings) != len(chunks):
            raise ValueError(f"Incohérence : {len(chunks)} chunks pour {len(embeddings)} embeddings.")
    offers["offer_id"] = offers["offer_id"].astype(str)
    chunks["offer_id"] = chunks["offer_id"].astype(str)
    offers["techs"] = offers["techs"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (list(x) if x is not None else []))
    bm25 = BM25Okapi([tokenize(t) for t in chunks["text"].fillna("")])
    return offers, chunks, embeddings, config, bm25


@st.cache_resource(show_spinner="Chargement du moteur sémantique…")
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource(show_spinner="Chargement de Qwen (peut être long sur CPU)…")
def load_qwen(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to("cpu").eval()
    return tok, model


# ---------------------------------------------------------------------------
# Filtres déduits de la requête
# ---------------------------------------------------------------------------
ROLE_PATTERNS = {
    "Data Scientist": r"data scientist|machine learning|\bnlp\b|computer vision",
    "Data Engineer": r"data engineer|ingenieur (de )?donnees|big data engineer|data platform",
    "Data Analyst": r"data analyst|analyste (de )?donnees|analyste data",
    "BI / Décisionnel": r"business intelligence|decisionnel|bi analyst|developpeur (ms |power )?bi\b",
    "Business Analyst": r"business analyst|analyste metier|analyste fonctionnel",
}
TECH_PATTERNS = {
    "Python": r"\bpython\b", "SQL": r"\bsql\b", "Spark": r"\bspark\b|pyspark", "Power BI": r"power\s*-?bi",
    "Tableau": r"\btableau\b(?!\s+de\s+bord)", "Excel": r"\bexcel\b", "Airflow": r"\bairflow\b",
    "Azure": r"\bazure\b", "AWS": r"\baws\b", "Docker": r"\bdocker\b", "Kubernetes": r"kubernetes|\bk8s\b",
    "Hadoop": r"\bhadoop\b|hdfs", "Kafka": r"\bkafka\b", "Databricks": r"\bdatabricks\b",
    "PostgreSQL": r"postgres", "MongoDB": r"mongodb|\bmongo\b", "Snowflake": r"snowflake", "Talend": r"talend",
    "SSIS": r"\bssis\b", "dbt": r"\bdbt\b", "Java": r"\bjava\b(?!script)", "Scala": r"\bscala\b",
    "Git": r"\bgit(hub|lab)?\b", "ETL": r"\betl\b", "TensorFlow": r"tensorflow", "PyTorch": r"pytorch",
}


def get_filters(query, offers, city, role, contract, remote, data_only):
    q = fold(query)
    f: dict = {}
    if city != "Toutes":
        f["city"] = city
    else:
        for c in sorted(offers["city"].dropna().astype(str).unique(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(fold(c))}\b", q):
                f["city"] = c
                break
    if role != "Tous":
        f["role"] = role
    else:
        for r, pat in ROLE_PATTERNS.items():
            if re.search(pat, q):
                f["role"] = r
                break
    if contract != "Tous":
        f["contract"] = contract
    elif re.search(r"\bstages?\b|stagiaire", q) and "Stage" in set(offers["contract"].dropna()):
        f["contract"] = "Stage"
    techs = [n for n, p in TECH_PATTERNS.items() if re.search(p, q)]
    if techs:
        f["techs"] = techs
        f["tech_operator"] = "OR" if re.search(r"\bou\b", q) else "AND"
    if re.search(r"debutant|sans experience|premier emploi|junior", q):
        f["junior"] = True
    if remote or re.search(r"teletravail|a distance|remote|hybride", q):
        f["remote"] = True
    if data_only and "role" not in f:
        f["data_only"] = True
    return f


def apply_filters(offers: pd.DataFrame, f: dict) -> pd.DataFrame:
    m = pd.Series(True, index=offers.index)
    if f.get("city"):
        m &= offers["city"].fillna("").map(fold).eq(fold(f["city"]))
    if f.get("role"):
        m &= offers["role"].fillna("").eq(f["role"])
    if f.get("data_only"):
        m &= offers["role"].isin(DATA_ROLES)
    if f.get("contract"):
        m &= offers["contract"].fillna("").map(fold).eq(fold(f["contract"]))
    if f.get("techs"):
        agg = any if f.get("tech_operator") == "OR" else all
        m &= offers["techs"].apply(lambda xs: agg(t in (xs or []) for t in f["techs"]))
    if f.get("junior"):
        m &= offers["experience"].fillna("").str.contains(r"(?i)junior|débutant|debutant|0.?1 an", regex=True)
    if f.get("remote"):
        m &= offers["remote"].fillna("").str.contains(r"(?i)oui|hybride|remote|télétravail|teletravail", regex=True)
    return offers[m]


FILTER_LABELS = {"city": "Ville", "role": "Métier", "contract": "Contrat", "techs": "Technologies",
                 "junior": "Débutant / junior", "remote": "Télétravail", "data_only": "Offres Data/BI"}


def describe_filters(f: dict) -> str:
    parts = []
    for k, label in FILTER_LABELS.items():
        if k in f:
            v = f[k]
            if k == "techs":
                v = (" ou " if f.get("tech_operator") == "OR" else " + ").join(v)
            elif isinstance(v, bool):
                v = "oui"
            parts.append(f"**{label}** : {v}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Recherche hybride
# ---------------------------------------------------------------------------
def rrf(rankings, k=60):
    sc: dict = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            sc[item] = sc.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(sc, key=sc.get, reverse=True)


def retrieve(query, offers, chunks, embeddings, bm25, embedder, f, top_k, mode):
    cand = apply_filters(offers, f) if f else offers
    if cand.empty:
        return [], 0
    allowed = set(cand["offer_id"])
    oids = chunks["offer_id"].to_numpy()
    mask = np.fromiter((o in allowed for o in oids), bool, len(oids))
    offer_rank, chunk_rank = [], []

    def offers_by_best_score(pos, scores):
        d = pd.DataFrame({"oid": oids[pos], "s": scores[pos]})
        return list(d.groupby("oid")["s"].max().sort_values(ascending=False).index)

    if mode in ("BM25", "Hybride"):
        s = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
        pos = np.flatnonzero(mask & (s > 0))
        if len(pos):
            offer_rank.append(offers_by_best_score(pos, s))
            chunk_rank.append([int(i) for i in pos[np.argsort(-s[pos])]])
    if mode in ("Dense", "Hybride") and embedder is not None and embeddings is not None:
        qv = embedder.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        s = embeddings @ qv
        pos = np.flatnonzero(mask)
        offer_rank.append(offers_by_best_score(pos, s))
        chunk_rank.append([int(i) for i in pos[np.argsort(-s[pos])]])
    if not offer_rank:
        return [], len(cand)
    ranked = offer_rank[0] if len(offer_rank) == 1 else rrf(offer_rank)
    ranked_chunks = chunk_rank[0] if len(chunk_rank) == 1 else rrf(chunk_rank)
    per_offer: dict = {}
    for idx in ranked_chunks:
        lst = per_offer.setdefault(oids[idx], [])
        if len(lst) < 3:
            lst.append(idx)
    return [{"offer_id": o, "chunk_ids": per_offer.get(o, [])} for o in ranked[:top_k]], len(cand)


# ---------------------------------------------------------------------------
# Génération (optionnelle)
# ---------------------------------------------------------------------------
def generate_answer(question, hits, offers_by_id, chunks, tok, model, search_query=""):
    """Répond à une question facultative ou résume les annonces retrouvées.

    Les métadonnées structurées servent de référence ; un repli factuel est
    utilisé si Qwen refuse, ne cite aucune source ou produit une réponse vide.
    """
    import torch

    blocks = []
    structured_summaries = []

    for n, h in enumerate(hits, 1):
        o = offers_by_id[h["offer_id"]]
        techs = list(o.techs or [])
        fields = {
            "Poste": str(o.title or "").strip(),
            "Entreprise": str(o.company or "").strip(),
            "Ville": str(o.city or "").strip(),
            "Contrat": str(o.contract or "").strip(),
            "Expérience": str(o.experience or "").strip(),
            "Formation": str(o.education or "").strip(),
            "Technologies détectées": ", ".join(techs),
        }
        meta = [
            f"{key} : {value if value else 'Non précisé'}"
            for key, value in fields.items()
        ]

        body = []
        for i in h["chunk_ids"][:2]:
            row = chunks.iloc[i]
            if str(row.get("section", "")).upper() != "FICHE":
                passage = str(row.get("text", "")).split("\n", 1)[-1].strip()
                if passage:
                    body.append(passage)

        blocks.append(f"SOURCE [{n}] — MÉTADONNÉES STRUCTURÉES\n" +
                      "\n".join(meta) +
                      ("\nExtrait de l'annonce :\n" + "\n".join(body)[:1400] if body else ""))
        structured_summaries.append(fields)

    system = (
        "Tu es l'assistant de SmartJob Morocco. Réponds en français, de façon concise "
        "et uniquement à partir des sources fournies. "
        "RÈGLE PRIORITAIRE : les métadonnées structurées de chaque source sont des "
        "informations disponibles et fiables. Tu dois les utiliser pour présenter "
        "les offres retrouvées, même si le texte de l'extrait ne les répète pas. "
        "Ne dis jamais que l'information est absente lorsqu'elle figure dans les "
        "métadonnées. N'invente et ne déduis aucune valeur manquante. "
        "Pour une recherche d'offres, résume les résultats : poste, entreprise, ville, "
        "contrat, expérience et technologies lorsqu'ils sont renseignés. "
        "Pour chaque annonce, ajoute sa citation exacte [1], [2], etc. "
        "Si un champ précis n'est pas renseigné, indique simplement « non précisé ». "
        "Réponse attendue : 3 à 5 phrases ou puces maximum."
    )
    if question.strip():
        task = f"QUESTION : {question.strip()}"
    else:
        task = (
            f"TÂCHE : résume les annonces retrouvées pour la recherche « {search_query} ». "
            "Présente les postes, entreprises, villes, contrats, expériences et technologies "
            "lorsqu'ils sont renseignés."
        )
    user = (
        "ANNONCES RETROUVÉES :\n\n" + "\n\n".join(blocks) +
        f"\n\n{task}\n\n"
        "Commence par les informations présentes dans les métadonnées. "
        "N'affirme pas qu'aucune information n'est disponible si une annonce est fournie."
    )

    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    ans = re.sub(r",\s*(?:,\s*)+", ",", ans)
    ans = re.sub(r"(\[\d+\])\s*,+\s*(?=$|\n)", r"\1", ans, flags=re.M)

    # Ne conserver que les références correspondant aux annonces effectivement fournies.
    refs = {int(x) for x in re.findall(r"\[(\d+)\]", ans)}
    for x in refs:
        if x < 1 or x > len(hits):
            ans = ans.replace(f"[{x}]", "")

    # Repli fiable : si Qwen refuse malgré des métadonnées disponibles,
    # produire une synthèse factuelle sans inventer de contenu.
    refusal = "n'apparaît pas" in ans.lower() or "n’apparait pas" in ans.lower()
    valid_refs = re.findall(r"\[(\d+)\]", ans)
    if not ans or refusal or not valid_refs:
        lines = []
        for n, fields in enumerate(structured_summaries, 1):
            parts = []
            for key in ("Poste", "Entreprise", "Ville", "Contrat", "Expérience", "Technologies détectées"):
                value = fields[key]
                if value:
                    parts.append(f"{key.lower()} : {value}")
            if parts:
                lines.append("• " + " ; ".join(parts) + f" [{n}]")
        ans = "\n".join(lines) if lines else "Les annonces retrouvées ne contiennent pas assez de métadonnées pour établir une synthèse."

    return ans


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------
def offer_card(o, chunk_ids, chunks, rank) -> str:
    pills = "".join(f'<span class="pill">{esc(v)}</span>' for v in (o.contract, o.experience) if v)
    pills += "".join(f'<span class="pill tech">{esc(t)}</span>' for t in (o.techs or [])[:8])
    text = " ".join(str(chunks.iloc[i]["text"]).split("\n", 1)[-1] for i in chunk_ids[:2]
                    if chunks.iloc[i]["section"] != "FICHE")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 150:                                   # extrait trop court -> début des missions
        text = re.sub(r"\s+", " ", str(getattr(o, "poste_txt", "") or "")).strip() or text
    excerpt = f'<div class="excerpt">{esc(text[:520])}{"…" if len(text) > 520 else ""}</div>' if text else ""
    url = str(o.url or "")
    link = (f'<div class="link"><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">Consulter l’annonce originale ↗</a></div>'
            if url.startswith(("http://", "https://")) else "")
    return (f'<div class="card"><div><span class="rank">{rank}.</span><span class="title">{esc(o.title)}</span></div>'
            f'<div class="meta">{esc(o.company or "Entreprise non précisée")} · {esc(o.city or "Ville non précisée")}</div>'
            f'{pills}{excerpt}{link}</div>')


def hbar(df: pd.DataFrame, label: str, value: str, color: str, height: int = 280):
    chart = (alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=3)
             .encode(x=alt.X(f"{value}:Q", title=None), y=alt.Y(f"{label}:N", sort="-x", title=None),
                     tooltip=[label, value])
             .properties(height=height))
    st.altair_chart(chart)


def analytic_answer(question, offers, f):
    sub = apply_filters(offers, f) if f else offers
    q = fold(question)
    if re.search(r"technolog|competence|outil|stack", q):
        c = Counter(t for xs in sub["techs"] for t in (xs or [])).most_common(10)
        return "**Technologies les plus détectées**\n\n" + "\n".join(f"- {n} : {k} annonces" for n, k in c)
    if re.search(r"\bville", q):
        return "**Répartition par ville**\n\n" + "\n".join(f"- {n} : {k}" for n, k in sub["city"].fillna("Non précisée").value_counts().head(10).items())
    if re.search(r"entreprise|societe", q):
        return "**Entreprises les plus représentées**\n\n" + "\n".join(f"- {n} : {k}" for n, k in sub["company"].fillna("Non précisée").value_counts().head(10).items())
    return f"**{len(sub)}** annonce(s) correspondent aux critères détectés, sur **{len(offers)}** annonces."


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
try:
    offers, chunks, embeddings, config, bm25 = load_data()
except Exception as exc:  # noqa: BLE001
    st.error(str(exc))
    st.info("Structure attendue : `app.py` + dossier `smartjob_final_artifacts/` "
            "(offers_clean.parquet, chunks.parquet, config.json, embeddings.npy optionnel).")
    st.stop()

if "role" not in offers:
    offers["role"] = "Autre / non classé"
offers_by_id = {str(r.offer_id): r for r in offers.itertuples(index=False)}
data_offers = offers[offers["role"].isin(DATA_ROLES)]
semantic_ok = embeddings is not None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtres")
    sel_city = st.selectbox("Ville", ["Toutes"] + sorted(offers["city"].dropna().astype(str).unique()))
    sel_role = st.selectbox("Métier", ["Tous"] + DATA_ROLES)
    sel_contract = st.selectbox("Contrat", ["Tous"] + sorted(offers["contract"].dropna().astype(str).unique()))
    sel_remote = st.checkbox("Télétravail / hybride uniquement")
    sel_data_only = st.checkbox("Offres Data/BI uniquement", help="Restreint la recherche aux métiers de la data.")
    st.divider()
    st.markdown("### Moteur")
    modes = ["Hybride", "BM25", "Dense"] if semantic_ok else ["BM25"]
    mode = st.selectbox("Méthode de recherche", modes)
    if not semantic_ok:
        st.caption("Recherche sémantique désactivée : `embeddings.npy` absent.")
    top_k = st.slider("Nombre de résultats", 3, 15, 8)
    use_qwen = st.checkbox("Synthèse assistée par Qwen", help="Exécution locale, lente sur CPU.")
    st.divider()
    st.caption(f"Source : {config.get('source', 'ReKrute')} · instantané de {len(offers):,} annonces".replace(",", " "))

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero"><div class="eyebrow">Marché de l'emploi · Maroc</div>
<h1>SmartJob Morocco</h1>
<p>Recherche intelligente d'offres (mots-clés + sémantique), analyse du marché et assistant RAG qui cite ses sources.</p></div>
""", unsafe_allow_html=True)

tab_search, tab_market, tab_profile, tab_about = st.tabs(["🔎 Recherche", "📊 Marché", "🎯 Mon profil", "ℹ️ Méthode"])

# ---------------------------------------------------------------------------
# Onglet Recherche
# ---------------------------------------------------------------------------
def _set_example(text: str):
    st.session_state["q_input"] = text
    st.session_state["do_search"] = True


def run_search(query: str):
    f = get_filters(query, offers, sel_city, sel_role, sel_contract, sel_remote, sel_data_only)
    state = {"query": query, "filters": f, "mode": mode, "hits": [], "candidates": 0, "note": None,
             "analytic": None, "answer": None}
    if re.search(r"combien|nombre d[' ]offres|how many", fold(query)):
        state["analytic"] = analytic_answer(query, offers, f)
    else:
        embedder, actual = None, mode
        if mode in ("Hybride", "Dense"):
            try:
                embedder = load_embedder(config.get("embedding_model", "intfloat/multilingual-e5-base"))
            except Exception as exc:  # noqa: BLE001
                actual = "BM25"
                state["note"] = f"Moteur sémantique indisponible ({type(exc).__name__}) : résultats calculés avec BM25."
        state["hits"], state["candidates"] = retrieve(query, offers, chunks, embeddings, bm25, embedder, f, top_k, actual)
        state["mode"] = actual
    st.session_state["last"] = state


with tab_search:
    with st.form("search_form"):
        c1, c2 = st.columns([6, 1])
        c1.text_input("Recherche", key="q_input", label_visibility="collapsed",
                      placeholder="Ex. Data Engineer Spark à Casablanca, stage Power BI, analyste de données débutant…")
        submitted = c2.form_submit_button("Rechercher", type="primary")
    st.caption("Essayez :")
    cols = st.columns(len(EXAMPLES))
    for col, (label, query_text) in zip(cols, EXAMPLES):
        col.button(label, key=f"ex_{label}", on_click=_set_example, args=(query_text,))

    if submitted or st.session_state.pop("do_search", False):
        q = st.session_state.get("q_input", "").strip()
        if q:
            with st.spinner("Recherche en cours…"):
                run_search(q)
        else:
            st.info("Saisissez un poste, une compétence ou une ville.")

    last = st.session_state.get("last")
    if last:
        if last["filters"]:
            st.markdown("Filtres détectés : " + describe_filters(last["filters"]))
        if last["note"]:
            st.warning(last["note"])
        if last["analytic"]:
            st.info(last["analytic"])
        elif not last["hits"]:
            st.info("Aucune offre ne correspond à tous les critères. Retirez un filtre ou reformulez la recherche.")
        else:
            st.caption(f"{last['candidates']} offre(s) après filtres · {len(last['hits'])} résultat(s) · méthode : {last['mode']}")
            if use_qwen:
                with st.expander("Synthèse Qwen", expanded=bool(last.get("answer"))):
                    user_q = st.text_input(
                        "Question sur ces offres (facultatif)",
                        key="qwen_q",
                        placeholder="Ex. Quel diplôme est demandé ? — laissez vide pour un résumé des offres",
                    )
                    if st.button("Générer la synthèse"):
                        try:
                            tok, model = load_qwen(config.get("llm_model") or "Qwen/Qwen2.5-3B-Instruct")
                            with st.spinner("Génération en cours — cela peut prendre quelques minutes sur CPU…"):
                                last["answer"] = generate_answer(
                                    user_q, last["hits"], offers_by_id, chunks, tok, model,
                                    search_query=last["query"],
                                )
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Qwen indisponible : {exc}")
                    if last.get("answer"):
                        st.markdown(last["answer"])
                        st.caption("Réponse générée par un modèle de 3 milliards de paramètres : vérifiez-la dans les annonces.")
            rows = []
            for rank, h in enumerate(last["hits"], 1):
                o = offers_by_id[h["offer_id"]]
                st.markdown(offer_card(o, h["chunk_ids"], chunks, rank), unsafe_allow_html=True)
                rows.append({"rang": rank, "poste": o.title, "entreprise": o.company, "ville": o.city, "contrat": o.contract,
                             "experience": o.experience, "technologies": ", ".join(o.techs or []), "url": o.url})
            st.download_button("⬇️ Exporter les résultats (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                               "smartjob_resultats.csv", "text/csv")

# ---------------------------------------------------------------------------
# Onglet Marché
# ---------------------------------------------------------------------------
with tab_market:
    k = st.columns(4)
    k[0].metric("Offres collectées", f"{len(offers):,}".replace(",", " "))
    k[1].metric("Offres Data / BI", len(data_offers))
    k[2].metric("Entreprises", f"{offers['company'].nunique():,}".replace(",", " "))
    k[3].metric("Villes", offers["city"].nunique())
    if data_offers.empty:
        st.info("Aucune offre Data/BI détectée dans ce jeu de données.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("**Offres Data/BI par métier**")
            d = data_offers["role"].value_counts().rename_axis("métier").reset_index(name="offres")
            hbar(d, "métier", "offres", "#3560A8")
        with b:
            st.markdown("**Technologies les plus demandées (offres Data/BI)**")
            d = pd.DataFrame(Counter(t for xs in data_offers["techs"] for t in xs).most_common(12), columns=["technologie", "offres"])
            hbar(d, "technologie", "offres", "#205B50", height=360)
        c, e = st.columns(2)
        with c:
            st.markdown("**Villes (offres Data/BI)**")
            d = data_offers["city"].fillna("Non précisée").value_counts().head(8).rename_axis("ville").reset_index(name="offres")
            hbar(d, "ville", "offres", "#F28E2B")
        with e:
            st.markdown("**Niveau d'expérience (offres Data/BI)**")
            d = data_offers["experience"].fillna("Non précisé").value_counts().rename_axis("expérience").reset_index(name="offres")
            hbar(d, "expérience", "offres", "#B07AA1")
    st.caption("Rôles et technologies détectés par règles explicables : de possibles erreurs subsistent.")

# ---------------------------------------------------------------------------
# Onglet Mon profil
# ---------------------------------------------------------------------------
with tab_profile:
    st.markdown("Sélectionnez vos compétences : l'application classe les offres selon les technologies que vous maîtrisez déjà et indique celles qui manquent.")
    all_techs = sorted({t for xs in offers["techs"] for t in xs})
    defaults = [t for t in ("Python", "SQL", "Power BI") if t in all_techs]
    mine = st.multiselect("Mes compétences", all_techs, default=defaults)
    p1, p2 = st.columns(2)
    scope_data = p1.checkbox("Offres Data/BI uniquement", value=True, key="scope_profile")
    p_city = p2.selectbox("Ville", ["Toutes"] + sorted(offers["city"].dropna().astype(str).unique()), key="city_profile")
    pool = data_offers if scope_data else offers
    if p_city != "Toutes":
        pool = pool[pool["city"] == p_city]
    rows = []
    for o in pool.itertuples(index=False):
        need = set(o.techs or [])
        got = need & set(mine)
        if got:
            rows.append({"score": round(len(got) + len(got) / len(need), 2), "poste": o.title, "entreprise": o.company or "—",
                         "ville": o.city or "—", "contrat": o.contract or "—", "compétences couvertes": ", ".join(sorted(got)),
                         "à acquérir": ", ".join(sorted(need - set(mine))[:6]) or "—", "annonce": o.url})
    if not mine:
        st.info("Choisissez au moins une compétence.")
    elif not rows:
        st.info("Aucune offre ne demande ces compétences dans ce périmètre.")
    else:
        df = pd.DataFrame(rows).sort_values("score", ascending=False).head(15)
        st.dataframe(df, hide_index=True,
                     column_config={"annonce": st.column_config.LinkColumn("annonce", display_text="Ouvrir ↗"),
                                    "score": st.column_config.NumberColumn(format="%.2f")})
        gap = Counter(t for xs in pool["techs"] for t in xs if t not in set(mine)).most_common(8)
        if gap:
            st.markdown("**Compétences les plus demandées que vous n'avez pas encore :** " +
                        " · ".join(f"{t} ({n})" for t, n in gap))

# ---------------------------------------------------------------------------
# Onglet Méthode
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**SmartJob Morocco** est un prototype construit à partir d'un instantané d'annonces collectées sur ReKrute.

**Pipeline** : collecte → nettoyage / extraction des sections → chunking → index BM25 + index vectoriel → fusion RRF → filtres structurés → assistant RAG (optionnel).

- **BM25** : recherche lexicale, très efficace sur les termes techniques (Spark, SSIS, Power BI…).
- **Embeddings multilingues** (`multilingual-e5-base`) : recherche sémantique pour les formulations libres.
- **RRF** : fusion des classements lexical et sémantique, au niveau des annonces.
- **Filtres structurés** : ville, métier, contrat, technologies, télétravail, déduits de la requête ou choisis dans la barre latérale.
- **Statistiques** : calculées avec pandas, jamais par le LLM.
- **Assistant RAG** (Qwen2.5-3B) : répond uniquement à partir des annonces retrouvées et cite ses sources.

**Limites** : corpus limité à un site et à un instantané ; rôles et technologies détectés par règles ; un modèle de 3 milliards de paramètres peut mal interpréter une annonce — vérifiez toujours la source.
""")