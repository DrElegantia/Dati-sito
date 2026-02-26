#!/usr/bin/env python3
"""
Sanremo — Chi si esibisce a metà serata ha più possibilità di vincere?

Analisi statistica + ML con:
  1) Spearman rank correlation (globale e per-anno)
  2) Kruskal-Wallis H test (5 fasce, per-serata)
  3) Chi² test (proporzione top finishers per fascia)
  4) Mann-Whitney U test (centro vs estremi)
  5) Friedman test (variabilità tra anni, anno come blocco)
  6) Spearman per-serata
  7) Kruskal-Wallis per-serata + Fisher combined
  8) ML: Random Forest + Logistic Regression (la fascia predice il piazzamento?)

Divisione primaria in 5 fasce per-serata (fasce bilanciate).
Tutti i risultati normalizzati per numero partecipanti.
Output: docs/sanremo_timing_analysis.json

Uso:
    python3 scripts/analisi_timing_sanremo.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    """Encoder per tipi numpy → tipi Python nativi."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dati_sremo")
INPUT_CSV = os.path.join(DATA_DIR, "sanremo_ordine_serate.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "docs", "sanremo_timing_analysis.json")

FASCE_5_LABELS = ["Apertura", "Seconda fascia", "Centro", "Quarta fascia", "Chiusura"]
FASCE_5_BINS = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]


# ============================================================
# Caricamento e preparazione dati
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, dtype={"classifica_finale": str, "classifica_serata": str})
    df["classifica_finale"] = pd.to_numeric(df["classifica_finale"], errors="coerce")
    df = df[df["classifica_finale"].notna()].copy()
    df["classifica_finale"] = df["classifica_finale"].astype(int)

    # Solo serate competitive 1-4 (nella finale l'ordine È la classifica)
    serate_competitive = [
        "Prima serata", "Seconda serata", "Terza serata", "Quarta serata",
    ]
    df = df[df["serata_name"].isin(serate_competitive)].copy()
    return df


def enrich_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge rank normalizzato e fasce a 5 livelli ai dati per-serata."""
    df = df.copy()
    n_per_year = df.groupby("year")["classifica_finale"].max().to_dict()
    df["n_partecipanti"] = df["year"].map(n_per_year)
    df["rank_norm"] = (df["classifica_finale"] - 1) / (df["n_partecipanti"] - 1).clip(lower=1)

    df["fascia_5"] = pd.cut(
        df["posizione_relativa"], bins=FASCE_5_BINS, labels=FASCE_5_LABELS,
    )
    df["fascia_3"] = pd.cut(
        df["posizione_relativa"],
        bins=[-0.001, 1/3, 2/3, 1.001],
        labels=["Inizio", "Centro", "Fine"],
    )
    df["is_top5"] = df["classifica_finale"] <= 5
    return df


def prepare_artist_data(df: pd.DataFrame) -> pd.DataFrame:
    """Per ogni artista/anno: posizione relativa media nelle serate 1-4."""
    n_per_year = df.groupby("year")["classifica_finale"].max().to_dict()

    artist_data = (
        df.groupby(["year", "artist"])
        .agg(
            pos_rel_media=("posizione_relativa", "mean"),
            n_serate=("serata", "nunique"),
            classifica_finale=("classifica_finale", "first"),
        )
        .reset_index()
    )
    artist_data["n_partecipanti"] = artist_data["year"].map(n_per_year)
    artist_data["rank_normalizzato"] = (
        (artist_data["classifica_finale"] - 1) /
        (artist_data["n_partecipanti"] - 1).clip(lower=1)
    )
    artist_data["fascia_5"] = pd.cut(
        artist_data["pos_rel_media"], bins=FASCE_5_BINS, labels=FASCE_5_LABELS,
    )
    artist_data["fascia_3"] = pd.cut(
        artist_data["pos_rel_media"],
        bins=[-0.001, 1/3, 2/3, 1.001],
        labels=["Inizio", "Centro", "Fine"],
    )
    return artist_data


# ============================================================
# Utilità
# ============================================================

def _sf(x):
    """Safe float per JSON."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 6)


def _interp(p, alpha=0.05):
    if p is None:
        return "non calcolabile"
    if p < 0.001:
        return "altamente significativo (p < 0.001)"
    if p < 0.01:
        return "molto significativo (p < 0.01)"
    if p < alpha:
        return "significativo (p < 0.05)"
    return "non significativo (p >= 0.05)"


def _desc_stats(series):
    return {
        "n": int(len(series)),
        "media": _sf(series.mean()),
        "mediana": _sf(series.median()),
        "dev_std": _sf(series.std()),
        "q25": _sf(series.quantile(0.25)),
        "q75": _sf(series.quantile(0.75)),
    }


# ============================================================
# 1. Spearman
# ============================================================

def test_spearman(raw: pd.DataFrame, artist: pd.DataFrame) -> dict:
    # Globale su dati per-serata
    rho_raw, p_raw = stats.spearmanr(raw["posizione_relativa"], raw["rank_norm"])

    # Globale su dati aggregati per artista
    rho_agg, p_agg = stats.spearmanr(artist["pos_rel_media"], artist["rank_normalizzato"])

    # Per anno (per-serata)
    per_anno = []
    for year in sorted(raw["year"].unique()):
        ydf = raw[raw["year"] == year]
        if len(ydf) < 5:
            continue
        r, pv = stats.spearmanr(ydf["posizione_relativa"], ydf["rank_norm"])
        per_anno.append({
            "anno": int(year), "n": int(len(ydf)),
            "rho": _sf(r), "p_value": _sf(pv),
            "significativo": bool(pv < 0.05) if pv is not None else False,
            "interpretazione": _interp(pv),
        })

    return {
        "test": "Spearman Rank Correlation",
        "descrizione": (
            "Correlazione tra posizione relativa e ranking finale normalizzato. "
            "rho > 0 = chi si esibisce tardi tende a piazzarsi peggio. "
            "Calcolato sia per-serata (367 osservazioni) che aggregato per artista (137)."
        ),
        "ipotesi_nulla": "Non c'è correlazione monotona tra ordine di esibizione e classifica finale",
        "per_serata": {
            "n": int(len(raw)),
            "rho": _sf(rho_raw), "p_value": _sf(p_raw),
            "significativo": bool(p_raw < 0.05),
            "interpretazione": _interp(p_raw),
        },
        "aggregato_artista": {
            "n": int(len(artist)),
            "rho": _sf(rho_agg), "p_value": _sf(p_agg),
            "significativo": bool(p_agg < 0.05),
            "interpretazione": _interp(p_agg),
        },
        "per_anno": per_anno,
    }


# ============================================================
# 2. Kruskal-Wallis (5 fasce, per-serata)
# ============================================================

def test_kruskal_wallis(raw: pd.DataFrame) -> dict:
    """KW sulle 5 fasce, applicato ai dati per-serata (fasce bilanciate)."""
    groups = []
    desc = []
    for fascia in FASCE_5_LABELS:
        g = raw[raw["fascia_5"] == fascia]["rank_norm"]
        if len(g) > 0:
            groups.append(g.values)
            desc.append({"fascia": fascia, **_desc_stats(g)})

    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        H, p = stats.kruskal(*groups)
        N = sum(len(g) for g in groups)
        k = len(groups)
        eta_sq = (H - k + 1) / (N - k) if N > k else None
    else:
        H, p, eta_sq = None, None, None

    return {
        "test": "Kruskal-Wallis H Test (5 fasce, per-serata)",
        "descrizione": (
            "Test non parametrico sui ranghi: le 5 fasce di esibizione producono "
            "distribuzioni diverse del ranking normalizzato? Calcolato sui dati "
            "per-serata (non mediati) dove le fasce sono bilanciate."
        ),
        "ipotesi_nulla": "Le distribuzioni del ranking sono identiche tra le 5 fasce",
        "statistiche_descrittive": desc,
        "H_statistic": _sf(H),
        "p_value": _sf(p),
        "eta_squared": _sf(eta_sq),
        "significativo": bool(p < 0.05) if p is not None else False,
        "interpretazione": _interp(p),
    }


# ============================================================
# 3. Chi² (5 fasce → Top 5)
# ============================================================

def test_chi_squared(raw: pd.DataFrame) -> dict:
    """Chi² sulla tabella di contingenza: fascia × Top 5."""
    contingency = []
    for fascia in FASCE_5_LABELS:
        g = raw[raw["fascia_5"] == fascia]
        if len(g) > 0:
            nt = int(g["is_top5"].sum())
            nn = int((~g["is_top5"]).sum())
            contingency.append({
                "fascia": fascia,
                "top_5": nt, "non_top_5": nn,
                "totale": nt + nn,
                "pct_top_5": _sf(nt / (nt + nn) * 100),
            })

    if len(contingency) >= 2:
        ct = np.array([[d["top_5"], d["non_top_5"]] for d in contingency])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n_total = ct.sum()
        min_dim = min(ct.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_total * max(min_dim, 1)))
    else:
        chi2, p, dof, cramers_v = None, None, None, None

    return {
        "test": "Chi-Squared Test (5 fasce, per-serata)",
        "descrizione": (
            "Test chi² sulla tabella di contingenza: fascia × arrivo in Top 5. "
            "Verifica se la probabilità di piazzarsi in Top 5 dipende dalla fascia."
        ),
        "ipotesi_nulla": "La proporzione di Top 5 è uguale tra le 5 fasce",
        "contingenza": contingency,
        "chi2": _sf(chi2),
        "p_value": _sf(p),
        "gradi_liberta": int(dof) if dof is not None else None,
        "cramers_v": _sf(cramers_v),
        "significativo": bool(p < 0.05) if p is not None else False,
        "interpretazione": _interp(p),
    }


# ============================================================
# 4. Mann-Whitney U (centro vs estremi, 5 fasce)
# ============================================================

def test_mann_whitney(raw: pd.DataFrame) -> dict:
    """Confronta le 3 fasce centrali (2-3-4) vs le 2 estreme (1+5)."""
    centro = raw[raw["fascia_5"].isin(["Seconda fascia", "Centro", "Quarta fascia"])]["rank_norm"]
    estremi = raw[raw["fascia_5"].isin(["Apertura", "Chiusura"])]["rank_norm"]

    if len(centro) >= 3 and len(estremi) >= 3:
        U, p = stats.mannwhitneyu(centro, estremi, alternative="two-sided")
        n1, n2 = len(centro), len(estremi)
        r_effect = 1 - (2 * U) / (n1 * n2)
    else:
        U, p, r_effect = None, None, None

    if len(centro) > 0 and len(estremi) > 0:
        if centro.mean() < estremi.mean():
            direzione = "Le fasce centrali (2-3-4) hanno ranking migliore delle estreme (1+5)"
        else:
            direzione = "Le fasce centrali (2-3-4) NON hanno ranking migliore delle estreme (1+5)"
    else:
        direzione = "Dati insufficienti"

    return {
        "test": "Mann-Whitney U Test (centro 3 fasce vs estremi 2 fasce)",
        "descrizione": (
            "Confronto non parametrico: le 3 fasce centrali (Seconda/Centro/Quarta) "
            "vs le 2 estreme (Apertura/Chiusura). Se il centro è vantaggioso, "
            "dovrebbe avere ranking normalizzato più basso."
        ),
        "ipotesi_nulla": "Le distribuzioni dei ranking sono identiche tra centro ed estremi",
        "centro": {"fasce": ["Seconda fascia", "Centro", "Quarta fascia"], **_desc_stats(centro)},
        "estremi": {"fasce": ["Apertura", "Chiusura"], **_desc_stats(estremi)},
        "U_statistic": _sf(U),
        "p_value": _sf(p),
        "rank_biserial_r": _sf(r_effect),
        "significativo": bool(p < 0.05) if p is not None else False,
        "direzione": direzione,
        "interpretazione": _interp(p),
    }


# ============================================================
# 5. Friedman (5 fasce, anno come blocco)
# ============================================================

def test_friedman(raw: pd.DataFrame) -> dict:
    """Friedman test: anno come blocco, 5 fasce come trattamenti."""
    years = sorted(raw["year"].unique())
    matrix = []
    years_used = []
    detail = []

    for year in years:
        ydf = raw[raw["year"] == year]
        row = []
        complete = True
        anno_detail = {"anno": int(year), "fasce": {}}
        for fascia in FASCE_5_LABELS:
            g = ydf[ydf["fascia_5"] == fascia]["rank_norm"]
            if len(g) == 0:
                complete = False
            else:
                row.append(g.median())
            anno_detail["fasce"][fascia] = {
                "n": int(len(g)),
                "mediana": _sf(g.median()) if len(g) > 0 else None,
                "media": _sf(g.mean()) if len(g) > 0 else None,
            }
        detail.append(anno_detail)
        if complete:
            matrix.append(row)
            years_used.append(int(year))

    if len(matrix) >= 3:
        m = np.array(matrix)
        stat, p = stats.friedmanchisquare(*[m[:, i] for i in range(m.shape[1])])
        W = stat / (len(matrix) * (m.shape[1] - 1))
    else:
        stat, p, W = None, None, None

    return {
        "test": "Friedman Test (5 fasce, anno come blocco)",
        "descrizione": (
            "Test non parametrico per misure ripetute. Ogni anno è un blocco, "
            "le 5 fasce sono i trattamenti. Kendall's W misura la concordanza "
            "tra anni (0 = nessun accordo, 1 = accordo perfetto)."
        ),
        "ipotesi_nulla": "Non c'è differenza sistematica tra le fasce al netto della variabilità tra anni",
        "n_anni_usati": len(years_used),
        "anni_usati": years_used,
        "friedman_chi2": _sf(stat),
        "p_value": _sf(p),
        "kendall_w": _sf(W),
        "significativo": bool(p < 0.05) if p is not None else False,
        "interpretazione": _interp(p),
        "dettaglio_per_anno": detail,
    }


# ============================================================
# 6. Spearman per-serata
# ============================================================

def test_spearman_per_serata(raw: pd.DataFrame) -> dict:
    per_serata = []
    for year in sorted(raw["year"].unique()):
        for serata in sorted(raw[raw["year"] == year]["serata"].unique()):
            sdf = raw[(raw["year"] == year) & (raw["serata"] == serata)]
            if len(sdf) < 5:
                continue
            rho, p = stats.spearmanr(sdf["posizione_relativa"], sdf["rank_norm"])
            per_serata.append({
                "anno": int(year), "serata": int(serata),
                "serata_name": sdf["serata_name"].iloc[0],
                "n": int(len(sdf)),
                "rho": _sf(rho), "p_value": _sf(p),
                "significativo": bool(p < 0.05) if p is not None else False,
                "interpretazione": _interp(p),
            })

    n_sig = sum(1 for s in per_serata if s["significativo"])
    return {
        "test": "Spearman per-serata",
        "descrizione": "Correlazione di Spearman calcolata per ogni singola serata.",
        "n_serate_analizzate": len(per_serata),
        "n_serate_significative": n_sig,
        "dettaglio": per_serata,
    }


# ============================================================
# 7. Kruskal-Wallis per-serata + Fisher combined
# ============================================================

def test_kruskal_per_serata(raw: pd.DataFrame) -> dict:
    """KW 5 fasce per ogni serata, poi Fisher combined."""
    results = []
    for year in sorted(raw["year"].unique()):
        for serata in sorted(raw[raw["year"] == year]["serata"].unique()):
            sdf = raw[(raw["year"] == year) & (raw["serata"] == serata)]
            if len(sdf) < 6:
                continue

            groups = []
            fasce_stats = {}
            for fascia in FASCE_5_LABELS:
                g = sdf[sdf["fascia_5"] == fascia]["rank_norm"]
                if len(g) >= 2:
                    groups.append(g.values)
                fasce_stats[fascia] = {"n": int(len(g)), "media": _sf(g.mean()) if len(g) > 0 else None}

            if len(groups) >= 2:
                H, pv = stats.kruskal(*groups)
            else:
                H, pv = None, None

            results.append({
                "anno": int(year), "serata": int(serata), "n": int(len(sdf)),
                "H": _sf(H), "p_value": _sf(pv),
                "significativo": bool(pv < 0.05) if pv is not None else False,
                "fasce": fasce_stats,
            })

    valid_p = [r["p_value"] for r in results if r["p_value"] is not None and r["p_value"] > 0]
    if len(valid_p) >= 2:
        fisher_stat = -2 * sum(np.log(p) for p in valid_p)
        fisher_df = 2 * len(valid_p)
        fisher_p = 1 - stats.chi2.cdf(fisher_stat, fisher_df)
    else:
        fisher_stat, fisher_df, fisher_p = None, None, None

    return {
        "test": "Kruskal-Wallis per-serata (5 fasce) + Fisher combined",
        "descrizione": (
            "KW con 5 fasce calcolato per ogni serata singolarmente (fasce bilanciate). "
            "I p-value vengono combinati con il metodo di Fisher per un test globale."
        ),
        "ipotesi_nulla": "In nessuna serata la fascia influenza la classifica",
        "n_serate": len(results),
        "n_significative": sum(1 for r in results if r["significativo"]),
        "fisher_combined": {
            "chi2": _sf(fisher_stat), "df": fisher_df, "p_value": _sf(fisher_p),
            "significativo": bool(fisher_p < 0.05) if fisher_p is not None else False,
            "interpretazione": _interp(fisher_p),
        },
        "dettaglio": results,
    }


# ============================================================
# 8. ML: la fascia predice il piazzamento?
# ============================================================

def test_ml(raw: pd.DataFrame) -> dict:
    """
    Machine Learning: la fascia di esibizione predice il piazzamento in Top 5?

    - Features: fascia_5 (one-hot), posizione_relativa, serata
    - Target: is_top5 (classificazione binaria)
    - Validazione: Leave-One-Year-Out (LOYO) cross-validation
    - Baseline: frequenza della classe maggioritaria
    - Modelli: Logistic Regression, Random Forest, Gradient Boosting
    """
    df = raw.reset_index(drop=True).copy()

    # Features
    fascia_dummies = pd.get_dummies(df["fascia_5"].astype(str), prefix="fascia", dtype=float)
    X = pd.concat([
        fascia_dummies.reset_index(drop=True),
        df[["posizione_relativa", "serata"]].reset_index(drop=True),
    ], axis=1)
    y = df["is_top5"].astype(int).values
    groups = df["year"].values

    baseline_acc = max(y.mean(), 1 - y.mean())
    baseline_label = "non_top5" if (1 - y.mean()) > y.mean() else "top5"

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    logo = LeaveOneGroupOut()
    model_results = []

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=logo, groups=groups)
        try:
            y_proba = cross_val_predict(model, X, y, cv=logo, groups=groups, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, y_proba)
        except Exception:
            auc = None

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)

        # Feature importance (fit su tutto per capire quali feature contano)
        model.fit(X, y)
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, "coef_"):
            imp = dict(zip(X.columns, np.abs(model.coef_[0])))
        else:
            imp = {}

        # Ordina per importanza
        imp_sorted = sorted(imp.items(), key=lambda x: -x[1])

        model_results.append({
            "modello": name,
            "accuracy": _sf(acc),
            "f1_score": _sf(f1),
            "auc_roc": _sf(auc),
            "miglioramento_su_baseline": _sf((acc - baseline_acc) / baseline_acc * 100),
            "feature_importance": [
                {"feature": k, "importanza": _sf(v)} for k, v in imp_sorted
            ],
        })

    # Analisi per fascia: probabilità predetta media di Top 5
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X, y)
    df["prob_top5"] = best_model.predict_proba(X)[:, 1]

    prob_per_fascia = []
    for fascia in FASCE_5_LABELS:
        g = df[df["fascia_5"] == fascia]
        if len(g) > 0:
            prob_per_fascia.append({
                "fascia": fascia,
                "prob_media_top5": _sf(g["prob_top5"].mean()),
                "prob_mediana_top5": _sf(g["prob_top5"].median()),
                "pct_effettiva_top5": _sf(g["is_top5"].mean() * 100),
                "n": int(len(g)),
            })

    return {
        "test": "Machine Learning — La fascia predice il Top 5?",
        "descrizione": (
            "Classificazione binaria (Top 5 sì/no) usando come features la fascia "
            "di esibizione (one-hot 5 livelli), la posizione relativa e la serata. "
            "Validazione Leave-One-Year-Out per evitare data leakage temporale. "
            "Se i modelli non battono il baseline, la fascia non ha potere predittivo."
        ),
        "dataset": {
            "n_osservazioni": int(len(df)),
            "n_top5": int(y.sum()),
            "pct_top5": _sf(y.mean() * 100),
            "n_anni": int(len(np.unique(groups))),
        },
        "baseline": {
            "strategia": f"Predici sempre '{baseline_label}' (classe maggioritaria)",
            "accuracy": _sf(baseline_acc),
        },
        "modelli": model_results,
        "probabilita_top5_per_fascia": prob_per_fascia,
    }


# ============================================================
# Dati per visualizzazione
# ============================================================

def build_viz_data(raw: pd.DataFrame, artist: pd.DataFrame) -> dict:
    # Scatter per-serata
    scatter_ps = []
    for _, r in raw.iterrows():
        scatter_ps.append({
            "anno": int(r["year"]), "serata": int(r["serata"]),
            "artista": r["artist"],
            "posizione_relativa": _sf(r["posizione_relativa"]),
            "rank_normalizzato": _sf(r["rank_norm"]),
            "classifica_finale": int(r["classifica_finale"]),
            "fascia_5": str(r["fascia_5"]),
        })

    # Scatter aggregato per artista
    scatter_agg = []
    for _, r in artist.iterrows():
        scatter_agg.append({
            "anno": int(r["year"]), "artista": r["artist"],
            "posizione_relativa": _sf(r["pos_rel_media"]),
            "rank_normalizzato": _sf(r["rank_normalizzato"]),
            "classifica_finale": int(r["classifica_finale"]),
            "fascia_5": str(r["fascia_5"]),
        })

    # Boxplot 5 fasce (per-serata)
    boxplot_5 = []
    for fascia in FASCE_5_LABELS:
        g = raw[raw["fascia_5"] == fascia]["rank_norm"]
        if len(g) > 0:
            boxplot_5.append({
                "fascia": fascia, "n": int(len(g)),
                "min": _sf(g.min()), "q25": _sf(g.quantile(0.25)),
                "mediana": _sf(g.median()), "q75": _sf(g.quantile(0.75)),
                "max": _sf(g.max()), "media": _sf(g.mean()),
            })

    # Heatmap 5 fasce: anno × fascia → mediana rank (per-serata)
    heatmap = []
    for year in sorted(raw["year"].unique()):
        for fascia in FASCE_5_LABELS:
            g = raw[(raw["year"] == year) & (raw["fascia_5"] == fascia)]["rank_norm"]
            if len(g) > 0:
                heatmap.append({
                    "anno": int(year), "fascia": fascia,
                    "mediana_rank": _sf(g.median()), "media_rank": _sf(g.mean()),
                    "n": int(len(g)),
                })

    # Top 5 per fascia (per-serata)
    top5 = []
    for fascia in FASCE_5_LABELS:
        g = raw[raw["fascia_5"] == fascia]
        if len(g) > 0:
            top5.append({
                "fascia": fascia,
                "pct_top_5": _sf(g["is_top5"].mean() * 100),
                "n_top_5": int(g["is_top5"].sum()),
                "n_totale": int(len(g)),
            })

    # Vincitori: dove si sono esibiti?
    vincitori = artist[artist["classifica_finale"] == 1]
    vincitori_data = []
    for _, r in vincitori.iterrows():
        vincitori_data.append({
            "anno": int(r["year"]), "artista": r["artist"],
            "posizione_relativa": _sf(r["pos_rel_media"]),
            "fascia_5": str(r["fascia_5"]),
        })

    # Boxplot per singola serata (5 fasce)
    boxplot_per_serata = []
    for year in sorted(raw["year"].unique()):
        for serata in sorted(raw[raw["year"] == year]["serata"].unique()):
            sdf = raw[(raw["year"] == year) & (raw["serata"] == serata)]
            if len(sdf) < 6:
                continue
            for fascia in FASCE_5_LABELS:
                g = sdf[sdf["fascia_5"] == fascia]["rank_norm"]
                if len(g) > 0:
                    boxplot_per_serata.append({
                        "anno": int(year), "serata": int(serata),
                        "fascia": fascia, "n": int(len(g)),
                        "media": _sf(g.mean()), "mediana": _sf(g.median()),
                    })

    return {
        "scatter_per_serata": scatter_ps,
        "scatter_aggregato": scatter_agg,
        "boxplot_5_fasce": boxplot_5,
        "heatmap_anno_fascia": heatmap,
        "top5_per_fascia": top5,
        "vincitori": vincitori_data,
        "boxplot_per_serata": boxplot_per_serata,
    }


# ============================================================
# Sintesi
# ============================================================

def build_summary(tests: list[dict], raw: pd.DataFrame, artist: pd.DataFrame) -> dict:
    sig_tests = []
    for t in tests:
        name = t.get("test", "")
        # Test con campo diretto
        if t.get("significativo"):
            sig_tests.append(name)
        # Spearman con sotto-campi
        elif "per_serata" in t and isinstance(t["per_serata"], dict) and t["per_serata"].get("significativo"):
            sig_tests.append(f"{name} (per-serata)")
        elif "aggregato_artista" in t and isinstance(t["aggregato_artista"], dict) and t["aggregato_artista"].get("significativo"):
            sig_tests.append(f"{name} (aggregato)")
        # Fisher combined
        elif "fisher_combined" in t and t["fisher_combined"].get("significativo"):
            sig_tests.append(f"{name} (Fisher)")

    vincitori = artist[artist["classifica_finale"] == 1]
    pos_media_vincitori = vincitori["pos_rel_media"].mean()

    centro_raw = raw[raw["fascia_5"].isin(["Seconda fascia", "Centro", "Quarta fascia"])]["rank_norm"]
    estremi_raw = raw[raw["fascia_5"].isin(["Apertura", "Chiusura"])]["rank_norm"]

    sig_count = len(sig_tests)
    if sig_count >= 3:
        strength = "forte"
    elif sig_count >= 2:
        strength = "moderata"
    elif sig_count >= 1:
        strength = "debole"
    else:
        strength = "assente"

    if len(centro_raw) > 0 and len(estremi_raw) > 0:
        diff = estremi_raw.mean() - centro_raw.mean()
        if centro_raw.mean() < estremi_raw.mean():
            direction = f"Le fasce centrali hanno ranking medio migliore degli estremi (differenza: {diff:.3f})"
        else:
            direction = f"Le fasce centrali NON hanno ranking migliore degli estremi (differenza: {diff:.3f})"
    else:
        direction = "Dati insufficienti"

    return {
        "domanda": "È vero che chi si esibisce a metà serata ha più possibilità di vincere?",
        "dataset": {
            "n_osservazioni_per_serata": int(len(raw)),
            "n_artisti_unici": int(len(artist)),
            "n_anni": int(raw["year"].nunique()),
            "anni": sorted(int(y) for y in raw["year"].unique()),
            "serate_analizzate": "1-4 (competitive, esclusa la finale)",
        },
        "normalizzazione": {
            "posizione": "posizione_relativa nella serata (0=primo, 1=ultimo)",
            "ranking": "rank_normalizzato = (rank-1)/(N-1), dove N = partecipanti nell'anno",
            "fasce": "5 fasce equispaziate sulla posizione relativa (Apertura/Seconda/Centro/Quarta/Chiusura)",
        },
        "n_test_eseguiti": len(tests),
        "test_significativi": sig_tests,
        "posizione_media_vincitori": _sf(pos_media_vincitori),
        "rank_medio_centro": _sf(centro_raw.mean()),
        "rank_medio_estremi": _sf(estremi_raw.mean()),
        "conclusione": {
            "evidenza": strength,
            "direzione": direction,
            "nota": (
                f"Su {len(tests)} test statistici + ML, {sig_count} risultano significativi (p < 0.05). "
                f"La posizione media dei vincitori nelle serate competitive è "
                f"{pos_media_vincitori:.2f} (0=primo, 1=ultimo)."
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SANREMO — ANALISI TIMING: CHI SI ESIBISCE A METÀ HA PIÙ CHANCE?")
    print("=" * 70)

    print("\n[1/6] Caricamento dati...")
    raw_df = load_data()
    print(f"  Righe con classifica finale (serate 1-4): {len(raw_df)}")
    print(f"  Anni: {sorted(raw_df['year'].unique())}")

    print("\n[2/6] Preparazione dati (5 fasce per-serata)...")
    raw = enrich_raw(raw_df)
    artist = prepare_artist_data(raw_df)
    print(f"  Osservazioni per-serata: {len(raw)}")
    print(f"  Artisti unici: {len(artist)}")
    print(f"  Distribuzione fasce per-serata:")
    for f in FASCE_5_LABELS:
        n = (raw["fascia_5"] == f).sum()
        print(f"    {f}: {n}")

    print("\n[3/6] Test statistici...")

    t1 = test_spearman(raw, artist)
    print(f"  1) Spearman per-serata: rho={t1['per_serata']['rho']}, p={t1['per_serata']['p_value']}")
    print(f"     Spearman aggregato:  rho={t1['aggregato_artista']['rho']}, p={t1['aggregato_artista']['p_value']}")

    t2 = test_kruskal_wallis(raw)
    print(f"  2) Kruskal-Wallis 5 fasce: H={t2['H_statistic']}, p={t2['p_value']}")

    t3 = test_chi_squared(raw)
    print(f"  3) Chi² 5 fasce: chi²={t3['chi2']}, p={t3['p_value']}")

    t4 = test_mann_whitney(raw)
    print(f"  4) Mann-Whitney centro vs estremi: U={t4['U_statistic']}, p={t4['p_value']}")

    t5 = test_friedman(raw)
    print(f"  5) Friedman 5 fasce: chi²={t5['friedman_chi2']}, p={t5['p_value']}, W={t5['kendall_w']}")

    t6 = test_spearman_per_serata(raw)
    print(f"  6) Spearman per-serata: {t6['n_serate_analizzate']} serate, {t6['n_serate_significative']} significative")

    t7 = test_kruskal_per_serata(raw)
    print(f"  7) Kruskal per-serata + Fisher: p_Fisher={t7['fisher_combined']['p_value']}")

    all_tests = [t1, t2, t3, t4, t5, t6, t7]

    print("\n[4/6] Machine Learning...")
    t8 = test_ml(raw)
    for m in t8["modelli"]:
        print(f"  {m['modello']}: acc={m['accuracy']}, f1={m['f1_score']}, AUC={m['auc_roc']}, "
              f"vs baseline={t8['baseline']['accuracy']}")
    print(f"  Probabilità Top 5 per fascia (RF):")
    for f in t8["probabilita_top5_per_fascia"]:
        print(f"    {f['fascia']}: prob={f['prob_media_top5']:.3f}, effettiva={f['pct_effettiva_top5']:.1f}%")
    all_tests.append(t8)

    print("\n[5/6] Visualizzazione...")
    viz = build_viz_data(raw, artist)
    for k, v in viz.items():
        print(f"  {k}: {len(v)} items")

    print("\n[6/6] Generazione JSON...")
    summary = build_summary(all_tests, raw, artist)

    output = {
        "meta": {
            "titolo": "Sanremo — L'ordine di esibizione influenza la classifica?",
            "domanda": "È vero che chi si esibisce a metà serata ha più possibilità di vincere?",
            "metodologia": (
                "Analisi non parametrica + ML sull'ordine di esibizione nelle serate "
                "competitive (1-4) del Festival di Sanremo. 5 fasce equispaziate come fattori. "
                "Due livelli: (a) per-serata con fasce bilanciate, (b) aggregato per artista. "
                "ML con Leave-One-Year-Out cross-validation."
            ),
            "fonte_dati": "dati_sremo/sanremo_ordine_serate.csv",
            "anni_analizzati": sorted(int(y) for y in raw["year"].unique()),
            "n_osservazioni": int(len(raw)),
            "n_artisti": int(len(artist)),
            "fasce": FASCE_5_LABELS,
        },
        "sintesi": summary,
        "test_statistici": {
            "spearman": t1,
            "kruskal_wallis_5fasce": t2,
            "chi_squared_5fasce": t3,
            "mann_whitney_centro_estremi": t4,
            "friedman_5fasce": t5,
            "spearman_per_serata": t6,
            "kruskal_per_serata_fisher": t7,
            "machine_learning": t8,
        },
        "visualizzazione": viz,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"\n  Output: {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("SINTESI")
    print("=" * 70)
    c = summary["conclusione"]
    print(f"  Evidenza: {c['evidenza']}")
    print(f"  Direzione: {c['direzione']}")
    print(f"  {c['nota']}")
    if summary["test_significativi"]:
        print(f"  Test significativi: {', '.join(summary['test_significativi'])}")
    else:
        print("  Nessun test significativo (p < 0.05)")
    print("=" * 70)


if __name__ == "__main__":
    main()
