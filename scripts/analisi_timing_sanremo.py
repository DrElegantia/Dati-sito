#!/usr/bin/env python3
"""
Sanremo — Chi si esibisce a metà serata ha più possibilità di vincere?

Analisi statistica con:
  1) Spearman rank correlation
  2) Kruskal-Wallis H test (fasce di esibizione)
  3) Chi² test (proporzione top finishers per fascia)
  4) Mann-Whitney U test (centro vs estremi)
  5) Friedman test (variabilità tra anni, anno come blocco)

Tutti i risultati normalizzati per numero partecipanti.
Output: docs/sanremo_timing_analysis.json

Uso:
    python3 scripts/analisi_timing_sanremo.py
"""

import csv
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

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


# ============================================================
# Caricamento e preparazione dati
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, dtype={"classifica_finale": str, "classifica_serata": str})

    # Converti classifica_finale in numerico
    df["classifica_finale"] = pd.to_numeric(df["classifica_finale"], errors="coerce")

    # Filtra solo righe con classifica finale valida
    df = df[df["classifica_finale"].notna()].copy()
    df["classifica_finale"] = df["classifica_finale"].astype(int)

    # Filtra solo serate competitive (1-4), escludendo la finale (serata 5)
    # perché nella finale l'ordine È la classifica
    df = df[df["serata"].isin([1, 2, 3, 4])].copy()

    # Filtra serate di nuove proposte / cover
    serate_competitive = [
        "Prima serata", "Seconda serata", "Terza serata", "Quarta serata",
    ]
    df = df[df["serata_name"].isin(serate_competitive)].copy()

    return df


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni artista/anno: calcola la posizione relativa media nelle serate 1-4,
    poi normalizza il ranking finale per il numero di partecipanti.
    """
    # Numero di partecipanti per anno (dalla classifica finale)
    n_per_year = df.groupby("year")["classifica_finale"].max().to_dict()

    # Posizione relativa media per artista/anno (media sulle serate a cui ha partecipato)
    artist_data = (
        df.groupby(["year", "artist"])
        .agg(
            pos_rel_media=("posizione_relativa", "mean"),
            n_serate=("serata", "nunique"),
            classifica_finale=("classifica_finale", "first"),
            serate_list=("serata", lambda x: sorted(x.unique().tolist())),
        )
        .reset_index()
    )

    # Normalizza ranking: 0 = vincitore, 1 = ultimo
    artist_data["n_partecipanti"] = artist_data["year"].map(n_per_year)
    artist_data["rank_normalizzato"] = (
        (artist_data["classifica_finale"] - 1) /
        (artist_data["n_partecipanti"] - 1).clip(lower=1)
    )

    # Fasce a 3 livelli
    artist_data["fascia_3"] = pd.cut(
        artist_data["pos_rel_media"],
        bins=[-0.001, 1/3, 2/3, 1.001],
        labels=["Inizio", "Centro", "Fine"],
    )

    # Fasce a 5 livelli
    artist_data["fascia_5"] = pd.cut(
        artist_data["pos_rel_media"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["Primo quinto", "Secondo quinto", "Terzo quinto", "Quarto quinto", "Ultimo quinto"],
    )

    return artist_data


# ============================================================
# Test statistici
# ============================================================

def _safe_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), 6)


def _interpret_p(p, alpha=0.05):
    if p is None:
        return "non calcolabile"
    if p < 0.001:
        return "altamente significativo (p < 0.001)"
    if p < 0.01:
        return "molto significativo (p < 0.01)"
    if p < alpha:
        return "significativo (p < 0.05)"
    return "non significativo (p >= 0.05)"


def test_spearman(data: pd.DataFrame) -> dict:
    """Test 1: Correlazione di Spearman tra posizione relativa media e ranking normalizzato."""

    # Globale
    rho, p = stats.spearmanr(data["pos_rel_media"], data["rank_normalizzato"])

    # Per anno
    per_anno = []
    for year in sorted(data["year"].unique()):
        ydf = data[data["year"] == year]
        if len(ydf) < 5:
            continue
        r, pv = stats.spearmanr(ydf["pos_rel_media"], ydf["rank_normalizzato"])
        per_anno.append({
            "anno": int(year),
            "n": int(len(ydf)),
            "rho": _safe_float(r),
            "p_value": _safe_float(pv),
            "significativo": pv < 0.05 if pv is not None else False,
            "interpretazione": _interpret_p(pv),
        })

    return {
        "test": "Spearman Rank Correlation",
        "descrizione": "Correlazione tra posizione relativa media nell'ordine di esibizione e classifica finale normalizzata. rho > 0 = chi si esibisce tardi ha ranking peggiore (numero più alto). rho < 0 = chi si esibisce tardi ha ranking migliore.",
        "ipotesi_nulla": "Non c'è correlazione monotona tra ordine di esibizione e classifica finale",
        "globale": {
            "n": int(len(data)),
            "rho": _safe_float(rho),
            "p_value": _safe_float(p),
            "significativo": bool(p < 0.05) if p is not None else False,
            "interpretazione": _interpret_p(p),
        },
        "per_anno": per_anno,
    }


def test_kruskal_wallis(data: pd.DataFrame) -> dict:
    """Test 2: Kruskal-Wallis H test sulle fasce di esibizione."""
    results = {}

    for n_fasce, col in [(3, "fascia_3"), (5, "fascia_5")]:
        groups = []
        labels = []
        desc_stats = []

        for fascia in data[col].cat.categories:
            g = data[data[col] == fascia]["rank_normalizzato"]
            if len(g) > 0:
                groups.append(g.values)
                labels.append(str(fascia))
                desc_stats.append({
                    "fascia": str(fascia),
                    "n": int(len(g)),
                    "media": _safe_float(g.mean()),
                    "mediana": _safe_float(g.median()),
                    "dev_std": _safe_float(g.std()),
                    "q25": _safe_float(g.quantile(0.25)),
                    "q75": _safe_float(g.quantile(0.75)),
                })

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            stat, p = stats.kruskal(*groups)
            # Effect size: eta-squared = (H - k + 1) / (N - k)
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = (stat - k + 1) / (N - k) if N > k else None
        else:
            stat, p, eta_sq = None, None, None

        results[f"fasce_{n_fasce}"] = {
            "n_fasce": n_fasce,
            "statistiche_descrittive": desc_stats,
            "H_statistic": _safe_float(stat),
            "p_value": _safe_float(p),
            "eta_squared": _safe_float(eta_sq),
            "significativo": bool(p < 0.05) if p is not None else False,
            "interpretazione": _interpret_p(p),
        }

    return {
        "test": "Kruskal-Wallis H Test",
        "descrizione": "Test non parametrico sui ranghi per verificare se la distribuzione del ranking normalizzato differisce tra le fasce di esibizione. Equivalente non parametrico dell'ANOVA a una via.",
        "ipotesi_nulla": "Le distribuzioni del ranking normalizzato sono identiche tra le fasce",
        "risultati": results,
    }


def test_chi_squared(data: pd.DataFrame) -> dict:
    """Test 3: Chi² test sulla proporzione di Top 5 per fascia."""
    results = {}

    for n_fasce, col in [(3, "fascia_3"), (5, "fascia_5")]:
        n_per_year = data.groupby("year")["n_partecipanti"].first()
        # Top 5 relativo: nel top 20% dei partecipanti
        data_copy = data.copy()
        data_copy["is_top"] = data_copy["classifica_finale"] <= 5

        contingency_data = []
        for fascia in data_copy[col].cat.categories:
            g = data_copy[data_copy[col] == fascia]
            if len(g) > 0:
                n_top = int(g["is_top"].sum())
                n_not_top = int((~g["is_top"]).sum())
                contingency_data.append({
                    "fascia": str(fascia),
                    "top_5": n_top,
                    "non_top_5": n_not_top,
                    "totale": n_top + n_not_top,
                    "pct_top_5": _safe_float(n_top / (n_top + n_not_top) * 100),
                })

        if len(contingency_data) >= 2:
            ct = np.array([[d["top_5"], d["non_top_5"]] for d in contingency_data])
            if ct.min() >= 0 and ct.sum() > 0:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                # Cramér's V
                n_total = ct.sum()
                min_dim = min(ct.shape) - 1
                cramers_v = np.sqrt(chi2 / (n_total * max(min_dim, 1))) if n_total > 0 else None
            else:
                chi2, p, dof, cramers_v = None, None, None, None
        else:
            chi2, p, dof, cramers_v = None, None, None, None

        results[f"fasce_{n_fasce}"] = {
            "n_fasce": n_fasce,
            "contingenza": contingency_data,
            "chi2": _safe_float(chi2),
            "p_value": _safe_float(p),
            "gradi_liberta": int(dof) if dof is not None else None,
            "cramers_v": _safe_float(cramers_v),
            "significativo": bool(p < 0.05) if p is not None else False,
            "interpretazione": _interpret_p(p),
        }

    return {
        "test": "Chi-Squared Test of Independence",
        "descrizione": "Test chi-quadrato sulla tabella di contingenza: fascia di esibizione × arrivo in Top 5. Verifica se la probabilità di arrivare in Top 5 è indipendente dalla fascia.",
        "ipotesi_nulla": "La proporzione di Top 5 è uguale tra tutte le fasce",
        "risultati": results,
    }


def test_mann_whitney(data: pd.DataFrame) -> dict:
    """Test 4: Mann-Whitney U test — Centro vs Estremi."""
    centro = data[data["fascia_3"] == "Centro"]["rank_normalizzato"]
    estremi = data[data["fascia_3"] != "Centro"]["rank_normalizzato"]

    if len(centro) >= 3 and len(estremi) >= 3:
        stat, p = stats.mannwhitneyu(centro, estremi, alternative="two-sided")
        # Rank-biserial correlation come effect size
        n1, n2 = len(centro), len(estremi)
        r_effect = 1 - (2 * stat) / (n1 * n2)
    else:
        stat, p, r_effect = None, None, None

    # Calcolo direzione
    if centro.mean() < estremi.mean():
        direzione = "Il centro ha ranking migliore (più basso) degli estremi"
    else:
        direzione = "Il centro ha ranking peggiore (più alto) degli estremi"

    return {
        "test": "Mann-Whitney U Test",
        "descrizione": "Test non parametrico: confronto del ranking normalizzato tra chi si esibisce al centro (fascia 2/3) e chi agli estremi (fascia 1/3 + 3/3).",
        "ipotesi_nulla": "Le distribuzioni dei ranking sono identiche tra centro ed estremi",
        "centro": {
            "n": int(len(centro)),
            "media": _safe_float(centro.mean()),
            "mediana": _safe_float(centro.median()),
            "dev_std": _safe_float(centro.std()),
        },
        "estremi": {
            "n": int(len(estremi)),
            "media": _safe_float(estremi.mean()),
            "mediana": _safe_float(estremi.median()),
            "dev_std": _safe_float(estremi.std()),
        },
        "U_statistic": _safe_float(stat),
        "p_value": _safe_float(p),
        "rank_biserial_r": _safe_float(r_effect),
        "significativo": bool(p < 0.05) if p is not None else False,
        "direzione": direzione,
        "interpretazione": _interpret_p(p),
    }


def test_friedman(data: pd.DataFrame) -> dict:
    """
    Test 5: Friedman test — variabilità tra anni.

    Ogni anno è un blocco, le fasce sono i trattamenti.
    Per ogni (anno, fascia) calcoliamo la mediana del ranking normalizzato.
    Serve per controllare se l'effetto è stabile o varia tra anni.
    """
    results = {}

    for n_fasce, col in [(3, "fascia_3"), (5, "fascia_5")]:
        # Matrice: righe = anni, colonne = fasce
        years = sorted(data["year"].unique())
        fasce = list(data[col].cat.categories)

        matrix = []
        years_used = []
        for year in years:
            ydf = data[data["year"] == year]
            row = []
            complete = True
            for fascia in fasce:
                g = ydf[ydf[col] == fascia]["rank_normalizzato"]
                if len(g) == 0:
                    complete = False
                    break
                row.append(g.median())
            if complete:
                matrix.append(row)
                years_used.append(int(year))

        detail_per_anno = []
        for year in years:
            ydf = data[data["year"] == year]
            anno_detail = {"anno": int(year), "fasce": {}}
            for fascia in fasce:
                g = ydf[ydf[col] == fascia]["rank_normalizzato"]
                if len(g) > 0:
                    anno_detail["fasce"][str(fascia)] = {
                        "n": int(len(g)),
                        "mediana": _safe_float(g.median()),
                        "media": _safe_float(g.mean()),
                    }
            detail_per_anno.append(anno_detail)

        if len(matrix) >= 3:  # Friedman richiede almeno 3 blocchi
            matrix_np = np.array(matrix)
            stat, p = stats.friedmanchisquare(*[matrix_np[:, i] for i in range(matrix_np.shape[1])])
            # Effect size: Kendall's W = chi2_F / (n * (k-1))
            n_blocks = len(matrix)
            k_treatments = len(fasce)
            kendall_w = stat / (n_blocks * (k_treatments - 1)) if n_blocks > 0 and k_treatments > 1 else None
        else:
            stat, p, kendall_w = None, None, None

        results[f"fasce_{n_fasce}"] = {
            "n_fasce": n_fasce,
            "n_anni_usati": len(years_used),
            "anni_usati": years_used,
            "friedman_chi2": _safe_float(stat),
            "p_value": _safe_float(p),
            "kendall_w": _safe_float(kendall_w),
            "significativo": bool(p < 0.05) if p is not None else False,
            "interpretazione": _interpret_p(p),
            "dettaglio_per_anno": detail_per_anno,
        }

    return {
        "test": "Friedman Test",
        "descrizione": "Test non parametrico per misure ripetute: ogni anno è un blocco, le fasce di esibizione sono i trattamenti. Verifica se l'effetto della fascia è coerente tra anni diversi. Kendall's W misura la concordanza (0 = nessun accordo, 1 = accordo perfetto).",
        "ipotesi_nulla": "Non c'è differenza sistematica tra le fasce, al netto della variabilità tra anni",
        "risultati": results,
    }


# ============================================================
# Test aggiuntivo: analisi per-serata
# ============================================================

def test_per_serata(raw_df: pd.DataFrame) -> dict:
    """
    Analisi Spearman per ogni singola serata (1-4) di ogni anno.
    Usa la posizione relativa nella serata vs classifica finale normalizzata.
    """
    n_per_year = raw_df.groupby("year")["classifica_finale"].max().to_dict()
    raw_df = raw_df.copy()
    raw_df["rank_norm"] = raw_df.apply(
        lambda r: (r["classifica_finale"] - 1) / max(n_per_year.get(r["year"], 1) - 1, 1),
        axis=1,
    )

    per_serata = []
    for year in sorted(raw_df["year"].unique()):
        for serata in sorted(raw_df[raw_df["year"] == year]["serata"].unique()):
            sdf = raw_df[(raw_df["year"] == year) & (raw_df["serata"] == serata)]
            if len(sdf) < 5:
                continue
            rho, p = stats.spearmanr(sdf["posizione_relativa"], sdf["rank_norm"])
            per_serata.append({
                "anno": int(year),
                "serata": int(serata),
                "serata_name": sdf["serata_name"].iloc[0],
                "n": int(len(sdf)),
                "rho": _safe_float(rho),
                "p_value": _safe_float(p),
                "significativo": bool(p < 0.05) if p is not None else False,
                "interpretazione": _interpret_p(p),
            })

    n_sig = sum(1 for s in per_serata if s["significativo"])
    return {
        "test": "Spearman per-serata",
        "descrizione": "Correlazione di Spearman calcolata separatamente per ogni serata di ogni anno. Permette di identificare se l'effetto dell'ordine varia tra le serate.",
        "n_serate_analizzate": len(per_serata),
        "n_serate_significative": n_sig,
        "dettaglio": per_serata,
    }


def test_kruskal_per_serata(raw_df: pd.DataFrame) -> dict:
    """
    Kruskal-Wallis sulle fasce, ma calcolato per-serata (senza mediare).
    Con 12-25 artisti per serata, 3 fasce danno ~4-8 artisti per fascia: ben bilanciate.
    Poi aggreghiamo i p-value con il metodo di Fisher per un test globale.
    """
    n_per_year = raw_df.groupby("year")["classifica_finale"].max().to_dict()
    raw_df = raw_df.copy()
    raw_df["rank_norm"] = raw_df.apply(
        lambda r: (r["classifica_finale"] - 1) / max(n_per_year.get(r["year"], 1) - 1, 1),
        axis=1,
    )

    # Fasce per serata (3 e 5)
    per_serata_results = []

    for year in sorted(raw_df["year"].unique()):
        for serata in sorted(raw_df[raw_df["year"] == year]["serata"].unique()):
            sdf = raw_df[(raw_df["year"] == year) & (raw_df["serata"] == serata)].copy()
            if len(sdf) < 6:
                continue

            sdf["fascia_3"] = pd.cut(
                sdf["posizione_relativa"],
                bins=[-0.001, 1/3, 2/3, 1.001],
                labels=["Inizio", "Centro", "Fine"],
            )

            groups = []
            labels = []
            for fascia in ["Inizio", "Centro", "Fine"]:
                g = sdf[sdf["fascia_3"] == fascia]["rank_norm"]
                if len(g) >= 2:
                    groups.append(g.values)
                    labels.append(fascia)

            if len(groups) >= 2:
                h_stat, p_val = stats.kruskal(*groups)
            else:
                h_stat, p_val = None, None

            # Statistiche per fascia
            fasce_stats = {}
            for fascia in ["Inizio", "Centro", "Fine"]:
                g = sdf[sdf["fascia_3"] == fascia]["rank_norm"]
                if len(g) > 0:
                    fasce_stats[fascia] = {
                        "n": int(len(g)),
                        "media": _safe_float(g.mean()),
                        "mediana": _safe_float(g.median()),
                    }

            per_serata_results.append({
                "anno": int(year),
                "serata": int(serata),
                "n": int(len(sdf)),
                "H": _safe_float(h_stat),
                "p_value": _safe_float(p_val),
                "significativo": bool(p_val < 0.05) if p_val is not None else False,
                "fasce": fasce_stats,
            })

    # Fisher's combined probability test per un p-value globale
    valid_p = [r["p_value"] for r in per_serata_results if r["p_value"] is not None]
    if len(valid_p) >= 2:
        fisher_stat = -2 * sum(np.log(p) for p in valid_p)
        fisher_df = 2 * len(valid_p)
        fisher_p = 1 - stats.chi2.cdf(fisher_stat, fisher_df)
    else:
        fisher_stat, fisher_df, fisher_p = None, None, None

    n_sig = sum(1 for r in per_serata_results if r["significativo"])

    return {
        "test": "Kruskal-Wallis per-serata + Fisher combined",
        "descrizione": (
            "Kruskal-Wallis H test calcolato separatamente per ogni serata (non mediato). "
            "Con 12-25 artisti per serata le 3 fasce sono bilanciate (~4-8 per fascia). "
            "I p-value individuali vengono poi combinati con il metodo di Fisher per un test globale."
        ),
        "ipotesi_nulla": "In nessuna serata la fascia di esibizione influenza la classifica",
        "n_serate_analizzate": len(per_serata_results),
        "n_serate_significative": n_sig,
        "fisher_combined": {
            "chi2": _safe_float(fisher_stat),
            "df": fisher_df,
            "p_value": _safe_float(fisher_p),
            "significativo": bool(fisher_p < 0.05) if fisher_p is not None else False,
            "interpretazione": _interpret_p(fisher_p),
        },
        "dettaglio": per_serata_results,
    }


# ============================================================
# Dati per visualizzazione
# ============================================================

def build_viz_data(data: pd.DataFrame) -> dict:
    """Dati strutturati per grafici sul frontend."""

    # Scatter: posizione relativa vs ranking normalizzato
    scatter = []
    for _, row in data.iterrows():
        scatter.append({
            "anno": int(row["year"]),
            "artista": row["artist"],
            "posizione_relativa": _safe_float(row["pos_rel_media"]),
            "rank_normalizzato": _safe_float(row["rank_normalizzato"]),
            "classifica_finale": int(row["classifica_finale"]),
            "n_partecipanti": int(row["n_partecipanti"]),
            "fascia_3": str(row["fascia_3"]),
            "fascia_5": str(row["fascia_5"]),
        })

    # Boxplot data: per ogni fascia
    boxplot_3 = []
    for fascia in data["fascia_3"].cat.categories:
        g = data[data["fascia_3"] == fascia]["rank_normalizzato"]
        if len(g) > 0:
            boxplot_3.append({
                "fascia": str(fascia),
                "n": int(len(g)),
                "min": _safe_float(g.min()),
                "q25": _safe_float(g.quantile(0.25)),
                "mediana": _safe_float(g.median()),
                "q75": _safe_float(g.quantile(0.75)),
                "max": _safe_float(g.max()),
                "media": _safe_float(g.mean()),
            })

    boxplot_5 = []
    for fascia in data["fascia_5"].cat.categories:
        g = data[data["fascia_5"] == fascia]["rank_normalizzato"]
        if len(g) > 0:
            boxplot_5.append({
                "fascia": str(fascia),
                "n": int(len(g)),
                "min": _safe_float(g.min()),
                "q25": _safe_float(g.quantile(0.25)),
                "mediana": _safe_float(g.median()),
                "q75": _safe_float(g.quantile(0.75)),
                "max": _safe_float(g.max()),
                "media": _safe_float(g.mean()),
            })

    # Heatmap: anno x fascia → mediana rank
    heatmap_data = []
    for year in sorted(data["year"].unique()):
        for fascia in data["fascia_3"].cat.categories:
            g = data[(data["year"] == year) & (data["fascia_3"] == fascia)]["rank_normalizzato"]
            if len(g) > 0:
                heatmap_data.append({
                    "anno": int(year),
                    "fascia": str(fascia),
                    "mediana_rank": _safe_float(g.median()),
                    "media_rank": _safe_float(g.mean()),
                    "n": int(len(g)),
                })

    # Percentuale Top 5 per fascia
    top5_by_band = []
    for fascia in data["fascia_3"].cat.categories:
        g = data[data["fascia_3"] == fascia]
        if len(g) > 0:
            pct = (g["classifica_finale"] <= 5).mean() * 100
            top5_by_band.append({
                "fascia": str(fascia),
                "pct_top_5": _safe_float(pct),
                "n_top_5": int((g["classifica_finale"] <= 5).sum()),
                "n_totale": int(len(g)),
            })

    # Vincitori: in quale fascia si sono esibiti?
    vincitori = data[data["classifica_finale"] == 1].copy()
    vincitori_data = []
    for _, row in vincitori.iterrows():
        vincitori_data.append({
            "anno": int(row["year"]),
            "artista": row["artist"],
            "posizione_relativa": _safe_float(row["pos_rel_media"]),
            "fascia_3": str(row["fascia_3"]),
            "fascia_5": str(row["fascia_5"]),
        })

    return {
        "scatter": scatter,
        "boxplot_3_fasce": boxplot_3,
        "boxplot_5_fasce": boxplot_5,
        "heatmap_anno_fascia": heatmap_data,
        "top5_per_fascia": top5_by_band,
        "vincitori": vincitori_data,
    }


# ============================================================
# Sintesi e conclusioni
# ============================================================

def build_summary(test_results: list[dict], data: pd.DataFrame) -> dict:
    """Sintesi testuale dei risultati."""
    n_sig = sum(
        1 for t in test_results
        if t.get("significativo") or (
            isinstance(t.get("risultati"), dict) and
            any(v.get("significativo") for v in t["risultati"].values() if isinstance(v, dict))
        )
    )

    # Conta significatività aggregata
    sig_tests = []
    for t in test_results:
        name = t.get("test", "")
        if "globale" in t and t["globale"].get("significativo"):
            sig_tests.append(name)
        elif t.get("significativo"):
            sig_tests.append(name)
        elif isinstance(t.get("risultati"), dict):
            for key, val in t["risultati"].items():
                if isinstance(val, dict) and val.get("significativo"):
                    sig_tests.append(f"{name} ({key})")

    # Direzione dell'effetto
    vincitori = data[data["classifica_finale"] == 1]
    pos_media_vincitori = vincitori["pos_rel_media"].mean()

    centro = data[data["fascia_3"] == "Centro"]["rank_normalizzato"]
    estremi = data[data["fascia_3"] != "Centro"]["rank_normalizzato"]

    return {
        "domanda": "È vero che chi si esibisce a metà serata ha più possibilità di vincere?",
        "dataset": {
            "n_artisti": int(len(data)),
            "n_anni": int(data["year"].nunique()),
            "anni": sorted(int(y) for y in data["year"].unique()),
            "serate_analizzate": "1-4 (competitive, esclusa la finale)",
        },
        "normalizzazione": {
            "posizione": "posizione_relativa media nelle serate 1-4 (0=primo, 1=ultimo)",
            "ranking": "rank_normalizzato = (rank-1)/(N-1), dove N=partecipanti nell'anno",
        },
        "n_test_eseguiti": len(test_results),
        "test_significativi": sig_tests,
        "posizione_media_vincitori": _safe_float(pos_media_vincitori),
        "rank_medio_centro": _safe_float(centro.mean()) if len(centro) > 0 else None,
        "rank_medio_estremi": _safe_float(estremi.mean()) if len(estremi) > 0 else None,
        "conclusione": _build_conclusion(test_results, pos_media_vincitori, centro, estremi),
    }


def _build_conclusion(tests, pos_media_vincitori, centro, estremi):
    sig_count = 0
    for t in tests:
        if t.get("significativo") or (
            t.get("globale", {}).get("significativo")
        ):
            sig_count += 1
        elif isinstance(t.get("risultati"), dict):
            for val in t["risultati"].values():
                if isinstance(val, dict) and val.get("significativo"):
                    sig_count += 1
                    break

    if sig_count >= 3:
        strength = "forte"
    elif sig_count >= 2:
        strength = "moderata"
    elif sig_count >= 1:
        strength = "debole"
    else:
        strength = "assente"

    if len(centro) > 0 and len(estremi) > 0:
        if centro.mean() < estremi.mean():
            direction = "Il centro sembra avere un leggero vantaggio"
        else:
            direction = "Il centro non sembra avere un vantaggio"
    else:
        direction = "Dati insufficienti per determinare la direzione"

    return {
        "evidenza": strength,
        "direzione": direction,
        "nota": f"Su {len(tests)} test statistici, {sig_count} risultano significativi (p < 0.05). La posizione media dei vincitori nelle serate competitive è {pos_media_vincitori:.2f} (0=primo, 1=ultimo).",
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SANREMO — ANALISI TIMING: CHI SI ESIBISCE A METÀ HA PIÙ CHANCE?")
    print("=" * 70)

    print("\n[1/5] Caricamento dati...")
    raw_df = load_data()
    print(f"  Righe con classifica finale (serate 1-4): {len(raw_df)}")
    print(f"  Anni: {sorted(raw_df['year'].unique())}")

    print("\n[2/5] Preparazione dati (normalizzazione + fasce)...")
    data = prepare_analysis_data(raw_df)
    print(f"  Artisti unici con dati completi: {len(data)}")
    for y in sorted(data["year"].unique()):
        ydf = data[data["year"] == y]
        print(f"    {y}: {len(ydf)} artisti, n_partecipanti={ydf['n_partecipanti'].iloc[0]}")

    print("\n[3/5] Esecuzione test statistici...")

    test_1 = test_spearman(data)
    print(f"  1) Spearman: rho={test_1['globale']['rho']}, p={test_1['globale']['p_value']}")

    test_2 = test_kruskal_wallis(data)
    for k, v in test_2["risultati"].items():
        print(f"  2) Kruskal-Wallis ({k}): H={v['H_statistic']}, p={v['p_value']}")

    test_3 = test_chi_squared(data)
    for k, v in test_3["risultati"].items():
        print(f"  3) Chi² ({k}): chi²={v['chi2']}, p={v['p_value']}")

    test_4 = test_mann_whitney(data)
    print(f"  4) Mann-Whitney: U={test_4['U_statistic']}, p={test_4['p_value']}")

    test_5 = test_friedman(data)
    for k, v in test_5["risultati"].items():
        print(f"  5) Friedman ({k}): chi²={v['friedman_chi2']}, p={v['p_value']}, W={v['kendall_w']}")

    test_6 = test_per_serata(raw_df)
    print(f"  6) Spearman per-serata: {test_6['n_serate_analizzate']} serate, {test_6['n_serate_significative']} significative")

    test_7 = test_kruskal_per_serata(raw_df)
    fc = test_7["fisher_combined"]
    print(f"  7) Kruskal per-serata + Fisher: {test_7['n_serate_analizzate']} serate, Fisher p={fc['p_value']}")

    all_tests = [test_1, test_2, test_3, test_4, test_5, test_6, test_7]

    print("\n[4/5] Costruzione dati per visualizzazione...")
    viz_data = build_viz_data(data)
    print(f"  Scatter points: {len(viz_data['scatter'])}")
    print(f"  Vincitori tracciati: {len(viz_data['vincitori'])}")

    # Aggiungi scatter per-serata (senza media) per visualizzazioni più granulari
    n_per_year = raw_df.groupby("year")["classifica_finale"].max().to_dict()
    scatter_per_serata = []
    for _, row in raw_df.iterrows():
        n = n_per_year.get(row["year"], 1)
        rn = (row["classifica_finale"] - 1) / max(n - 1, 1)
        scatter_per_serata.append({
            "anno": int(row["year"]),
            "serata": int(row["serata"]),
            "artista": row["artist"],
            "posizione_relativa": _safe_float(row["posizione_relativa"]),
            "rank_normalizzato": _safe_float(rn),
            "classifica_finale": int(row["classifica_finale"]),
        })
    viz_data["scatter_per_serata"] = scatter_per_serata
    print(f"  Scatter per-serata: {len(scatter_per_serata)}")

    # Boxplot per-serata: fasce nella singola serata
    boxplot_per_serata = []
    for year in sorted(raw_df["year"].unique()):
        for serata in sorted(raw_df[raw_df["year"] == year]["serata"].unique()):
            sdf = raw_df[(raw_df["year"] == year) & (raw_df["serata"] == serata)].copy()
            if len(sdf) < 6:
                continue
            sdf["fascia"] = pd.cut(
                sdf["posizione_relativa"],
                bins=[-0.001, 1/3, 2/3, 1.001],
                labels=["Inizio", "Centro", "Fine"],
            )
            n = n_per_year.get(year, 1)
            for fascia in ["Inizio", "Centro", "Fine"]:
                g = sdf[sdf["fascia"] == fascia]["classifica_finale"].apply(
                    lambda r: (r - 1) / max(n - 1, 1)
                )
                if len(g) > 0:
                    boxplot_per_serata.append({
                        "anno": int(year),
                        "serata": int(serata),
                        "fascia": fascia,
                        "n": int(len(g)),
                        "media": _safe_float(g.mean()),
                        "mediana": _safe_float(g.median()),
                    })
    viz_data["boxplot_per_serata"] = boxplot_per_serata

    print("\n[5/5] Generazione JSON...")
    summary = build_summary(all_tests, data)

    output = {
        "meta": {
            "titolo": "Sanremo — L'ordine di esibizione influenza la classifica?",
            "domanda": "È vero che chi si esibisce a metà serata ha più possibilità di vincere?",
            "metodologia": (
                "Analisi statistica non parametrica sull'ordine di esibizione nelle serate "
                "competitive (1-4) del Festival di Sanremo. Posizione e ranking normalizzati "
                "per il numero di partecipanti. Fasce di esibizione usate come fattori categorici. "
                "Due livelli di analisi: (a) posizione media per artista su tutte le serate, "
                "(b) analisi per-serata con fasce bilanciate (~4-8 artisti per fascia)."
            ),
            "fonte_dati": "dati_sremo/sanremo_ordine_serate.csv",
            "anni_analizzati": sorted(int(y) for y in data["year"].unique()),
            "n_artisti_totali": int(len(data)),
            "n_osservazioni_per_serata": int(len(raw_df)),
            "serate_incluse": "1-4 (competitive)",
            "serate_escluse": "5 (finale — l'ordine è la classifica stessa)",
        },
        "sintesi": summary,
        "test_statistici": {
            "spearman": test_1,
            "kruskal_wallis": test_2,
            "chi_squared": test_3,
            "mann_whitney": test_4,
            "friedman": test_5,
            "spearman_per_serata": test_6,
            "kruskal_per_serata_fisher": test_7,
        },
        "visualizzazione": viz_data,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"\n  JSON salvato: {OUTPUT_JSON}")

    # Stampa sintesi
    print("\n" + "=" * 70)
    print("SINTESI")
    print("=" * 70)
    print(f"  Domanda: {summary['domanda']}")
    print(f"  Evidenza: {summary['conclusione']['evidenza']}")
    print(f"  Direzione: {summary['conclusione']['direzione']}")
    print(f"  {summary['conclusione']['nota']}")
    print(f"  Posizione media vincitori: {summary['posizione_media_vincitori']:.2f}")
    if summary["test_significativi"]:
        print(f"  Test significativi: {', '.join(summary['test_significativi'])}")
    else:
        print("  Nessun test significativo (p < 0.05)")
    print("=" * 70)


if __name__ == "__main__":
    main()
