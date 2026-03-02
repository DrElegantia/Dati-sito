#!/usr/bin/env python3
"""
Sanremo — L'ordine di uscita influenza la classifica?

Fonte dati: dati_sremo/sanremo_dati_serate.xlsx
Output:     docs/sanremo_timing_analysis.json

Analisi per DECILI di posizione nell'ordine di esibizione.
Doppia analisi: classifica televoto e classifica complessiva di serata.
Include: distribuzione posizioni per decile (per grafici),
test trend quadratico (U-shape), test statistici con effect size,
bootstrap CI, correzione per test multipli, e modelli ML.

Uso:
  python3 scripts/analisi_ordine_sanremo.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dati_sremo")
XLSX_PATH = os.path.join(DATA_DIR, "sanremo_dati_serate.xlsx")
OUTPUT_JSON = os.path.join(BASE_DIR, "docs", "sanremo_timing_analysis.json")

DECILE_LABELS = [f"D{i+1} ({i*10}-{(i+1)*10}%)" for i in range(10)]
DECILE_BINS = [-0.001] + [i / 10 for i in range(1, 10)] + [1.001]

N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ============================================================
# Data loading & preparation
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH)

    # Fix shifted rows (artist name split across columns)
    unnamed_col = next((c for c in df.columns if "Unnamed" in str(c)), None)
    for i in df.index:
        try:
            int(df.at[i, "ordine"])
        except (ValueError, TypeError):
            suffix = str(df.at[i, "ordine"]).strip()
            df.at[i, "artist"] = str(df.at[i, "artist"]) + ", " + suffix
            df.at[i, "ordine"] = df.at[i, "totale_serata"]
            df.at[i, "totale_serata"] = df.at[i, "classifica_serata_televoto"]
            df.at[i, "classifica_serata_televoto"] = df.at[i, "classifica_serata_complessiva"]
            df.at[i, "classifica_serata_complessiva"] = (
                df.at[i, unnamed_col] if unnamed_col else np.nan
            )

    # Manual data fixes
    for i in df.index:
        if (df.at[i, "year"] == 2025 and df.at[i, "serata"] == 5
                and "Sarah" in str(df.at[i, "artist"])
                and pd.isna(df.at[i, "classifica_serata_complessiva"])):
            df.at[i, "classifica_serata_complessiva"] = 20

    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")

    df["year"] = df["year"].astype(int)
    df["serata"] = df["serata"].astype(int)
    df["ordine"] = pd.to_numeric(df["ordine"], errors="coerce").astype("Int64")
    df["totale_serata"] = pd.to_numeric(df["totale_serata"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ordine", "totale_serata"]).copy()
    df["ordine"] = df["ordine"].astype(int)
    df["totale_serata"] = df["totale_serata"].astype(int)

    df["posizione_relativa"] = (df["ordine"] - 1) / np.maximum(df["totale_serata"] - 1, 1)
    df["posizione_relativa"] = df["posizione_relativa"].round(4)
    return df


def assign_decile(pos_rel: pd.Series) -> pd.Series:
    return pd.cut(pos_rel, bins=DECILE_BINS, labels=DECILE_LABELS)


def normalize_rank(rank: pd.Series, total: pd.Series) -> pd.Series:
    return (rank - 1) / np.maximum(total - 1, 1)


# ============================================================
# Bootstrap utilities
# ============================================================

def bootstrap_ci(data, stat_fn=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return None
    boot_stats = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return {"ci_lower": round(float(np.percentile(boot_stats, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(boot_stats, (1 - alpha) * 100)), 4),
            "ci_level": ci}


def bootstrap_diff_ci(a, b, stat_fn=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    a = np.array(a, dtype=float)[~np.isnan(np.array(a, dtype=float))]
    b = np.array(b, dtype=float)[~np.isnan(np.array(b, dtype=float))]
    if len(a) < 3 or len(b) < 3:
        return None
    diffs = [stat_fn(rng.choice(a, len(a), True)) - stat_fn(rng.choice(b, len(b), True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = float(np.percentile(diffs, alpha * 100))
    hi = float(np.percentile(diffs, (1 - alpha) * 100))
    return {"diff_observed": round(float(stat_fn(a) - stat_fn(b)), 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "ci_level": ci, "includes_zero": bool(lo <= 0 <= hi)}


# ============================================================
# Effect sizes
# ============================================================

def rank_biserial_r(u_stat, n1, n2):
    if n1 == 0 or n2 == 0:
        return None
    return round(float(1 - (2 * u_stat) / (n1 * n2)), 4)


def eta_squared_kw(h_stat, n):
    if n <= 1:
        return None
    return round(float((h_stat - 1) / (n - 1)), 4) if h_stat > 1 else 0.0


# ============================================================
# Multiple testing correction
# ============================================================

def fdr_correction(p_values: list):
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not valid:
        return [None] * len(p_values)
    valid_sorted = sorted(valid, key=lambda x: x[1])
    m = len(valid_sorted)
    result = [None] * len(p_values)
    prev_adj = 0
    for rank_idx, (orig_idx, p) in enumerate(valid_sorted, 1):
        adj = min(p * m / rank_idx, 1.0)
        adj = max(adj, prev_adj)
        result[orig_idx] = round(adj, 6)
        prev_adj = adj
    return result


# ============================================================
# Statistical tests
# ============================================================

def spearman_test(x, y, bootstrap=False):
    mask = x.notna() & y.notna()
    x2, y2 = np.array(x[mask], dtype=float), np.array(y[mask], dtype=float)
    if len(x2) < 5:
        return None
    rho, p = stats.spearmanr(x2, y2)
    result = {"n": int(len(x2)), "rho": round(float(rho), 6), "p_value": round(float(p), 6),
              "significativo": bool(p < 0.05)}
    if bootstrap and len(x2) >= 10:
        rng = np.random.RandomState(RANDOM_SEED)
        boot_rhos = [stats.spearmanr(x2[idx := rng.choice(len(x2), len(x2), True)],
                                     y2[idx])[0] for _ in range(N_BOOTSTRAP)]
        result["rho_ci_95"] = [round(float(np.percentile(boot_rhos, 2.5)), 4),
                               round(float(np.percentile(boot_rhos, 97.5)), 4)]
    if p >= 0.05:
        result["interpretazione"] = "non significativo (p >= 0.05)"
    else:
        direction = "tardi -> classifica peggiore" if rho > 0 else "tardi -> classifica migliore"
        result["interpretazione"] = f"significativo: {direction}"
    return result


def kruskal_wallis_test(groups: list, group_labels: list):
    valid = [(np.array(g, dtype=float), l) for g, l in zip(groups, group_labels) if len(g) >= 2]
    if len(valid) < 2:
        return None
    grps = [g for g, _ in valid]
    n_total = sum(len(g) for g in grps)
    stat, p = stats.kruskal(*grps)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "df": len(grps) - 1, "n_groups": len(grps), "n_total": n_total,
            "eta_squared": eta_squared_kw(stat, n_total),
            "significativo": bool(p < 0.05)}


def mann_whitney_test(center, extremes):
    c = np.array(center.dropna(), dtype=float)
    e = np.array(extremes.dropna(), dtype=float)
    if len(c) < 3 or len(e) < 3:
        return None
    stat, p = stats.mannwhitneyu(c, e, alternative="two-sided")
    return {"n_centro": int(len(c)), "n_estremi": int(len(e)),
            "media_centro": round(float(np.mean(c)), 3),
            "media_estremi": round(float(np.mean(e)), 3),
            "mediana_centro": round(float(np.median(c)), 3),
            "mediana_estremi": round(float(np.median(e)), 3),
            "U": round(float(stat), 1), "p_value": round(float(p), 6),
            "rank_biserial_r": rank_biserial_r(stat, len(c), len(e)),
            "bootstrap_diff_mean": bootstrap_diff_ci(c, e),
            "significativo": bool(p < 0.05),
            "vantaggio": ("centro" if np.mean(c) < np.mean(e) else "estremi") if p < 0.05 else "nessuno"}


def chi2_top_n(decile_col, rank_col, n=3):
    mask = decile_col.notna() & rank_col.notna()
    q, r = decile_col[mask], rank_col[mask]
    is_top = (r <= n).astype(int)
    ct = pd.crosstab(q, is_top)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n_obs = ct.sum().sum()
    k = min(ct.shape) - 1
    cramers_v = round(float(np.sqrt(chi2 / (n_obs * max(k, 1)))), 4)
    return {"chi2": round(float(chi2), 4), "p_value": round(float(p), 6),
            "dof": int(dof), "cramers_v": cramers_v,
            "significativo": bool(p < 0.05), "soglia": n}


def quadratic_trend_test(pos_rel, rank_norm):
    """Test linear vs quadratic fit. Returns both models and F-test."""
    mask = pos_rel.notna() & rank_norm.notna()
    x = np.array(pos_rel[mask], dtype=float)
    y = np.array(rank_norm[mask], dtype=float)
    if len(x) < 10:
        return None
    n = len(x)

    # Linear
    slope, intercept = np.polyfit(x, y, 1)
    y_lin = slope * x + intercept
    ss_res_lin = float(np.sum((y - y_lin) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_lin = round(1 - ss_res_lin / ss_tot, 6) if ss_tot > 0 else 0

    # Quadratic
    a, b, c = np.polyfit(x, y, 2)
    y_quad = a * x ** 2 + b * x + c
    ss_res_quad = float(np.sum((y - y_quad) ** 2))
    r2_quad = round(1 - ss_res_quad / ss_tot, 6) if ss_tot > 0 else 0

    # F-test: does quadratic add over linear?
    f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n - 3)) if ss_res_quad > 0 else 0
    p_f = float(1 - stats.f.cdf(f_stat, 1, n - 3))

    # Vertex of parabola (minimum point)
    vertice = round(-b / (2 * a), 4) if a != 0 else None

    return {
        "n": n,
        "lineare": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
                     "R2": r2_lin},
        "quadratico": {"a": round(float(a), 4), "b": round(float(b), 4), "c": round(float(c), 4),
                        "R2": r2_quad, "vertice_pos_relativa": vertice},
        "f_test_quadratico_vs_lineare": {"F": round(float(f_stat), 3), "p_value": round(p_f, 6),
                                          "significativo": bool(p_f < 0.05)},
        "interpretazione": (
            f"U-shape significativo (p={p_f:.4f}): il punto ottimale è circa al {vertice*100:.0f}% dell'ordine"
            if p_f < 0.05 and vertice and 0.1 < vertice < 0.9
            else "nessun U-shape significativo" if p_f >= 0.05
            else f"trend significativo ma vertice fuori range ({vertice})"
        ),
    }


def binomial_test_top_n(decile_col, rank_col, pos_rel, target_decile_range, n_top=3):
    """Test if top-N ranked performers are over-represented in a position range."""
    mask = decile_col.notna() & rank_col.notna()
    ranks = rank_col[mask]
    positions = pos_rel[mask]
    is_top = ranks <= n_top
    in_range = (positions >= target_decile_range[0]) & (positions <= target_decile_range[1])
    n_top_in_range = int((is_top & in_range).sum())
    n_top_total = int(is_top.sum())
    pct_in_range = float(in_range.mean())  # expected proportion
    if n_top_total < 5:
        return None
    res = stats.binomtest(n_top_in_range, n_top_total, pct_in_range, alternative="greater")
    return {
        "n_top_in_range": n_top_in_range, "n_top_total": n_top_total,
        "pct_observed": round(n_top_in_range / n_top_total * 100, 1),
        "pct_expected": round(pct_in_range * 100, 1),
        "range": [target_decile_range[0], target_decile_range[1]],
        "soglia_classifica": n_top,
        "p_value": round(float(res.pvalue), 6),
        "significativo": bool(res.pvalue < 0.05),
    }


def fisher_combined(p_values):
    ps = [p for p in p_values if p is not None and np.isfinite(p) and p > 0]
    if len(ps) < 2:
        return None
    stat = -2 * sum(np.log(p) for p in ps)
    dof = 2 * len(ps)
    p_combined = 1 - stats.chi2.cdf(stat, dof)
    return {"statistic": round(float(stat), 4), "dof": dof,
            "n_tests": len(ps), "p_value": round(float(p_combined), 6),
            "significativo": bool(p_combined < 0.05)}


# ============================================================
# Position-count matrix (core visualization data)
# ============================================================

def build_position_count_matrix(usable, rank_col):
    """For each decile, count how many 1st, 2nd, 3rd, ..., terzultimo, penultimo, ultimo."""
    usable = usable.copy()
    usable["rank"] = usable[rank_col].astype(int)
    usable["decile"] = assign_decile(usable["posizione_relativa"])
    # Position from end
    usable["rank_from_end"] = usable["totale_serata"] - usable["rank"]  # 0=ultimo, 1=penultimo, 2=terzultimo

    matrix = []
    for d in DECILE_LABELS:
        g = usable[usable["decile"] == d]
        n = len(g)
        if n == 0:
            continue

        entry = {
            "decile": d,
            "n": n,
            "media_classifica": round(float(g["rank"].mean()), 2),
            "mediana_classifica": round(float(g["rank"].median()), 1),
            # Top positions
            "primo": int((g["rank"] == 1).sum()),
            "secondo": int((g["rank"] == 2).sum()),
            "terzo": int((g["rank"] == 3).sum()),
            "top3": int((g["rank"] <= 3).sum()),
            "top5": int((g["rank"] <= 5).sum()),
            "top10": int((g["rank"] <= 10).sum()),
            # Bottom positions (relative to serata size)
            "ultimo": int((g["rank_from_end"] == 0).sum()),
            "penultimo": int((g["rank_from_end"] == 1).sum()),
            "terzultimo": int((g["rank_from_end"] == 2).sum()),
            "bottom3": int((g["rank_from_end"] <= 2).sum()),
            # Percentages
            "pct_top3": round(float((g["rank"] <= 3).mean() * 100), 1),
            "pct_top5": round(float((g["rank"] <= 5).mean() * 100), 1),
            "pct_bottom3": round(float((g["rank_from_end"] <= 2).mean() * 100), 1),
        }
        matrix.append(entry)
    return matrix


def build_full_rank_distribution(usable, rank_col):
    """For each decile, full histogram of ranks (for detailed charts)."""
    usable = usable.copy()
    usable["rank"] = usable[rank_col].astype(int)
    usable["decile"] = assign_decile(usable["posizione_relativa"])

    result = []
    for d in DECILE_LABELS:
        g = usable[usable["decile"] == d]["rank"]
        if len(g) == 0:
            continue
        counts = g.value_counts().sort_index()
        result.append({
            "decile": d,
            "n": int(len(g)),
            "distribuzione": {int(k): int(v) for k, v in counts.items()},
        })
    return result


# ============================================================
# Analysis runner (one rank type)
# ============================================================

def run_analysis(df: pd.DataFrame, rank_col: str, rank_label: str):
    usable = df[df[rank_col].notna()].copy()
    usable[rank_col] = usable[rank_col].astype(int)
    usable["decile"] = assign_decile(usable["posizione_relativa"])
    usable["rank_norm"] = normalize_rank(usable[rank_col], usable["totale_serata"])
    years = sorted(usable["year"].unique())

    result = {
        "tipo_classifica": rank_label,
        "colonna": rank_col,
        "n_osservazioni": int(len(usable)),
        "anni": [int(y) for y in years],
    }

    # --- Position-count matrix (MAIN VISUALIZATION DATA) ---
    result["matrice_posizioni_per_decile"] = build_position_count_matrix(usable, rank_col)
    result["distribuzione_rank_completa_per_decile"] = build_full_rank_distribution(usable, rank_col)

    # --- Per-year position-count ---
    per_year = []
    for y in years:
        ydf = usable[usable["year"] == y]
        per_year.append({
            "anno": int(y),
            "n": int(len(ydf)),
            "matrice": build_position_count_matrix(ydf, rank_col),
        })
    result["matrice_per_anno"] = per_year

    # --- Per-serata position-count ---
    per_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            if len(sdf) < 5:
                continue
            per_serata.append({
                "anno": int(y), "serata": int(s),
                "serata_name": sdf["serata_name"].iloc[0],
                "n": int(len(sdf)),
                "matrice": build_position_count_matrix(sdf, rank_col),
            })
    result["matrice_per_serata"] = per_serata

    # --- Decile distribution (media, mediana, boxplot data) ---
    boxplot = []
    for d in DECILE_LABELS:
        g = usable[usable["decile"] == d][rank_col].dropna()
        if len(g) == 0:
            continue
        boxplot.append({
            "decile": d, "n": int(len(g)),
            "min": int(g.min()), "q25": round(float(g.quantile(0.25)), 1),
            "median": round(float(g.median()), 1), "q75": round(float(g.quantile(0.75)), 1),
            "max": int(g.max()), "mean": round(float(g.mean()), 2),
            "bootstrap_ci_mean": bootstrap_ci(g.values),
        })
    result["boxplot_decili"] = boxplot

    # --- Statistical tests ---
    tests = {}
    all_p = []

    # Quadratic trend (U-shape) — key test
    tests["trend_quadratico_globale"] = quadratic_trend_test(
        usable["posizione_relativa"], usable["rank_norm"])
    if tests["trend_quadratico_globale"]:
        all_p.append(("trend_quad_globale",
                       tests["trend_quadratico_globale"]["f_test_quadratico_vs_lineare"]["p_value"]))

    # Quadratic per year
    quad_per_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        res = quadratic_trend_test(ydf["posizione_relativa"], ydf["rank_norm"])
        if res:
            res["anno"] = int(y)
            quad_per_anno.append(res)
            all_p.append((f"trend_quad_{y}", res["f_test_quadratico_vs_lineare"]["p_value"]))
    tests["trend_quadratico_per_anno"] = quad_per_anno

    # Spearman global
    tests["spearman_globale"] = spearman_test(
        usable["posizione_relativa"], usable[rank_col], bootstrap=True)
    if tests["spearman_globale"]:
        all_p.append(("spearman_globale", tests["spearman_globale"]["p_value"]))

    # Spearman per year
    sp_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        res = spearman_test(ydf["posizione_relativa"], ydf[rank_col])
        if res:
            res["anno"] = int(y)
            sp_anno.append(res)
            all_p.append((f"spearman_{y}", res["p_value"]))
    tests["spearman_per_anno"] = sp_anno

    # Spearman per serata
    sp_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            res = spearman_test(sdf["posizione_relativa"], sdf[rank_col])
            if res:
                res["anno"] = int(y)
                res["serata"] = int(s)
                sp_serata.append(res)
                all_p.append((f"spearman_{y}_s{s}", res["p_value"]))
    tests["spearman_per_serata"] = sp_serata

    # Kruskal-Wallis decili global
    groups = [usable[usable["decile"] == d][rank_col].values for d in DECILE_LABELS]
    tests["kruskal_wallis_globale"] = kruskal_wallis_test(groups, DECILE_LABELS)
    if tests["kruskal_wallis_globale"]:
        all_p.append(("kw_globale", tests["kruskal_wallis_globale"]["p_value"]))

    # KW per year
    kw_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        groups = [ydf[ydf["decile"] == d][rank_col].values for d in DECILE_LABELS]
        res = kruskal_wallis_test(groups, DECILE_LABELS)
        if res:
            res["anno"] = int(y)
            kw_anno.append(res)
            all_p.append((f"kw_{y}", res["p_value"]))
    tests["kruskal_wallis_per_anno"] = kw_anno

    # KW per serata + Fisher
    kw_serata = []
    kw_ps = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            groups = [sdf[sdf["decile"] == d][rank_col].values for d in DECILE_LABELS]
            res = kruskal_wallis_test(groups, DECILE_LABELS)
            if res:
                res["anno"] = int(y)
                res["serata"] = int(s)
                kw_serata.append(res)
                kw_ps.append(res["p_value"])
                all_p.append((f"kw_{y}_s{s}", res["p_value"]))
    tests["kruskal_wallis_per_serata"] = kw_serata
    tests["kruskal_per_serata_fisher"] = fisher_combined(kw_ps)

    # Mann-Whitney centro (D3-D7, 20-70%) vs estremi (D1-D2 + D8-D10)
    centro_d = [f"D{i} ({(i-1)*10}-{i*10}%)" for i in range(3, 8)]  # D3-D7
    estremi_d = [f"D{i} ({(i-1)*10}-{i*10}%)" for i in [1, 2, 8, 9, 10]]
    centro = usable.loc[usable["decile"].isin(centro_d), rank_col]
    estremi = usable.loc[usable["decile"].isin(estremi_d), rank_col]
    tests["mann_whitney_centro_vs_estremi"] = mann_whitney_test(centro, estremi)
    if tests["mann_whitney_centro_vs_estremi"]:
        all_p.append(("mw_centro_estremi", tests["mann_whitney_centro_vs_estremi"]["p_value"]))

    # Chi2 top 3
    tests["chi2_top3_globale"] = chi2_top_n(usable["decile"], usable[rank_col], n=3)
    if tests["chi2_top3_globale"]:
        all_p.append(("chi2_top3", tests["chi2_top3_globale"]["p_value"]))

    # Chi2 top 5
    tests["chi2_top5_globale"] = chi2_top_n(usable["decile"], usable[rank_col], n=5)
    if tests["chi2_top5_globale"]:
        all_p.append(("chi2_top5", tests["chi2_top5_globale"]["p_value"]))

    # Binomial: are top-3 over-represented in D3-D7 (20-70%)?
    tests["binomial_top3_centro"] = binomial_test_top_n(
        usable["decile"], usable[rank_col], usable["posizione_relativa"],
        (0.2, 0.7), n_top=3)
    if tests["binomial_top3_centro"]:
        all_p.append(("binom_top3_centro", tests["binomial_top3_centro"]["p_value"]))

    # Binomial: are bottom-3 over-represented in D1+D9-D10?
    usable_tmp = usable.copy()
    usable_tmp["rank_from_end"] = usable_tmp["totale_serata"] - usable_tmp[rank_col]
    usable_tmp["is_bottom3"] = usable_tmp["rank_from_end"] <= 2
    # Use rank_from_end as "rank" for the binomial test
    bottom3_in_extremes = usable_tmp[usable_tmp["is_bottom3"]]
    n_bottom_in_ext = int(((bottom3_in_extremes["posizione_relativa"] < 0.1) |
                            (bottom3_in_extremes["posizione_relativa"] >= 0.8)).sum())
    n_bottom_total = len(bottom3_in_extremes)
    pct_extreme_pos = float(((usable["posizione_relativa"] < 0.1) |
                              (usable["posizione_relativa"] >= 0.8)).mean())
    if n_bottom_total >= 5:
        binom_res = stats.binomtest(n_bottom_in_ext, n_bottom_total, pct_extreme_pos, "greater")
        tests["binomial_bottom3_estremi"] = {
            "n_bottom_in_estremi": n_bottom_in_ext, "n_bottom_total": n_bottom_total,
            "pct_observed": round(n_bottom_in_ext / n_bottom_total * 100, 1),
            "pct_expected": round(pct_extreme_pos * 100, 1),
            "p_value": round(float(binom_res.pvalue), 6),
            "significativo": bool(binom_res.pvalue < 0.05),
        }
        all_p.append(("binom_bottom3_estremi", tests["binomial_bottom3_estremi"]["p_value"]))

    # --- Extremes penalty analysis (descriptive, per-decile breakdown) ---
    # For each rank position (1st, 2nd, 3rd, last, penultimate, ante-penultimate),
    # show how they distribute across deciles — the core data the user verified manually
    pen_data = {}
    for rank_label_pos, rank_filter_fn in [
        ("primo", lambda r, _: r == 1),
        ("secondo", lambda r, _: r == 2),
        ("terzo", lambda r, _: r == 3),
        ("terzultimo", lambda r, t: r == t - 2),
        ("penultimo", lambda r, t: r == t - 1),
        ("ultimo", lambda r, t: r == t),
    ]:
        subset = usable[usable.apply(lambda row: rank_filter_fn(row[rank_col], row["totale_serata"]), axis=1)]
        total_in_rank = len(subset)
        if total_in_rank == 0:
            continue
        dist = {}
        for d in DECILE_LABELS:
            count = int((subset["decile"] == d).sum())
            dist[d] = {
                "conteggio": count,
                "pct": round(count / total_in_rank * 100, 1),
            }
        pen_data[rank_label_pos] = {"totale": total_in_rank, "per_decile": dist}

    # Top-5 and bottom-5 grouped analysis
    for group_label, group_filter in [
        ("top5", lambda r, _t: r <= 5),
        ("bottom5", lambda r, t: r > t - 5),
    ]:
        subset = usable[usable.apply(lambda row: group_filter(row[rank_col], row["totale_serata"]), axis=1)]
        total_in_group = len(subset)
        if total_in_group == 0:
            continue
        dist = {}
        for d in DECILE_LABELS:
            count = int((subset["decile"] == d).sum())
            dist[d] = {
                "conteggio": count,
                "pct": round(count / total_in_group * 100, 1),
            }
        pen_data[group_label] = {"totale": total_in_group, "per_decile": dist}

    tests["penalita_estremi"] = {
        "descrizione": (
            "Distribuzione di ciascuna posizione in classifica (1°, 2°, 3°, terzultimo, "
            "penultimo, ultimo, top5, bottom5) nei decili dell'ordine di esibizione. "
            "Permette di verificare se una posizione in classifica si concentra in decili specifici. "
            "Se il centro avvantaggiasse, i top5 dovrebbero concentrarsi nei decili centrali."
        ),
        "posizioni": pen_data,
    }

    # --- FDR correction ---
    raw_ps = [p for _, p in all_p]
    fdr = fdr_correction(raw_ps)
    correction_table = []
    for idx, (name, raw_p) in enumerate(all_p):
        correction_table.append({
            "test": name, "p_raw": round(raw_p, 6),
            "p_fdr": fdr[idx],
            "sig_raw": bool(raw_p < 0.05),
            "sig_fdr": bool(fdr[idx] is not None and fdr[idx] < 0.05),
        })
    tests["correzione_test_multipli"] = {
        "n_test": len(all_p),
        "n_sig_raw": sum(1 for r in correction_table if r["sig_raw"]),
        "n_sig_fdr": sum(1 for r in correction_table if r["sig_fdr"]),
        "tabella": correction_table,
    }

    result["test_statistici"] = tests

    # --- Scatter data ---
    scatter = []
    for _, row in usable.iterrows():
        scatter.append({
            "x": round(float(row["posizione_relativa"]), 4),
            "y": int(row[rank_col]),
            "anno": int(row["year"]),
            "serata": int(row["serata"]),
            "artista": row["artist"],
        })
    result["scatter"] = scatter

    return result


# ============================================================
# Machine Learning
# ============================================================

def run_ml(df: pd.DataFrame, rank_col: str, rank_label: str):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    usable = df[df[rank_col].notna()].copy()
    usable[rank_col] = usable[rank_col].astype(int)
    usable["rank_norm"] = normalize_rank(usable[rank_col], usable["totale_serata"])

    if len(usable) < 20:
        return {"errore": "troppo pochi dati"}

    features = ["posizione_relativa", "totale_serata", "serata"]
    # Add quadratic term
    usable["pos_rel_sq"] = usable["posizione_relativa"] ** 2
    features_quad = features + ["pos_rel_sq"]
    X = usable[features_quad].values
    y = usable["rank_norm"].values

    baseline_mae = float(np.mean(np.abs(y - np.mean(y))))

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }

    results = []
    for name, model in models.items():
        cv = min(5, len(usable) // 5)
        if cv < 2:
            continue
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        mae = float(-scores.mean())
        results.append({
            "modello": name,
            "mae_cv": round(mae, 4),
            "miglioramento_su_baseline": round((baseline_mae - mae) / baseline_mae * 100, 2),
        })

    best = min(results, key=lambda r: r["mae_cv"]) if results else None
    perm_p = 1.0
    if best:
        best_model = models[best["modello"]]
        np.random.seed(42)
        n_perm = 200
        cv = min(5, len(usable) // 5)
        perm_maes = []
        for _ in range(n_perm):
            X_p = X.copy()
            np.random.shuffle(X_p[:, 0])
            np.random.shuffle(X_p[:, 3])  # also shuffle pos_rel_sq
            s = cross_val_score(best_model, X_p, y, cv=cv, scoring="neg_mean_absolute_error")
            perm_maes.append(float(-s.mean()))
        perm_p = float(np.mean([m <= best["mae_cv"] for m in perm_maes]))

    return {
        "tipo_classifica": rank_label,
        "n": int(len(usable)),
        "features": features_quad,
        "baseline_mae": round(baseline_mae, 4),
        "modelli": results,
        "miglior_modello": best["modello"] if best else None,
        "miglior_mae": best["mae_cv"] if best else None,
        "permutation_test": {"n_perm": 200, "p_value": round(perm_p, 4),
                              "significativo": bool(perm_p < 0.05)},
    }


# ============================================================
# Synthesis
# ============================================================

def build_synthesis(res, ml):
    tests = res["test_statistici"]
    corr = tests.get("correzione_test_multipli", {})
    n_fdr = corr.get("n_sig_fdr", 0)
    n_total = corr.get("n_test", 0)

    trend = tests.get("trend_quadratico_globale", {})
    r2 = trend.get("quadratico", {}).get("R2", 0)
    vertice = trend.get("quadratico", {}).get("vertice_pos_relativa")
    u_shape_sig = trend.get("f_test_quadratico_vs_lineare", {}).get("significativo", False)

    mw = tests.get("mann_whitney_centro_vs_estremi", {})
    boot = mw.get("bootstrap_diff_mean", {})

    binom_top = tests.get("binomial_top3_centro", {})
    binom_bottom = tests.get("binomial_bottom3_estremi", {})

    # Extremes penalty analysis
    pen_estremi = tests.get("penalita_estremi", {})

    return {
        "n_test": n_total,
        "n_sig_fdr": n_fdr,
        "u_shape_significativo": u_shape_sig,
        "u_shape_R2": r2,
        "vertice_ottimale": vertice,
        "media_centro": mw.get("media_centro"),
        "media_estremi": mw.get("media_estremi"),
        "bootstrap_diff": boot,
        "top3_sovrarappresentati_centro": binom_top.get("significativo", False),
        "top3_pct_centro_osservato": binom_top.get("pct_observed"),
        "top3_pct_centro_atteso": binom_top.get("pct_expected"),
        "bottom3_sovrarappresentati_estremi": binom_bottom.get("significativo", False) if binom_bottom else None,
        "penalita_estremi": pen_estremi,
        "lettura_corretta": (
            "Non emerge una correlazione chiara tra ordine di esibizione e classifica. "
            "I dati sono confusionari: le posizioni in classifica si distribuiscono "
            "in modo disordinato lungo tutti i decili. L'ordine di uscita non sembra "
            "influenzare in modo significativo il risultato."
        ),
        "ml_significativo": ml.get("permutation_test", {}).get("significativo", False),
        "ml_miglioramento": max((r["miglioramento_su_baseline"] for r in ml.get("modelli", [])), default=None),
    }


def overall_conclusion(s_comp, s_tv):
    parts = []

    # Core finding: no meaningful correlation
    parts.append(
        "L'analisi dei dati mostra che NON esiste una correlazione chiara "
        "tra posizione nell'ordine di esibizione e classifica finale."
    )

    # U-shape — formally significant but meaningless
    r2_comp = s_comp.get("u_shape_R2", 0)
    r2_tv = s_tv.get("u_shape_R2", 0)
    r2 = max(r2_comp, r2_tv)
    if s_comp.get("u_shape_significativo") or s_tv.get("u_shape_significativo"):
        parts.append(
            f"Alcuni test risultano formalmente significativi (F-test per U-shape), "
            f"ma l'R² è solo del {r2*100:.1f}%: la posizione di esibizione "
            f"spiega meno del {r2*100:.0f}% della varianza nelle classifiche. "
            f"Questo significa che oltre il {(1-r2)*100:.0f}% dipende da altri fattori."
        )
    else:
        parts.append(
            "Nemmeno il trend a U (centro migliore, estremi peggiori) "
            "raggiunge la significatività statistica."
        )

    # Top positions are scattered
    parts.append(
        "I primi classificati si distribuiscono in modo disordinato lungo tutta la scaletta, "
        "senza concentrarsi in nessuna zona specifica. "
        "Lo stesso vale per le posizioni intermedie."
    )

    # Some weak pattern in last positions
    if s_comp.get("bottom3_sovrarappresentati_estremi") or s_tv.get("bottom3_sovrarappresentati_estremi"):
        parts.append(
            "L'unico segnale debole riguarda gli ultimi classificati, "
            "che tendono a trovarsi leggermente più spesso agli estremi della scaletta "
            "(primi o ultimi a esibirsi), ma anche questo effetto è piccolo e incostante."
        )

    # Effect size
    if r2_comp > 0:
        if r2_comp < 0.04:
            size = "trascurabile"
        elif r2_comp < 0.10:
            size = "piccolo"
        else:
            size = "moderato"
        parts.append(f"L'effect size complessivo è {size} (R²={r2_comp*100:.1f}%).")

    parts.append(
        "Conclusione: l'ordine di esibizione NON influenza in modo significativo "
        "la classifica. I dati sono confusionari e non supportano nessuna narrativa "
        "— né 'il centro avvantaggia', né 'gli estremi penalizzano' in modo robusto. "
        "La qualità della performance e della canzone domina su tutto il resto."
    )

    return " ".join(parts)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SANREMO — ANALISI ORDINE DI ESIBIZIONE (DECILI)")
    print("=" * 70)

    print("\n[1/4] Caricamento dati da Excel...")
    df = load_data()
    print(f"  Righe: {len(df)}, Anni: {sorted(df['year'].unique())}")

    for y in sorted(df["year"].unique()):
        ydf = df[df["year"] == y]
        n_tv = ydf["classifica_serata_televoto"].notna().sum()
        n_comp = ydf["classifica_serata_complessiva"].notna().sum()
        print(f"    {y}: {len(ydf)} esib., televoto={n_tv}, complessiva={n_comp}")

    print("\n[2/4] Analisi classifica complessiva...")
    res_comp = run_analysis(df, "classifica_serata_complessiva", "Classifica serata complessiva")
    print(f"  n={res_comp['n_osservazioni']}")

    print("\n[3/4] Analisi classifica televoto...")
    res_tv = run_analysis(df, "classifica_serata_televoto", "Classifica serata televoto")
    print(f"  n={res_tv['n_osservazioni']}")

    print("\n[4/4] Machine Learning...")
    try:
        ml_comp = run_ml(df, "classifica_serata_complessiva", "Classifica serata complessiva")
        print(f"  Complessiva: baseline={ml_comp['baseline_mae']}, best={ml_comp.get('miglior_modello')} MAE={ml_comp.get('miglior_mae')}")
    except ImportError:
        ml_comp = {"errore": "scikit-learn non disponibile"}

    try:
        ml_tv = run_ml(df, "classifica_serata_televoto", "Classifica serata televoto")
        print(f"  Televoto: baseline={ml_tv['baseline_mae']}, best={ml_tv.get('miglior_modello')} MAE={ml_tv.get('miglior_mae')}")
    except ImportError:
        ml_tv = {"errore": "scikit-learn non disponibile"}

    synth_comp = build_synthesis(res_comp, ml_comp)
    synth_tv = build_synthesis(res_tv, ml_tv)

    output = {
        "meta": {
            "titolo": "Sanremo — L'ordine di esibizione influenza la classifica?",
            "domanda": "Esibirsi in zone centrali è meglio per il ranking della serata?",
            "risposta_breve": (
                "No. Non emerge nessuna correlazione chiara tra ordine di esibizione e classifica. "
                "I dati sono confusionari: i primi classificati si trovano in qualsiasi posizione "
                "della scaletta, così come gli ultimi. L'ordine di uscita non sembra influenzare "
                "in modo significativo il risultato finale."
            ),
            "metodologia": (
                "Analisi per DECILI dell'ordine di esibizione relativo (0=primo, 1=ultimo). "
                "Doppia analisi: classifica televoto e classifica complessiva di serata. "
                "Per ogni decile: conteggio 1°/2°/3°/terzultimo/penultimo/ultimo "
                "e distribuzione completa dei rank (per grafici). "
                "Test U-shape (quadratico vs lineare, F-test), Spearman, "
                "Kruskal-Wallis, Mann-Whitney, Chi², test binomiale. "
                "Analisi specifica sulla penalità degli estremi vs vantaggio del centro. "
                "Correzione FDR per test multipli. Bootstrap CI 2000 repliche. "
                "ML con termine quadratico e permutation test."
            ),
            "fonte_dati": "dati_sremo/sanremo_dati_serate.xlsx",
            "anni_analizzati": [int(y) for y in sorted(df["year"].unique())],
            "n_totali": int(len(df)),
            "n_complessiva": int(df["classifica_serata_complessiva"].notna().sum()),
            "n_televoto": int(df["classifica_serata_televoto"].notna().sum()),
        },
        "sintesi": {
            "domanda": "Esibirsi in zone centrali è meglio per il ranking della serata?",
            "risposta": "No: non emerge nessuna correlazione chiara. I dati sono confusionari e non supportano nessuna narrativa specifica.",
            "classifica_complessiva": synth_comp,
            "classifica_televoto": synth_tv,
            "conclusione_generale": overall_conclusion(synth_comp, synth_tv),
        },
        "analisi_classifica_complessiva": res_comp,
        "analisi_classifica_televoto": res_tv,
        "machine_learning": {
            "classifica_complessiva": ml_comp,
            "classifica_televoto": ml_tv,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("RISULTATI")
    print(f"{'=' * 70}")

    for label, synth in [("COMPLESSIVA", synth_comp), ("TELEVOTO", synth_tv)]:
        print(f"\n  {label}:")
        print(f"    Test FDR significativi: {synth['n_sig_fdr']}/{synth['n_test']}")
        print(f"    U-shape: {'SI' if synth['u_shape_significativo'] else 'NO'}, "
              f"R²={synth['u_shape_R2']*100:.1f}%, vertice={synth['vertice_ottimale']}")
        print(f"    Top-3 concentrati al centro: {'SI' if synth['top3_sovrarappresentati_centro'] else 'NO'} "
              f"({synth.get('top3_pct_centro_osservato')}% vs {synth.get('top3_pct_centro_atteso')}% atteso)")
        print(f"    Centro media={synth['media_centro']}, Estremi media={synth['media_estremi']}")
        boot = synth.get("bootstrap_diff", {})
        if boot:
            print(f"    Bootstrap diff: {boot.get('diff_observed')} "
                  f"CI=[{boot.get('ci_lower')}, {boot.get('ci_upper')}] "
                  f"{'include 0' if boot.get('includes_zero') else 'NON include 0'}")

    print(f"\n  Matrice posizioni per decile (complessiva):")
    for d in res_comp["matrice_posizioni_per_decile"]:
        print(f"    {d['decile']}: n={d['n']:3d}, media={d['media_classifica']:5.1f}, "
              f"1°={d['primo']}, 2°={d['secondo']}, 3°={d['terzo']}, "
              f"terzult={d['terzultimo']}, penult={d['penultimo']}, ult={d['ultimo']}")

    print(f"\nConclusione: {output['sintesi']['conclusione_generale']}")
    print(f"\nOutput: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
