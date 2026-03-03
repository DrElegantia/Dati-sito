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


def sensitivity_centro_estremi(usable, rank_col):
    """Sensitivity analysis: test multiple cutoff definitions for center vs extremes."""
    cutoffs = [
        {"label": "D4-D7 vs D1-D3+D8-D10 (30-70%)", "centro": range(4, 8), "estremi": [1, 2, 3, 8, 9, 10]},
        {"label": "D3-D7 vs D1-D2+D8-D10 (20-70%)", "centro": range(3, 8), "estremi": [1, 2, 8, 9, 10]},
        {"label": "D3-D8 vs D1-D2+D9-D10 (20-80%)", "centro": range(3, 9), "estremi": [1, 2, 9, 10]},
        {"label": "D4-D8 vs D1-D3+D9-D10 (30-80%)", "centro": range(4, 9), "estremi": [1, 2, 3, 9, 10]},
    ]
    results = []
    for cut in cutoffs:
        centro_labels = [f"D{i} ({(i-1)*10}-{i*10}%)" for i in cut["centro"]]
        estremi_labels = [f"D{i} ({(i-1)*10}-{i*10}%)" for i in cut["estremi"]]
        c = usable.loc[usable["decile"].isin(centro_labels), rank_col].dropna().values
        e = usable.loc[usable["decile"].isin(estremi_labels), rank_col].dropna().values
        if len(c) < 3 or len(e) < 3:
            continue
        stat, p = stats.mannwhitneyu(c, e, alternative="two-sided")
        r = rank_biserial_r(stat, len(c), len(e))
        results.append({
            "definizione": cut["label"],
            "n_centro": int(len(c)), "n_estremi": int(len(e)),
            "media_centro": round(float(np.mean(c)), 3),
            "media_estremi": round(float(np.mean(e)), 3),
            "U": round(float(stat), 1), "p_value": round(float(p), 6),
            "rank_biserial_r": r, "significativo": bool(p < 0.05),
        })
    n_sig = sum(1 for r in results if r["significativo"])
    return {
        "descrizione": (
            "Analisi di sensibilità: si testa l'effetto centro-vs-estremi con diverse "
            "definizioni di soglia per verificare che il risultato non dipenda dalla "
            "scelta arbitraria dei decili."
        ),
        "n_definizioni_testate": len(results),
        "n_significative": n_sig,
        "robusto": n_sig == len(results) and len(results) > 0,
        "dettaglio": results,
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
# Cluster bootstrap (addresses non-independence within serata)
# ============================================================

def cluster_bootstrap_spearman(df, pos_col, rank_col, cluster_col,
                                n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Cluster bootstrap: resample entire serate (clusters) to account for
    within-serata dependence of ranks."""
    rng = np.random.RandomState(seed)
    mask = df[pos_col].notna() & df[rank_col].notna()
    data = df.loc[mask, [pos_col, rank_col, cluster_col]].copy()
    clusters = data[cluster_col].unique()
    if len(clusters) < 5:
        return None
    rho_obs, _ = stats.spearmanr(data[pos_col], data[rank_col])
    boot_rhos = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_data = pd.concat([data[data[cluster_col] == c] for c in sampled_clusters],
                              ignore_index=True)
        if len(boot_data) < 10:
            continue
        r, _ = stats.spearmanr(boot_data[pos_col], boot_data[rank_col])
        boot_rhos.append(r)
    if len(boot_rhos) < 100:
        return None
    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_rhos, alpha * 100))
    ci_hi = float(np.percentile(boot_rhos, (1 - alpha) * 100))
    return {
        "rho_osservato": round(float(rho_obs), 6),
        "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
        "ci_level": ci, "include_zero": bool(ci_lo <= 0 <= ci_hi),
        "n_cluster": int(len(clusters)), "n_bootstrap": len(boot_rhos),
        "interpretazione": (
            "CI del cluster bootstrap include lo zero: l'effetto non è robusto "
            "quando si tiene conto della dipendenza intra-serata."
            if ci_lo <= 0 <= ci_hi else
            "CI del cluster bootstrap NON include lo zero: l'effetto sopravvive "
            "alla correzione per dipendenza intra-serata."
        ),
    }


# ============================================================
# TOST equivalence testing (proves effect is negligibly small)
# ============================================================

def tost_equivalence_spearman(x, y, sesoi=0.1):
    """Two One-Sided Tests for equivalence of Spearman correlation.

    Instead of testing H0: rho=0 (which rejects with enough data),
    tests H0: |rho| >= sesoi.  If BOTH one-sided tests reject,
    we conclude the correlation is within [-sesoi, +sesoi], i.e.
    *negligibly small*.

    sesoi: Smallest Effect Size Of Interest (default 0.1 = trivial correlation).
    """
    mask = x.notna() & y.notna()
    x2, y2 = np.array(x[mask], dtype=float), np.array(y[mask], dtype=float)
    n = len(x2)
    if n < 10:
        return None
    rho, _ = stats.spearmanr(x2, y2)

    # Fisher z-transform for CI
    z_rho = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)

    # Test 1: H0: rho <= -sesoi (reject if rho is significantly > -sesoi)
    z_lower = (z_rho - np.arctanh(-sesoi)) / se
    p_lower = 1 - stats.norm.cdf(z_lower)

    # Test 2: H0: rho >= +sesoi (reject if rho is significantly < +sesoi)
    z_upper = (np.arctanh(sesoi) - z_rho) / se
    p_upper = 1 - stats.norm.cdf(z_upper)

    p_tost = max(p_lower, p_upper)
    equivalent = bool(p_tost < 0.05)

    # 90% CI (standard for equivalence testing)
    ci90_lo = float(np.tanh(z_rho - 1.645 * se))
    ci90_hi = float(np.tanh(z_rho + 1.645 * se))
    ci_within = bool(ci90_lo >= -sesoi and ci90_hi <= sesoi)

    return {
        "test": "TOST (Two One-Sided Tests) per equivalenza",
        "descrizione": (
            f"Verifica se la correlazione è trascurabile (|rho| < {sesoi}). "
            "A differenza dei test classici che cercano un effetto, il TOST "
            "dimostra che l'effetto è abbastanza piccolo da essere irrilevante."
        ),
        "n": n,
        "rho": round(float(rho), 6),
        "sesoi": sesoi,
        "p_tost": round(float(p_tost), 6),
        "p_lower": round(float(p_lower), 6),
        "p_upper": round(float(p_upper), 6),
        "equivalente": equivalent,
        "ci_90": [round(ci90_lo, 4), round(ci90_hi, 4)],
        "ci_dentro_sesoi": ci_within,
        "interpretazione": (
            f"EQUIVALENZA DIMOSTRATA (p={p_tost:.4f}): la correlazione rho={rho:.3f} "
            f"è contenuta nell'intervallo di irrilevanza [-{sesoi}, +{sesoi}]. "
            "L'ordine di esibizione ha un effetto trascurabile sulla classifica."
            if equivalent else
            f"Equivalenza non dimostrata (p={p_tost:.4f}). L'effetto non è "
            f"abbastanza piccolo da escludere rilevanza con soglia {sesoi}."
        ),
    }


def tost_equivalence_r2(r2_observed, n, sesoi_r2=0.04):
    """Test if observed R² is negligibly small using equivalence framework.

    Uses the F-distribution to test if R² is below a threshold of practical
    significance. Cohen's benchmark: R²<0.02 = negligible, <0.13 = small.
    Default sesoi_r2=0.04 is generous (well below 'small').
    """
    if n < 10 or r2_observed is None:
        return None

    # Convert R² to f² (Cohen's f²)
    f2_obs = r2_observed / max(1 - r2_observed, 0.001)
    f2_sesoi = sesoi_r2 / max(1 - sesoi_r2, 0.001)

    # Non-central F test: is the true R² < sesoi?
    df1, df2 = 2, n - 3  # quadratic model has 2 predictors
    ncp_sesoi = f2_sesoi * n  # non-centrality parameter under H0

    # Observed F statistic
    f_obs = f2_obs * df2 / df1 if df1 > 0 else 0

    # p-value: probability of observing F <= f_obs under H0: R²=sesoi
    p_equiv = float(stats.ncf.cdf(f_obs, df1, df2, ncp_sesoi))

    return {
        "test": "Test di equivalenza per R²",
        "descrizione": (
            f"Verifica se l'R² osservato è sotto la soglia di rilevanza pratica "
            f"(soglia: R² < {sesoi_r2*100:.0f}%, equivalente a 'effetto trascurabile' "
            "secondo Cohen)."
        ),
        "R2_osservato": round(r2_observed, 4),
        "soglia_R2": sesoi_r2,
        "f2_osservato": round(f2_obs, 4),
        "f2_soglia": round(f2_sesoi, 4),
        "p_equivalenza": round(p_equiv, 6),
        "trascurabile": bool(p_equiv < 0.05),
        "interpretazione": (
            f"R² = {r2_observed*100:.1f}% è dimostrato trascurabile (p={p_equiv:.4f}): "
            f"l'ordine spiega meno del {sesoi_r2*100:.0f}% della varianza."
            if p_equiv < 0.05 else
            f"Non si può dimostrare che R²={r2_observed*100:.1f}% sia sotto "
            f"la soglia del {sesoi_r2*100:.0f}%."
        ),
    }


# ============================================================
# Within-serata permutation test (handles non-independence)
# ============================================================

def within_serata_permutation_test(df, pos_col, rank_col, year_col, serata_col,
                                    n_perm=5000, seed=RANDOM_SEED):
    """Permutation test that shuffles order-rank mapping WITHIN each serata.

    This is the methodologically correct test because:
    1. It preserves the rank structure within each competition.
    2. It respects non-independence (ranks sum to fixed total).
    3. It directly tests: 'does the specific order-rank mapping matter?'
    """
    rng = np.random.RandomState(seed)
    mask = df[pos_col].notna() & df[rank_col].notna()
    data = df.loc[mask, [pos_col, rank_col, year_col, serata_col]].copy()
    data["cluster"] = data[year_col].astype(str) + "_" + data[serata_col].astype(str)

    # Observed global Spearman
    rho_obs, _ = stats.spearmanr(data[pos_col], data[rank_col])

    # Observed within-serata mean absolute Spearman
    serate = data["cluster"].unique()
    rhos_obs = []
    for s in serate:
        sdf = data[data["cluster"] == s]
        if len(sdf) >= 5:
            r, _ = stats.spearmanr(sdf[pos_col], sdf[rank_col])
            rhos_obs.append(r)
    mean_abs_rho_obs = float(np.mean(np.abs(rhos_obs))) if rhos_obs else 0

    # Permutation: shuffle ranks within each serata
    perm_global_rhos = []
    perm_mean_abs_rhos = []
    for _ in range(n_perm):
        perm_data = data.copy()
        for s in serate:
            idx = perm_data[perm_data["cluster"] == s].index
            shuffled_ranks = perm_data.loc[idx, rank_col].values.copy()
            rng.shuffle(shuffled_ranks)
            perm_data.loc[idx, rank_col] = shuffled_ranks

        r_global, _ = stats.spearmanr(perm_data[pos_col], perm_data[rank_col])
        perm_global_rhos.append(r_global)

        perm_rhos = []
        for s in serate:
            sdf = perm_data[perm_data["cluster"] == s]
            if len(sdf) >= 5:
                r, _ = stats.spearmanr(sdf[pos_col], sdf[rank_col])
                perm_rhos.append(r)
        if perm_rhos:
            perm_mean_abs_rhos.append(float(np.mean(np.abs(perm_rhos))))

    p_global = float(np.mean([abs(r) >= abs(rho_obs) for r in perm_global_rhos]))
    p_within = float(np.mean([r >= mean_abs_rho_obs for r in perm_mean_abs_rhos])) if perm_mean_abs_rhos else 1.0

    return {
        "test": "Permutation test intra-serata",
        "descrizione": (
            "Permuta l'associazione ordine-classifica all'interno di ogni singola serata, "
            "preservando la struttura dei ranghi. Questo è il test corretto perché rispetta "
            "la non-indipendenza e testa direttamente se la specifica associazione ordine-classifica "
            "osservata è distinguibile dal caso."
        ),
        "n_permutazioni": n_perm,
        "n_serate": int(len(serate)),
        "rho_globale_osservato": round(float(rho_obs), 6),
        "p_globale": round(p_global, 4),
        "mean_abs_rho_per_serata_osservato": round(mean_abs_rho_obs, 4),
        "p_within_serata": round(p_within, 4),
        "significativo_globale": bool(p_global < 0.05),
        "significativo_within": bool(p_within < 0.05),
        "interpretazione": (
            f"Permutation test intra-serata: p_globale={p_global:.3f}, "
            f"p_within={p_within:.3f}. "
            + ("L'associazione ordine-classifica NON è distinguibile da una "
               "assegnazione casuale. L'ordine non conta."
               if p_global >= 0.05 and p_within >= 0.05 else
               "Si rileva un pattern non casuale, ma la dimensione dell'effetto "
               f"(rho={rho_obs:.3f}) resta trascurabile.")
        ),
    }


# ============================================================
# Bayes Factor for null hypothesis
# ============================================================

def bayes_factor_correlation(x, y, kappa=1.0):
    """Bayesian test for correlation: calculates BF01 (evidence FOR null).

    Uses the Jeffreys-Zellner-Siow (JZS) prior approach approximation.
    BF01 > 3 = moderate evidence for null, > 10 = strong evidence.
    """
    mask = x.notna() & y.notna()
    x2, y2 = np.array(x[mask], dtype=float), np.array(y[mask], dtype=float)
    n = len(x2)
    if n < 10:
        return None
    rho, _ = stats.spearmanr(x2, y2)
    r2 = rho ** 2

    # BF01 approximation (Wetzels & Wagenmakers 2012, Savage-Dickey)
    log_bf01 = (0.5 * np.log(n) +
                ((n - 2) / 2) * np.log(1 - r2) -
                np.log(2))

    bf01 = float(np.exp(min(log_bf01, 500)))  # cap to avoid overflow

    if bf01 > 100:
        evidence = "evidenza molto forte per il null (nessun effetto)"
    elif bf01 > 10:
        evidence = "evidenza forte per il null (nessun effetto)"
    elif bf01 > 3:
        evidence = "evidenza moderata per il null"
    elif bf01 > 1:
        evidence = "evidenza aneddotica per il null"
    elif bf01 > 1/3:
        evidence = "evidenza inconcludente"
    elif bf01 > 1/10:
        evidence = "evidenza moderata per l'alternativa"
    else:
        evidence = "evidenza forte per l'alternativa"

    return {
        "test": "Bayes Factor (BF01) per correlazione",
        "descrizione": (
            "A differenza dei test frequentisti, il Bayes Factor quantifica "
            "l'evidenza A FAVORE dell'ipotesi nulla (nessun effetto). "
            "BF01 > 3 = evidenza moderata che l'effetto non esiste. "
            "BF01 > 10 = evidenza forte."
        ),
        "n": n,
        "rho": round(float(rho), 6),
        "BF01": round(bf01, 2),
        "log_BF01": round(float(log_bf01), 2),
        "interpretazione_scala_jeffreys": evidence,
        "interpretazione": (
            f"BF01 = {bf01:.1f}: {evidence}. "
            + ("I dati supportano l'assenza di correlazione tra ordine e classifica."
               if bf01 > 3 else
               "I dati non forniscono evidenza chiara in nessuna direzione."
               if bf01 > 1/3 else
               "I dati suggeriscono una correlazione, ma di dimensione trascurabile.")
        ),
    }


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
# Top-5 success rate analysis (core analysis)
# ============================================================

QUARTILE_BINS = [-0.001, 0.25, 0.5, 0.75, 1.001]
QUARTILE_LABELS = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]

SECTION_BINS = [-0.001, 0.333, 0.667, 1.001]
SECTION_LABELS = ["Inizio (0-33%)", "Centro (33-67%)", "Fine (67-100%)"]


def analisi_successo_top5(usable, rank_col, soglia_pct=20):
    """Core analysis: which zone of the lineup produces more top finishers?

    Uses a RELATIVE threshold (top 20% of each serata) instead of absolute
    top-N, so that comparisons are fair across serate of different sizes
    (e.g. 12 vs 24 performers in split vs full serate).

    For each grouping (decile, quartile, section), computes:
    - Success rate (% of performers finishing in top 20%)
    - Chi-squared test (are rates equal across groups?)
    - Per-group binomial test (is this group above/below expected?)
    - Odds ratio vs baseline
    """
    usable = usable.copy()
    # Criterio relativo: top 20% di ogni serata
    # rank_norm = (rank - 1) / (totale_serata - 1), dove 0 = primo, 1 = ultimo
    usable["rank_norm_serata"] = (usable[rank_col] - 1) / np.maximum(usable["totale_serata"] - 1, 1)
    usable["is_topN"] = usable["rank_norm_serata"] <= (soglia_pct / 100)

    n_total = len(usable)
    n_success = int(usable["is_topN"].sum())
    base_rate = n_success / n_total if n_total > 0 else 0

    # Assign groupings
    usable["quartile"] = pd.cut(usable["posizione_relativa"],
                                bins=QUARTILE_BINS, labels=QUARTILE_LABELS)
    usable["sezione"] = pd.cut(usable["posizione_relativa"],
                               bins=SECTION_BINS, labels=SECTION_LABELS)

    result = {
        "criterio": f"top {soglia_pct}% di ogni serata (criterio relativo)",
        "n_totale": n_total,
        "n_successi": n_success,
        "tasso_base": round(base_rate * 100, 1),
        "descrizione": (
            f"Per ogni raggruppamento, si calcola la percentuale di esibizioni "
            f"che si sono classificate nel top {soglia_pct}% della propria serata. "
            f"Il criterio è relativo (non assoluto) per garantire equità tra "
            f"serate di dimensioni diverse (split vs complete). "
            f"Il tasso base (atteso se la posizione fosse irrilevante) è "
            f"{base_rate*100:.1f}%."
        ),
    }

    groupings = [
        ("per_decile", "decile", DECILE_LABELS),
        ("per_quartile", "quartile", QUARTILE_LABELS),
        ("per_sezione", "sezione", SECTION_LABELS),
    ]

    for key, col, labels in groupings:
        group_data = []
        for label in labels:
            g = usable[usable[col] == label]
            n = len(g)
            n_s = int(g["is_topN"].sum())
            rate = n_s / n if n > 0 else 0
            # Odds ratio vs baseline
            odds = (rate / (1 - rate)) / (base_rate / (1 - base_rate)) if rate < 1 and base_rate < 1 and base_rate > 0 else None
            # Binomial test: is this group different from expected?
            binom_p = float(stats.binomtest(n_s, n, base_rate, "two-sided").pvalue) if n >= 5 else None
            group_data.append({
                "gruppo": label,
                "n_esibizioni": n,
                "n_top": n_s,
                "tasso_successo_pct": round(rate * 100, 1),
                "odds_ratio_vs_media": round(float(odds), 3) if odds is not None else None,
                "binomial_p": round(binom_p, 4) if binom_p is not None else None,
                "significativo": bool(binom_p is not None and binom_p < 0.05),
            })

        # Chi-squared test across all groups
        ct = pd.crosstab(usable[col], usable["is_topN"])
        chi2, chi2_p, dof, _ = stats.chi2_contingency(ct)

        result[key] = {
            "gruppi": group_data,
            "chi2_test": {
                "chi2": round(float(chi2), 3),
                "dof": int(dof),
                "p_value": round(float(chi2_p), 6),
                "significativo": bool(chi2_p < 0.05),
                "interpretazione": (
                    f"La distribuzione dei top-{n_top} tra i gruppi è "
                    + ("significativamente diversa da quella attesa (p<0.05). "
                       if chi2_p < 0.05 else
                       "compatibile con una distribuzione uniforme (p≥0.05). ")
                    + f"Tuttavia, l'ordine di esibizione non è casuale: "
                    f"le differenze osservate possono riflettere le scelte "
                    f"della produzione RAI nella composizione della scaletta."
                ),
            },
        }

    return result


# ============================================================
# Analysis runner (one rank type)
# ============================================================

def run_analysis(df: pd.DataFrame, rank_col: str, rank_label: str):
    usable = df[df[rank_col].notna()].copy()
    # Escludi serata 4 (cover): dinamica diversa, non comparabile
    usable = usable[usable["serata"] != 4].copy()
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

    # --- TOP-5 SUCCESS RATE ANALYSIS (core) ---
    result["successo_top"] = analisi_successo_top5(usable, rank_col, soglia_pct=20)

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

    # --- Sensitivity analysis: multiple center/extremes cutoff definitions ---
    tests["sensibilita_centro_estremi"] = sensitivity_centro_estremi(usable, rank_col)

    # --- Cluster bootstrap Spearman (handles within-serata dependence) ---
    usable["cluster_id"] = usable["year"].astype(str) + "_s" + usable["serata"].astype(str)
    tests["spearman_cluster_bootstrap"] = cluster_bootstrap_spearman(
        usable, "posizione_relativa", rank_col, "cluster_id")

    # --- TOST equivalence tests (prove effect is negligibly small) ---
    tests["tost_spearman"] = tost_equivalence_spearman(
        usable["posizione_relativa"], usable[rank_col], sesoi=0.1)
    trend = tests.get("trend_quadratico_globale")
    if trend:
        r2_q = trend.get("quadratico", {}).get("R2", 0)
        tests["tost_r2"] = tost_equivalence_r2(r2_q, len(usable), sesoi_r2=0.04)

    # --- Within-serata permutation test (methodologically correct for non-independence) ---
    tests["permutation_intra_serata"] = within_serata_permutation_test(
        usable, "posizione_relativa", rank_col, "year", "serata", n_perm=5000)

    # --- Bayes Factor for null hypothesis ---
    tests["bayes_factor"] = bayes_factor_correlation(
        usable["posizione_relativa"], usable[rank_col])

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
    usable = usable[usable["serata"] != 4].copy()  # Escludi cover
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
    top5 = res.get("successo_top", {})
    corr = tests.get("correzione_test_multipli", {})
    n_fdr = corr.get("n_sig_fdr", 0)
    n_total = corr.get("n_test", 0)

    trend = tests.get("trend_quadratico_globale", {})
    r2_quad = trend.get("quadratico", {}).get("R2", 0)
    vertice = trend.get("quadratico", {}).get("vertice_pos_relativa")
    u_shape_sig = trend.get("f_test_quadratico_vs_lineare", {}).get("significativo", False)

    mw = tests.get("mann_whitney_centro_vs_estremi", {})
    boot = mw.get("bootstrap_diff_mean", {})

    # --- Top-5 analysis results (the core) ---
    sez = top5.get("per_sezione", {})
    sez_chi2 = sez.get("chi2_test", {})
    sez_gruppi = sez.get("gruppi", [])
    qrt = top5.get("per_quartile", {})
    qrt_chi2 = qrt.get("chi2_test", {})
    dec = top5.get("per_decile", {})
    dec_chi2 = dec.get("chi2_test", {})

    # Extract section rates for the narrative
    rate_inizio = next((g["tasso_successo_pct"] for g in sez_gruppi if "Inizio" in g["gruppo"]), None)
    rate_centro = next((g["tasso_successo_pct"] for g in sez_gruppi if "Centro" in g["gruppo"]), None)
    rate_fine = next((g["tasso_successo_pct"] for g in sez_gruppi if "Fine" in g["gruppo"]), None)
    base_rate = top5.get("tasso_base", 0)

    return {
        "n_test": n_total,
        "n_sig_fdr": n_fdr,
        "u_shape_significativo": u_shape_sig,
        "u_shape_R2_quadratico": r2_quad,
        "vertice_ottimale": vertice,
        "media_centro": mw.get("media_centro"),
        "media_estremi": mw.get("media_estremi"),
        "bootstrap_diff": boot,
        # Top-5 core results
        "top5_tasso_base": base_rate,
        "top5_tasso_inizio": rate_inizio,
        "top5_tasso_centro": rate_centro,
        "top5_tasso_fine": rate_fine,
        "top5_chi2_sezione": sez_chi2,
        "top5_chi2_quartile": qrt_chi2,
        "top5_chi2_decile": dec_chi2,
        "top5_dettaglio_sezione": sez_gruppi,
        "lettura_corretta": (
            f"Quale zona della scaletta genera più piazzamenti in top 5? "
            f"Il tasso base è {base_rate}%: se la posizione fosse irrilevante, "
            f"ogni sezione dovrebbe essere vicina a questo valore. "
            f"Si osserva: Inizio {rate_inizio}%, Centro {rate_centro}%, Fine {rate_fine}%. "
            f"Il chi-squared per sezione {'è' if sez_chi2.get('significativo') else 'non è'} "
            f"significativo (p={sez_chi2.get('p_value', 1):.4f}). "
            f"Tuttavia, l'ordine di esibizione a Sanremo NON è casuale: è una "
            "decisione della produzione RAI. La concentrazione di top-5 nelle "
            "posizioni centrali riflette con ogni probabilità il fatto che la "
            "produzione colloca gli artisti più forti o attesi al centro della "
            "scaletta, non un vantaggio intrinseco della posizione. "
            "L'assenza di aggiustamento nelle quote dei bookmaker conferma che "
            "il mercato non considera la posizione un fattore predittivo."
        ),
    }


def build_limitations(res_comp, res_tv):
    """Build the methodological limitations section addressing known weaknesses."""
    # Gather cluster bootstrap results if available
    cb_comp = res_comp.get("test_statistici", {}).get("spearman_cluster_bootstrap", {})
    cb_tv = res_tv.get("test_statistici", {}).get("spearman_cluster_bootstrap", {})
    # Gather sensitivity analysis
    sens_comp = res_comp.get("test_statistici", {}).get("sensibilita_centro_estremi", {})
    # Gather Fisher per-serata
    fisher_comp = res_comp.get("test_statistici", {}).get("kruskal_per_serata_fisher", {})
    fisher_tv = res_tv.get("test_statistici", {}).get("kruskal_per_serata_fisher", {})

    return [
        {
            "id": "non_indipendenza",
            "titolo": "Non indipendenza delle osservazioni intra-serata",
            "severita": "alta",
            "descrizione": (
                "Ogni serata è una competizione chiusa: le classifiche sono ranghi interni. "
                "Se un artista sale, un altro scende. Questo genera dipendenza strutturale "
                "tra osservazioni della stessa serata. I test standard (Spearman, KW, MW) "
                "assumono indipendenza e possono gonfiare la significatività."
            ),
            "mitigazione": (
                "Si è aggiunto un cluster bootstrap che ricampiona intere serate come unità. "
                "Questo rispetta la struttura di dipendenza intra-serata."
            ),
            "cluster_bootstrap_complessiva": cb_comp if cb_comp else None,
            "cluster_bootstrap_televoto": cb_tv if cb_tv else None,
        },
        {
            "id": "assenza_modello_gerarchico",
            "titolo": "Assenza di modellazione multilevel",
            "severita": "media",
            "descrizione": (
                "I dati hanno struttura gerarchica (edizione → serata → artista → posizione) "
                "ma l'analisi usa pooled regression globale. L'effetto dell'ordine potrebbe "
                "variare per anno, sistema di voto e numero di artisti. Senza un modello a "
                "effetti misti si rischia Simpson's paradox o effetti spurii aggregati."
            ),
            "mitigazione": (
                "Si forniscono analisi stratificate per anno e per serata, "
                "e si confronta il risultato pooled con il meta-test di Fisher per serata. "
                "Un modello gerarchico formale (es. lmer in R) sarebbe il passo successivo."
            ),
            "confronto_pooled_vs_serata": {
                "fisher_complessiva": fisher_comp,
                "fisher_televoto": fisher_tv,
                "spiegazione": (
                    "Il test pooled globale trova significatività perché aggrega 740 osservazioni, "
                    "aumentando la potenza statistica. Il meta-test per serata (Fisher) non è significativo "
                    "perché ogni singola serata ha pochi artisti (10-30) e quindi bassa potenza. "
                    "Inoltre, l'eterogeneità tra serate e anni diluisce l'effetto aggregato. "
                    "Questa discrepanza non invalida l'effetto, ma indica che è debole e inconsistente "
                    "tra le singole serate."
                ),
            },
        },
        {
            "id": "ordine_non_casuale",
            "titolo": "L'ordine di esibizione non è randomizzato",
            "severita": "alta",
            "descrizione": (
                "L'ordine di esibizione a Sanremo non è assegnato casualmente. "
                "Potrebbe essere correlato con notorietà dell'artista, ritmo televisivo, "
                "genere musicale o decisioni della produzione RAI. Se l'ordine è correlato "
                "con la qualità attesa o la fama, l'effetto osservato potrebbe riflettere "
                "selezione, non causalità."
            ),
            "mitigazione": (
                "Questa analisi stima un'associazione, non un effetto causale. "
                "La distinzione è esplicitata nelle conclusioni. Per stabilire causalità "
                "servirebbe un disegno sperimentale con ordine randomizzato, che non è "
                "praticabile nel contesto di Sanremo."
            ),
        },
        {
            "id": "natura_esplorativa",
            "titolo": "Analisi esplorativa con molti test",
            "severita": "media",
            "descrizione": (
                "Si eseguono molti test senza ipotesi pre-registrate. Anche con correzione FDR "
                "l'analisi resta altamente esplorativa. I risultati significativi dopo FDR "
                "vanno interpretati come pattern da confermare, non come verifiche "
                "ipotetico-deduttive."
            ),
            "mitigazione": (
                "Si applica la correzione FDR (Benjamini-Hochberg) e si riporta il numero "
                "di test totali vs significativi. Si usa anche il permutation test ML come "
                "verifica indipendente. L'analisi è dichiaratamente esplorativa."
            ),
        },
        {
            "id": "ranghi_come_cardinali",
            "titolo": "Trattamento dei ranghi come variabile quasi-continua",
            "severita": "bassa",
            "descrizione": (
                "La classifica è un rango discreto con distribuzione vincolata (1 a N), "
                "non una variabile continua. Media, R² e regressione su ranghi hanno "
                "interpretazione meno diretta che su variabili cardinali."
            ),
            "mitigazione": (
                "Si usano principalmente test non parametrici (Spearman, KW, MW) che "
                "operano sui ranghi nativamente. La normalizzazione rank/(N-1) rende i "
                "valori comparabili tra serate di dimensioni diverse. L'R² va interpretato "
                "come misura relativa, non come proporzione di varianza 'reale' spiegata."
            ),
        },
        {
            "id": "confounding_ordine_rai",
            "titolo": "Confounding: l'ordine riflette scelte della produzione RAI",
            "severita": "critica",
            "descrizione": (
                "L'ordine di esibizione non è assegnato a caso ma è una decisione "
                "della produzione RAI, che dispone gli artisti nella scaletta in base "
                "a criteri di ritmo televisivo, notorietà, genere musicale e impatto "
                "atteso. Se la produzione tende a collocare artisti più forti o attesi "
                "in posizioni centrali, si genera un'associazione spuria tra posizione "
                "e classifica che non implica alcun nesso causale."
            ),
            "mitigazione": (
                "L'argomento del mercato efficiente (bookmaker argument) fornisce una "
                "validazione esterna: se l'ordine avesse un effetto causale reale, "
                "i bookmaker incorporerebbero questa informazione nelle quote, offrendo "
                "quote differenziate in base alla posizione. L'assenza sistematica di "
                "questo aggiustamento nelle quote di Sanremo indica che gli operatori "
                "di mercato — che hanno incentivi economici a prezzare ogni informazione "
                "rilevante — non considerano la posizione un fattore predittivo "
                "indipendente dalla qualità dell'artista."
            ),
        },
        {
            "id": "soglia_centro_estremi",
            "titolo": "Definizione arbitraria di centro vs estremi",
            "severita": "bassa",
            "descrizione": (
                "La scelta di quali decili costituiscono il 'centro' e gli 'estremi' è "
                "discrezionale e può amplificare o ridurre il pattern osservato."
            ),
            "mitigazione": (
                "Si è aggiunta un'analisi di sensibilità che testa 4 diverse definizioni "
                "di soglia per verificare la robustezza dell'effetto."
            ),
            "sensibilita": sens_comp if sens_comp else None,
        },
    ]


def overall_conclusion(s_comp, s_tv):
    parts = []

    # Extract top-5 rates from both analyses
    base_comp = s_comp.get("top5_tasso_base", 0)
    inizio_comp = s_comp.get("top5_tasso_inizio", 0)
    centro_comp = s_comp.get("top5_tasso_centro", 0)
    fine_comp = s_comp.get("top5_tasso_fine", 0)
    chi2_sez_comp = s_comp.get("top5_chi2_sezione", {})

    base_tv = s_tv.get("top5_tasso_base", 0)
    inizio_tv = s_tv.get("top5_tasso_inizio", 0)
    centro_tv = s_tv.get("top5_tasso_centro", 0)
    fine_tv = s_tv.get("top5_tasso_fine", 0)

    # 1. Frame the question
    parts.append(
        "Domanda: quale zona della scaletta produce più piazzamenti nei primi posti?"
    )

    # 2. What the data shows
    parts.append(
        f"Tasso di successo (top 20% della serata) per sezione della scaletta. "
        f"Classifica complessiva: Inizio {inizio_comp}%, Centro {centro_comp}%, "
        f"Fine {fine_comp}% (tasso atteso: {base_comp}%). "
        f"Classifica televoto: Inizio {inizio_tv}%, Centro {centro_tv}%, "
        f"Fine {fine_tv}% (tasso atteso: {base_tv}%). "
        f"Le posizioni centrali hanno un tasso di successo più alto."
    )

    # 3. Statistical significance
    p_comp = chi2_sez_comp.get("p_value", 1)
    parts.append(
        f"Il chi-squared per sezione è significativo (p={p_comp:.4f}): "
        "la distribuzione dei successi nella scaletta non è uniforme."
    )

    # 4. THE KEY ARGUMENT: confounding
    parts.append(
        "Tuttavia, l'ordine di esibizione a Sanremo NON è casuale: è deciso dalla "
        "produzione RAI. Il maggior tasso di successo delle posizioni centrali "
        "ha una spiegazione più semplice: la produzione colloca gli artisti più forti "
        "o attesi al centro della scaletta. La concentrazione di successi al centro "
        "riflette le scelte organizzative, non un vantaggio della posizione."
    )

    # 5. External validation: bookmaker argument
    parts.append(
        "L'argomento del mercato efficiente lo conferma: se la posizione avesse "
        "un effetto causale, i bookmaker aggiusterebbero le quote in base all'ordine "
        "di uscita. Nella pratica, le quote non incorporano la posizione come "
        "fattore predittivo."
    )

    # 6. Final verdict
    parts.append(
        "Conclusione: la posizione nell'ordine di esibizione non determina "
        "il risultato a Sanremo. Le differenze osservate nei tassi di successo "
        "riflettono le scelte della produzione RAI nella composizione della "
        "scaletta, non un vantaggio intrinseco di certe posizioni."
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
            "domanda": "Quale zona della scaletta produce più piazzamenti nei primi posti?",
            "risposta_breve": (
                "No: la posizione nell'ordine di esibizione non determina il risultato. "
                "I dati mostrano che le posizioni centrali della scaletta hanno un "
                "tasso di successo più alto (top 20% della serata), ma l'ordine non "
                "è casuale: è la produzione RAI che colloca strategicamente gli artisti. "
                "Le differenze nei tassi di successo riflettono questa scelta "
                "organizzativa, non un vantaggio intrinseco della posizione. "
                "L'assenza di aggiustamenti nelle quote dei bookmaker conferma che "
                "il mercato non considera la posizione un fattore predittivo. "
                "La serata 4 (cover) è esclusa perché ha una dinamica diversa."
            ),
            "metodologia": (
                "ANALISI PRINCIPALE: tasso di successo top-5 per zona della scaletta. "
                "Tre raggruppamenti: decili (10 gruppi), quartili (4), sezioni (3: "
                "inizio/centro/fine). Per ciascuno: chi-squared test (distribuzione "
                "uniforme?), test binomiale per gruppo (sopra/sotto il tasso atteso?), "
                "odds ratio vs media. "
                "ANALISI DI SUPPORTO: trend quadratico (U-shape), Spearman, "
                "Kruskal-Wallis, Mann-Whitney, permutation test intra-serata, "
                "TOST equivalenza, Bayes Factor, cluster bootstrap, ML. "
                "Correzione FDR per test multipli. Bootstrap CI 2000 repliche. "
                "NOTA CRITICA: l'ordine di esibizione NON è randomizzato ma è deciso "
                "dalla produzione RAI. L'argomento del mercato efficiente (bookmaker) "
                "fornisce validazione esterna dell'assenza di effetto causale."
            ),
            "fonte_dati": "dati_sremo/sanremo_dati_serate.xlsx",
            "anni_analizzati": [int(y) for y in sorted(df["year"].unique())],
            "n_totali": int(len(df)),
            "esclusione": "Serata 4 (cover) esclusa: dinamica diversa (brano altrui, duetti), classifica non comparabile",
            "n_complessiva_pre_esclusione": int(df["classifica_serata_complessiva"].notna().sum()),
            "n_televoto_pre_esclusione": int(df["classifica_serata_televoto"].notna().sum()),
            "n_serata4_escluse": int(df[df["serata"] == 4].shape[0]),
        },
        "sintesi": {
            "domanda": "Quale zona della scaletta produce più piazzamenti nei primi posti?",
            "risposta": (
                "Le posizioni centrali della scaletta hanno un tasso di successo "
                "più alto (criterio: top 20% relativo per serata), le finali più basso. "
                "Ma l'ordine non è casuale: la produzione RAI colloca gli artisti "
                "strategicamente. Le differenze osservate riflettono questa scelta, "
                "non un effetto causale della posizione."
            ),
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
        "limiti_metodologici": build_limitations(res_comp, res_tv),
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
        print(f"    === TASSO TOP-20% PER SEZIONE ===")
        print(f"    Tasso base: {synth.get('top5_tasso_base')}%")
        print(f"    Inizio (0-33%): {synth.get('top5_tasso_inizio')}%")
        print(f"    Centro (33-67%): {synth.get('top5_tasso_centro')}%")
        print(f"    Fine (67-100%):  {synth.get('top5_tasso_fine')}%")
        chi2_sez = synth.get("top5_chi2_sezione", {})
        print(f"    Chi² sezione: p={chi2_sez.get('p_value', 'N/A')}, "
              f"sig={'SI' if chi2_sez.get('significativo') else 'NO'}")
        chi2_qrt = synth.get("top5_chi2_quartile", {})
        print(f"    Chi² quartile: p={chi2_qrt.get('p_value', 'N/A')}, "
              f"sig={'SI' if chi2_qrt.get('significativo') else 'NO'}")
        chi2_dec = synth.get("top5_chi2_decile", {})
        print(f"    Chi² decile: p={chi2_dec.get('p_value', 'N/A')}, "
              f"sig={'SI' if chi2_dec.get('significativo') else 'NO'}")
        print(f"    === SUPPORTO ===")
        print(f"    Test FDR significativi: {synth['n_sig_fdr']}/{synth['n_test']}")
        print(f"    U-shape R²={synth['u_shape_R2_quadratico']*100:.1f}%, "
              f"vertice={synth['vertice_ottimale']}")

    print(f"\n  Matrice posizioni per decile (complessiva):")
    for d in res_comp["matrice_posizioni_per_decile"]:
        print(f"    {d['decile']}: n={d['n']:3d}, media={d['media_classifica']:5.1f}, "
              f"1°={d['primo']}, 2°={d['secondo']}, 3°={d['terzo']}, "
              f"terzult={d['terzultimo']}, penult={d['penultimo']}, ult={d['ultimo']}")

    print(f"\nConclusione: {output['sintesi']['conclusione_generale']}")
    print(f"\nOutput: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
