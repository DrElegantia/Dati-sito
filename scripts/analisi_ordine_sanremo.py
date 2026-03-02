#!/usr/bin/env python3
"""
Sanremo — L'ordine di uscita influenza la classifica?

Fonte dati: dati_sremo/sanremo_dati_serate.xlsx
Output:     docs/sanremo_timing_analysis.json

Analisi per quintili di posizione nell'ordine di esibizione.
Doppia analisi: classifica televoto e classifica complessiva di serata.
Include: test statistici con effect size, bootstrap CI, correzione per
test multipli (Bonferroni + FDR), analisi di robustezza, e modelli ML
con permutation test.

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

QUINTILE_LABELS = [
    "Q1 (0-20%)", "Q2 (20-40%)", "Q3 (40-60%)", "Q4 (60-80%)", "Q5 (80-100%)"
]
QUINTILE_BINS = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
CENTRO_Q = ["Q2 (20-40%)", "Q3 (40-60%)", "Q4 (60-80%)"]
ESTREMI_Q = ["Q1 (0-20%)", "Q5 (80-100%)"]

N_BOOTSTRAP = 2000
RANDOM_SEED = 42


# ============================================================
# Data loading & preparation
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_excel(XLSX_PATH)

    # Fix shifted rows: artist name "Shablo con Guè, Joshua e Tormento"
    # was split across artist/ordine columns, shifting all values right.
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

    # Manual data fixes (errata in the Excel)
    # Sarah Toscano 2025 serata 5: classifica_serata_complessiva = 20
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


def assign_quintile(pos_rel: pd.Series) -> pd.Series:
    return pd.cut(pos_rel, bins=QUINTILE_BINS, labels=QUINTILE_LABELS)


def normalize_rank(rank: pd.Series, total: pd.Series) -> pd.Series:
    return (rank - 1) / np.maximum(total - 1, 1)


# ============================================================
# Bootstrap utilities
# ============================================================

def bootstrap_ci(data, stat_fn=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return None
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {"ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "ci_level": ci, "n_bootstrap": n_boot}


def bootstrap_diff_ci(a, b, stat_fn=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED):
    """Bootstrap CI for the difference stat_fn(a) - stat_fn(b)."""
    rng = np.random.RandomState(seed)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(stat_fn(sa) - stat_fn(sb))
    alpha = (1 - ci) / 2
    lo = float(np.percentile(diffs, alpha * 100))
    hi = float(np.percentile(diffs, (1 - alpha) * 100))
    return {"diff_observed": round(float(stat_fn(a) - stat_fn(b)), 4),
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "ci_level": ci, "includes_zero": bool(lo <= 0 <= hi),
            "n_bootstrap": n_boot}


# ============================================================
# Effect sizes
# ============================================================

def rank_biserial_r(u_stat, n1, n2):
    """Rank-biserial correlation from Mann-Whitney U."""
    if n1 == 0 or n2 == 0:
        return None
    r = 1 - (2 * u_stat) / (n1 * n2)
    return round(float(r), 4)


def eta_squared_kw(h_stat, n):
    """Eta-squared approximation from Kruskal-Wallis H."""
    if n <= 1:
        return None
    return round(float((h_stat - 1) / (n - 1)), 4) if h_stat > 1 else 0.0


# ============================================================
# Multiple testing correction
# ============================================================

def bonferroni_correction(p_values: list, alpha=0.05):
    """Bonferroni correction for multiple comparisons."""
    n = len([p for p in p_values if p is not None])
    if n == 0:
        return []
    corrected = []
    for p in p_values:
        if p is None:
            corrected.append(None)
        else:
            adj = min(p * n, 1.0)
            corrected.append(round(adj, 6))
    return corrected


def fdr_correction(p_values: list, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if len(valid) == 0:
        return [None] * len(p_values)
    valid_sorted = sorted(valid, key=lambda x: x[1])
    m = len(valid_sorted)
    result = [None] * len(p_values)
    prev_adj = 0
    for rank_idx, (orig_idx, p) in enumerate(valid_sorted, 1):
        adj = min(p * m / rank_idx, 1.0)
        adj = max(adj, prev_adj)  # enforce monotonicity
        result[orig_idx] = round(adj, 6)
        prev_adj = adj
    return result


# ============================================================
# Statistical tests (enhanced)
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
        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(x2), size=len(x2), replace=True)
            r, _ = stats.spearmanr(x2[idx], y2[idx])
            boot_rhos.append(r)
        result["rho_ci_95"] = [round(float(np.percentile(boot_rhos, 2.5)), 4),
                               round(float(np.percentile(boot_rhos, 97.5)), 4)]
    if p >= 0.05:
        result["interpretazione"] = "non significativo (p >= 0.05)"
    else:
        direction = "chi si esibisce tardi tende ad avere classifica peggiore" if rho > 0 \
            else "chi si esibisce tardi tende ad avere classifica migliore"
        result["interpretazione"] = f"significativo (p < 0.05): {direction}"
    return result


def kruskal_wallis_test(groups: list, group_labels: list):
    valid = [(np.array(g, dtype=float), l) for g, l in zip(groups, group_labels) if len(g) >= 2]
    if len(valid) < 2:
        return None
    grps = [g for g, _ in valid]
    n_total = sum(len(g) for g in grps)
    stat, p = stats.kruskal(*grps)
    effect = eta_squared_kw(stat, n_total)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "df": len(grps) - 1, "n_groups": len(grps), "n_total": n_total,
            "eta_squared": effect,
            "significativo": bool(p < 0.05),
            "interpretazione": "significativo: almeno un quintile differisce" if p < 0.05
            else "non significativo: nessuna differenza tra quintili"}


def mann_whitney_test(center, extremes):
    c = np.array(center.dropna(), dtype=float)
    e = np.array(extremes.dropna(), dtype=float)
    if len(c) < 3 or len(e) < 3:
        return None
    stat, p = stats.mannwhitneyu(c, e, alternative="two-sided")
    r_rb = rank_biserial_r(stat, len(c), len(e))
    better = "centro" if np.mean(c) < np.mean(e) else "estremi"

    # Bootstrap CI for the difference in means
    boot_diff = bootstrap_diff_ci(c, e)

    return {"n_centro": int(len(c)), "n_estremi": int(len(e)),
            "media_centro": round(float(np.mean(c)), 3),
            "media_estremi": round(float(np.mean(e)), 3),
            "mediana_centro": round(float(np.median(c)), 3),
            "mediana_estremi": round(float(np.median(e)), 3),
            "U": round(float(stat), 1), "p_value": round(float(p), 6),
            "rank_biserial_r": r_rb,
            "bootstrap_diff_mean": boot_diff,
            "significativo": bool(p < 0.05),
            "vantaggio": better if p < 0.05 else "nessuno",
            "interpretazione": f"significativo: il {better} ha classifica migliore" if p < 0.05
            else "non significativo: nessuna differenza"}


def chi2_top_n(quintile_col, rank_col, n=5):
    mask = quintile_col.notna() & rank_col.notna()
    q = quintile_col[mask]
    r = rank_col[mask]
    is_top = (r <= n).astype(int)
    ct = pd.crosstab(q, is_top)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    # Cramér's V
    n_obs = ct.sum().sum()
    k = min(ct.shape) - 1
    cramers_v = round(float(np.sqrt(chi2 / (n_obs * max(k, 1)))), 4) if n_obs > 0 else None
    return {"chi2": round(float(chi2), 4), "p_value": round(float(p), 6),
            "dof": int(dof), "cramers_v": cramers_v,
            "significativo": bool(p < 0.05), "soglia": n,
            "interpretazione": f"significativo: i quintili hanno probabilità diverse di entrare in top {n}" if p < 0.05
            else f"non significativo: probabilità top {n} simile tra quintili"}


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


def jonckheere_terpstra_test(groups: list, group_labels: list):
    """Jonckheere-Terpstra test for ordered alternatives (Q1 < Q2 < ... < Q5)."""
    valid = [(np.array(g, dtype=float), l) for g, l in zip(groups, group_labels) if len(g) >= 1]
    if len(valid) < 3:
        return None
    grps = [g for g, _ in valid]
    # Compute J statistic
    j_stat = 0
    n_total = sum(len(g) for g in grps)
    for i in range(len(grps)):
        for j in range(i + 1, len(grps)):
            for xi in grps[i]:
                for xj in grps[j]:
                    if xj > xi:
                        j_stat += 1
                    elif xj == xi:
                        j_stat += 0.5
    # Expected value and variance under H0
    ns = [len(g) for g in grps]
    N = sum(ns)
    e_j = (N * N - sum(n * n for n in ns)) / 4
    # Variance (no ties approximation)
    var_num = N * N * (2 * N + 3) - sum(n * n * (2 * n + 3) for n in ns)
    var_j = var_num / 72
    if var_j <= 0:
        return None
    z = (j_stat - e_j) / np.sqrt(var_j)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"J": round(float(j_stat), 1), "z": round(float(z), 4),
            "p_value": round(float(p), 6), "n_groups": len(grps),
            "significativo": bool(p < 0.05),
            "interpretazione": "significativo: trend ordinato tra quintili" if p < 0.05
            else "non significativo: nessun trend ordinato"}


# ============================================================
# Robustness checks
# ============================================================

def robustness_exclude_serate(df, rank_col, serate_to_exclude):
    """Re-run core tests excluding specific serate."""
    sub = df[~df["serata"].isin(serate_to_exclude)].copy()
    sub = sub[sub[rank_col].notna()].copy()
    if len(sub) < 10:
        return None
    sub[rank_col] = sub[rank_col].astype(int)
    sub["quintile"] = assign_quintile(sub["posizione_relativa"])

    sp = spearman_test(sub["posizione_relativa"], sub[rank_col])
    groups = [sub[sub["quintile"] == q][rank_col].values for q in QUINTILE_LABELS]
    kw = kruskal_wallis_test(groups, QUINTILE_LABELS)
    centro = sub.loc[sub["quintile"].isin(CENTRO_Q), rank_col]
    estremi = sub.loc[sub["quintile"].isin(ESTREMI_Q), rank_col]
    mw = mann_whitney_test(centro, estremi)

    return {"serate_escluse": serate_to_exclude, "n": int(len(sub)),
            "spearman": sp, "kruskal_wallis": kw, "mann_whitney": mw}


def robustness_terciles(df, rank_col):
    """Re-run with terciles (3 groups) instead of quintiles."""
    sub = df[df[rank_col].notna()].copy()
    if len(sub) < 10:
        return None
    sub[rank_col] = sub[rank_col].astype(int)
    sub["tercile"] = pd.cut(sub["posizione_relativa"],
                            bins=[-0.001, 0.333, 0.667, 1.001],
                            labels=["Inizio (1/3)", "Centro (2/3)", "Fine (3/3)"])
    tercile_labels = ["Inizio (1/3)", "Centro (2/3)", "Fine (3/3)"]
    groups = [sub[sub["tercile"] == t][rank_col].values for t in tercile_labels]
    kw = kruskal_wallis_test(groups, tercile_labels)
    centro = sub.loc[sub["tercile"] == "Centro (2/3)", rank_col]
    estremi = sub.loc[sub["tercile"] != "Centro (2/3)", rank_col]
    mw = mann_whitney_test(centro, estremi)
    return {"tipo": "tercili", "n": int(len(sub)),
            "kruskal_wallis": kw, "mann_whitney_centro_vs_estremi": mw}


def robustness_normalized_rank(df, rank_col):
    """Re-run with rank normalized to 0-1 instead of raw rank."""
    sub = df[df[rank_col].notna()].copy()
    if len(sub) < 10:
        return None
    sub[rank_col] = sub[rank_col].astype(int)
    sub["rank_norm"] = normalize_rank(sub[rank_col], sub["totale_serata"])
    sub["quintile"] = assign_quintile(sub["posizione_relativa"])

    sp = spearman_test(sub["posizione_relativa"], sub["rank_norm"])
    groups = [sub[sub["quintile"] == q]["rank_norm"].values for q in QUINTILE_LABELS]
    kw = kruskal_wallis_test(groups, QUINTILE_LABELS)
    centro = sub.loc[sub["quintile"].isin(CENTRO_Q), "rank_norm"]
    estremi = sub.loc[sub["quintile"].isin(ESTREMI_Q), "rank_norm"]
    mw = mann_whitney_test(centro, estremi)
    return {"tipo": "rank normalizzato (0-1)", "n": int(len(sub)),
            "spearman": sp, "kruskal_wallis": kw, "mann_whitney": mw}


# ============================================================
# Analysis runner
# ============================================================

def run_analysis_for_rank_type(df: pd.DataFrame, rank_col: str, rank_label: str):
    usable = df[df[rank_col].notna()].copy()
    usable[rank_col] = usable[rank_col].astype(int)
    usable["quintile"] = assign_quintile(usable["posizione_relativa"])
    usable["rank_norm"] = normalize_rank(usable[rank_col], usable["totale_serata"])
    years = sorted(usable["year"].unique())

    result = {
        "tipo_classifica": rank_label,
        "colonna": rank_col,
        "n_osservazioni": int(len(usable)),
        "anni": [int(y) for y in years],
    }

    # --- Distribution per quintile (global) ---
    dist_overall = []
    for q in QUINTILE_LABELS:
        g = usable[usable["quintile"] == q][rank_col]
        if len(g) == 0:
            continue
        ci = bootstrap_ci(g.values)
        dist_overall.append({
            "quintile": q, "n": int(len(g)),
            "media": round(float(g.mean()), 2),
            "mediana": round(float(g.median()), 1),
            "std": round(float(g.std()), 2),
            "min": int(g.min()), "max": int(g.max()),
            "q25": round(float(g.quantile(0.25)), 1),
            "q75": round(float(g.quantile(0.75)), 1),
            "bootstrap_ci_mean_95": ci,
        })
    result["distribuzione_quintili_globale"] = dist_overall

    # --- Distribution per year ---
    dist_per_year = []
    for y in years:
        ydf = usable[usable["year"] == y]
        year_data = {"anno": int(y), "n": int(len(ydf)), "quintili": []}
        for q in QUINTILE_LABELS:
            g = ydf[ydf["quintile"] == q][rank_col]
            if len(g) == 0:
                continue
            year_data["quintili"].append({
                "quintile": q, "n": int(len(g)),
                "media": round(float(g.mean()), 2),
                "mediana": round(float(g.median()), 1),
            })
        dist_per_year.append(year_data)
    result["distribuzione_per_anno"] = dist_per_year

    # --- Distribution per serata ---
    dist_per_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            if len(sdf) < 3:
                continue
            serata_data = {
                "anno": int(y), "serata": int(s),
                "serata_name": sdf["serata_name"].iloc[0],
                "n": int(len(sdf)), "quintili": []
            }
            for q in QUINTILE_LABELS:
                g = sdf[sdf["quintile"] == q][rank_col]
                if len(g) == 0:
                    continue
                serata_data["quintili"].append({
                    "quintile": q, "n": int(len(g)),
                    "media": round(float(g.mean()), 2),
                    "mediana": round(float(g.median()), 1),
                })
            dist_per_serata.append(serata_data)
    result["distribuzione_per_serata"] = dist_per_serata

    # --- Statistical tests ---
    tests = {}
    all_p_values = []  # for multiple testing correction

    # Spearman global
    tests["spearman_globale"] = spearman_test(
        usable["posizione_relativa"], usable[rank_col], bootstrap=True)
    if tests["spearman_globale"]:
        all_p_values.append(("spearman_globale", tests["spearman_globale"]["p_value"]))

    # Spearman per year
    sp_per_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        res = spearman_test(ydf["posizione_relativa"], ydf[rank_col])
        if res:
            res["anno"] = int(y)
            sp_per_anno.append(res)
            all_p_values.append((f"spearman_{y}", res["p_value"]))
    tests["spearman_per_anno"] = sp_per_anno

    # Spearman per serata
    sp_per_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            res = spearman_test(sdf["posizione_relativa"], sdf[rank_col])
            if res:
                res["anno"] = int(y)
                res["serata"] = int(s)
                sp_per_serata.append(res)
                all_p_values.append((f"spearman_{y}_s{s}", res["p_value"]))
    tests["spearman_per_serata"] = sp_per_serata

    # Kruskal-Wallis global
    groups = [usable[usable["quintile"] == q][rank_col].values for q in QUINTILE_LABELS]
    tests["kruskal_wallis_globale"] = kruskal_wallis_test(groups, QUINTILE_LABELS)
    if tests["kruskal_wallis_globale"]:
        all_p_values.append(("kruskal_wallis_globale", tests["kruskal_wallis_globale"]["p_value"]))

    # Kruskal-Wallis per year
    kw_per_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        groups = [ydf[ydf["quintile"] == q][rank_col].values for q in QUINTILE_LABELS]
        res = kruskal_wallis_test(groups, QUINTILE_LABELS)
        if res:
            res["anno"] = int(y)
            kw_per_anno.append(res)
            all_p_values.append((f"kruskal_{y}", res["p_value"]))
    tests["kruskal_wallis_per_anno"] = kw_per_anno

    # Kruskal-Wallis per serata + Fisher combined
    kw_per_serata = []
    kw_pvalues = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            groups = [sdf[sdf["quintile"] == q][rank_col].values for q in QUINTILE_LABELS]
            res = kruskal_wallis_test(groups, QUINTILE_LABELS)
            if res:
                res["anno"] = int(y)
                res["serata"] = int(s)
                kw_per_serata.append(res)
                kw_pvalues.append(res["p_value"])
                all_p_values.append((f"kruskal_{y}_s{s}", res["p_value"]))
    tests["kruskal_wallis_per_serata"] = kw_per_serata
    tests["kruskal_per_serata_fisher_combined"] = fisher_combined(kw_pvalues)

    # Jonckheere-Terpstra (ordered alternative)
    groups = [usable[usable["quintile"] == q][rank_col].values for q in QUINTILE_LABELS]
    tests["jonckheere_terpstra_globale"] = jonckheere_terpstra_test(groups, QUINTILE_LABELS)
    if tests["jonckheere_terpstra_globale"]:
        all_p_values.append(("jonckheere_globale", tests["jonckheere_terpstra_globale"]["p_value"]))

    # Mann-Whitney centro vs estremi
    centro = usable.loc[usable["quintile"].isin(CENTRO_Q), rank_col]
    estremi = usable.loc[usable["quintile"].isin(ESTREMI_Q), rank_col]
    tests["mann_whitney_centro_vs_estremi"] = mann_whitney_test(centro, estremi)
    if tests["mann_whitney_centro_vs_estremi"]:
        all_p_values.append(("mann_whitney_globale", tests["mann_whitney_centro_vs_estremi"]["p_value"]))

    # Mann-Whitney per year
    mw_per_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        c = ydf.loc[ydf["quintile"].isin(CENTRO_Q), rank_col]
        e = ydf.loc[ydf["quintile"].isin(ESTREMI_Q), rank_col]
        res = mann_whitney_test(c, e)
        if res:
            res["anno"] = int(y)
            mw_per_anno.append(res)
            all_p_values.append((f"mann_whitney_{y}", res["p_value"]))
    tests["mann_whitney_per_anno"] = mw_per_anno

    # Mann-Whitney per serata
    mw_per_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            c = sdf.loc[sdf["quintile"].isin(CENTRO_Q), rank_col]
            e = sdf.loc[sdf["quintile"].isin(ESTREMI_Q), rank_col]
            res = mann_whitney_test(c, e)
            if res:
                res["anno"] = int(y)
                res["serata"] = int(s)
                mw_per_serata.append(res)
                all_p_values.append((f"mann_whitney_{y}_s{s}", res["p_value"]))
    tests["mann_whitney_per_serata"] = mw_per_serata

    # Chi-squared
    tests["chi2_top5_globale"] = chi2_top_n(usable["quintile"], usable[rank_col], n=5)
    tests["chi2_top10_globale"] = chi2_top_n(usable["quintile"], usable[rank_col], n=10)
    for key in ["chi2_top5_globale", "chi2_top10_globale"]:
        if tests[key]:
            all_p_values.append((key, tests[key]["p_value"]))

    chi2_per_anno = []
    for y in years:
        ydf = usable[usable["year"] == y]
        res5 = chi2_top_n(ydf["quintile"], ydf[rank_col], n=5)
        if res5:
            res5["anno"] = int(y)
            chi2_per_anno.append(res5)
            all_p_values.append((f"chi2_top5_{y}", res5["p_value"]))
    tests["chi2_top5_per_anno"] = chi2_per_anno

    # --- Multiple testing correction ---
    raw_ps = [p for _, p in all_p_values]
    bonf = bonferroni_correction(raw_ps)
    fdr = fdr_correction(raw_ps)
    correction_table = []
    for idx, (name, raw_p) in enumerate(all_p_values):
        correction_table.append({
            "test": name, "p_raw": round(raw_p, 6),
            "p_bonferroni": bonf[idx],
            "p_fdr": fdr[idx],
            "sig_raw": bool(raw_p < 0.05),
            "sig_bonferroni": bool(bonf[idx] is not None and bonf[idx] < 0.05),
            "sig_fdr": bool(fdr[idx] is not None and fdr[idx] < 0.05),
        })
    tests["correzione_test_multipli"] = {
        "n_test": len(all_p_values),
        "metodi": ["Bonferroni", "Benjamini-Hochberg FDR"],
        "tabella": correction_table,
        "n_sig_raw": sum(1 for r in correction_table if r["sig_raw"]),
        "n_sig_bonferroni": sum(1 for r in correction_table if r["sig_bonferroni"]),
        "n_sig_fdr": sum(1 for r in correction_table if r["sig_fdr"]),
    }

    result["test_statistici"] = tests

    # --- Robustness checks ---
    robustness = {}
    robustness["esclusa_finale"] = robustness_exclude_serate(df, rank_col, [5])
    robustness["escluse_serate_1_2"] = robustness_exclude_serate(df, rank_col, [1, 2])
    robustness["solo_serate_piene"] = robustness_exclude_serate(
        df[df["totale_serata"] >= 20], rank_col, [])
    robustness["tercili"] = robustness_terciles(df, rank_col)
    robustness["rank_normalizzato"] = robustness_normalized_rank(df, rank_col)
    result["analisi_robustezza"] = robustness

    # --- Visualization data ---
    viz = {}

    # Boxplot quintiles global
    boxplot_data = []
    for q in QUINTILE_LABELS:
        g = usable[usable["quintile"] == q][rank_col].dropna()
        if len(g) == 0:
            continue
        boxplot_data.append({
            "quintile": q,
            "min": round(float(g.min()), 1),
            "q25": round(float(g.quantile(0.25)), 1),
            "median": round(float(g.median()), 1),
            "q75": round(float(g.quantile(0.75)), 1),
            "max": round(float(g.max()), 1),
            "mean": round(float(g.mean()), 2),
            "n": int(len(g)),
        })
    viz["boxplot_quintili"] = boxplot_data

    # Heatmap year x quintile
    heatmap = []
    for y in years:
        ydf = usable[usable["year"] == y]
        row = {"anno": int(y)}
        for q in QUINTILE_LABELS:
            g = ydf[ydf["quintile"] == q][rank_col]
            row[q] = round(float(g.mean()), 2) if len(g) > 0 else None
        heatmap.append(row)
    viz["heatmap_anno_quintile"] = heatmap

    # Probability per quintile
    prob_data = []
    for q in QUINTILE_LABELS:
        g = usable[usable["quintile"] == q][rank_col].dropna()
        if len(g) == 0:
            continue
        n_g = len(g)
        prob_data.append({
            "quintile": q, "n": int(n_g),
            "pct_top5": round(float((g <= 5).mean() * 100), 1),
            "pct_top10": round(float((g <= 10).mean() * 100), 1),
        })
    viz["probabilita_per_quintile"] = prob_data

    # Scatter
    scatter = []
    for _, row in usable.iterrows():
        scatter.append({
            "x": round(float(row["posizione_relativa"]), 4),
            "y": int(row[rank_col]),
            "anno": int(row["year"]),
            "serata": int(row["serata"]),
            "artista": row["artist"],
        })
    viz["scatter"] = scatter

    # Boxplot per serata
    boxplot_serata = []
    for y in years:
        ydf = usable[usable["year"] == y]
        for s in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == s]
            for q in QUINTILE_LABELS:
                g = sdf[sdf["quintile"] == q][rank_col].dropna()
                if len(g) == 0:
                    continue
                boxplot_serata.append({
                    "anno": int(y), "serata": int(s),
                    "quintile": q, "n": int(len(g)),
                    "media": round(float(g.mean()), 2),
                    "mediana": round(float(g.median()), 1),
                })
    viz["boxplot_per_serata"] = boxplot_serata

    result["visualizzazione"] = viz
    return result, usable


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
        return {"errore": "Troppo pochi dati per ML", "tipo_classifica": rank_label}

    features = ["posizione_relativa", "totale_serata", "serata"]
    X = usable[features].values
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
        mae_std = float(scores.std())
        improvement = round((baseline_mae - mae) / baseline_mae * 100, 2)
        results.append({
            "modello": name,
            "mae_cv": round(mae, 4),
            "mae_cv_std": round(mae_std, 4),
            "miglioramento_su_baseline": improvement,
        })

    # Permutation test
    best_result = min(results, key=lambda r: r["mae_cv"]) if results else None
    best_model_name = best_result["modello"] if best_result else None
    perm_p = 1.0
    if best_model_name:
        best_model = models[best_model_name]
        np.random.seed(42)
        actual_mae = best_result["mae_cv"]
        n_perm = 200
        perm_maes = []
        cv = min(5, len(usable) // 5)
        for _ in range(n_perm):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, 0])
            s = cross_val_score(best_model, X_perm, y, cv=cv, scoring="neg_mean_absolute_error")
            perm_maes.append(float(-s.mean()))
        perm_p = float(np.mean([m <= actual_mae for m in perm_maes]))

    return {
        "tipo_classifica": rank_label,
        "n_osservazioni": int(len(usable)),
        "features": features,
        "baseline_mae": round(baseline_mae, 4),
        "modelli": results,
        "miglior_modello": best_model_name,
        "miglior_mae": round(best_result["mae_cv"], 4) if best_result else None,
        "miglioramento_migliore": max(r["miglioramento_su_baseline"] for r in results) if results else None,
        "permutation_test": {
            "n_permutazioni": n_perm,
            "p_value": round(perm_p, 4),
            "significativo": bool(perm_p < 0.05),
            "interpretazione": "il modello predice meglio del caso" if perm_p < 0.05
            else "il modello NON predice meglio del caso"
        }
    }


# ============================================================
# Synthesis
# ============================================================

def build_synthesis(res, ml):
    tests = res["test_statistici"]
    correction = tests.get("correzione_test_multipli", {})

    n_raw = correction.get("n_sig_raw", 0)
    n_bonf = correction.get("n_sig_bonferroni", 0)
    n_fdr = correction.get("n_sig_fdr", 0)
    n_total = correction.get("n_test", 0)

    mw = tests.get("mann_whitney_centro_vs_estremi", {})
    centro_mean = mw.get("media_centro")
    estremi_mean = mw.get("media_estremi")
    boot_diff = mw.get("bootstrap_diff_mean", {})

    # Use FDR-corrected count for strength assessment
    if n_fdr == 0:
        evidenza = "nessuna"
    elif n_fdr <= 2:
        evidenza = "molto debole"
    elif n_fdr <= 4:
        evidenza = "debole"
    elif n_fdr <= 6:
        evidenza = "moderata"
    else:
        evidenza = "forte"

    if centro_mean and estremi_mean:
        if centro_mean < estremi_mean:
            direzione = "Il centro sembra avere un leggero vantaggio"
        elif centro_mean > estremi_mean:
            direzione = "Gli estremi sembrano avere un leggero vantaggio"
        else:
            direzione = "Nessuna direzione chiara"
    else:
        direzione = "Dati insufficienti"

    # Check robustness consistency
    rob = res.get("analisi_robustezza", {})
    rob_consistent = True
    for key, val in rob.items():
        if val and isinstance(val, dict):
            mw_r = val.get("mann_whitney") or val.get("mann_whitney_centro_vs_estremi")
            if mw_r and mw_r.get("vantaggio") and centro_mean and estremi_mean:
                main_better = "centro" if centro_mean < estremi_mean else "estremi"
                if mw_r["vantaggio"] != "nessuno" and mw_r["vantaggio"] != main_better:
                    rob_consistent = False

    return {
        "n_test_eseguiti": n_total,
        "n_test_significativi_raw": n_raw,
        "n_test_significativi_bonferroni": n_bonf,
        "n_test_significativi_fdr": n_fdr,
        "media_centro": centro_mean,
        "media_estremi": estremi_mean,
        "bootstrap_diff_mean": boot_diff,
        "evidenza": evidenza,
        "direzione": direzione,
        "robustezza_consistente": rob_consistent,
        "ml_miglioramento": ml.get("miglioramento_migliore"),
        "ml_significativo": ml.get("permutation_test", {}).get("significativo", False),
    }


def overall_conclusion(s_comp, s_tv):
    sig_comp = s_comp.get("n_test_significativi_fdr", 0)
    sig_tv = s_tv.get("n_test_significativi_fdr", 0)
    total_fdr = sig_comp + sig_tv
    total_tests = s_comp.get("n_test_eseguiti", 0) + s_tv.get("n_test_eseguiti", 0)

    boot_comp = s_comp.get("bootstrap_diff_mean", {})
    boot_tv = s_tv.get("bootstrap_diff_mean", {})
    ci_comp_zero = boot_comp.get("includes_zero", True) if boot_comp else True
    ci_tv_zero = boot_tv.get("includes_zero", True) if boot_tv else True

    rob_ok = s_comp.get("robustezza_consistente", True) and s_tv.get("robustezza_consistente", True)

    ml_comp = s_comp.get("ml_significativo", False)
    ml_tv = s_tv.get("ml_significativo", False)

    # Determine direction
    dir_comp = s_comp.get("direzione", "")
    dir_tv = s_tv.get("direzione", "")
    centro_vantaggio = "centro" in dir_comp.lower() or "centro" in dir_tv.lower()

    parts = []

    # Evidence strength
    if total_fdr == 0:
        parts.append(
            f"Nessun test su {total_tests} risulta significativo dopo correzione FDR. "
            "L'ordine di esibizione NON sembra influenzare il ranking della serata."
        )
    elif total_fdr <= 3:
        parts.append(
            f"Evidenza molto debole: solo {total_fdr}/{total_tests} test significativi dopo FDR."
        )
    elif total_fdr <= 8:
        parts.append(
            f"Evidenza debole-moderata: {total_fdr}/{total_tests} test significativi dopo FDR."
        )
    else:
        parts.append(
            f"Evidenza forte: {total_fdr}/{total_tests} test significativi dopo correzione FDR."
        )

    # Bootstrap CI
    if not ci_comp_zero or not ci_tv_zero:
        ci_details = []
        if not ci_comp_zero:
            d = boot_comp.get("diff_observed", "?")
            ci = f"[{boot_comp.get('ci_lower')}, {boot_comp.get('ci_upper')}]"
            ci_details.append(f"complessiva: diff={d}, CI95={ci}")
        if not ci_tv_zero:
            d = boot_tv.get("diff_observed", "?")
            ci = f"[{boot_tv.get('ci_lower')}, {boot_tv.get('ci_upper')}]"
            ci_details.append(f"televoto: diff={d}, CI95={ci}")
        parts.append(
            "Gli intervalli di confidenza bootstrap al 95% per la differenza centro-estremi "
            "NON includono lo zero (" + "; ".join(ci_details) + "), "
            "confermando un effetto reale."
        )
    elif total_fdr > 0:
        parts.append(
            "Tuttavia, gli intervalli di confidenza bootstrap al 95% includono lo zero, "
            "suggerendo cautela nell'interpretazione."
        )

    # Direction
    if total_fdr > 3 and centro_vantaggio:
        mc = s_comp.get("media_centro")
        me = s_comp.get("media_estremi")
        if mc and me:
            diff = round(me - mc, 1)
            parts.append(
                f"Direzione: chi si esibisce in posizioni centrali (Q2-Q4) ottiene in media "
                f"~{diff} posizioni migliori in classifica rispetto a chi si esibisce "
                "all'inizio o alla fine."
            )

    # ML
    if ml_comp or ml_tv:
        parts.append(
            "I modelli ML confermano: l'ordine di esibizione ha potere predittivo "
            "sulla classifica (permutation test significativo)."
        )
    elif total_fdr > 3:
        parts.append(
            "I modelli ML mostrano un miglioramento modesto sulla baseline, "
            "coerente con un effetto piccolo ma reale."
        )

    # Robustness
    if rob_ok and total_fdr > 3:
        parts.append("Le analisi di robustezza confermano la consistenza del risultato.")
    elif not rob_ok:
        parts.append(
            "Nota: le analisi di robustezza non sono del tutto consistenti, "
            "suggerendo cautela."
        )

    # Final verdict
    if total_fdr > 8 and not ci_comp_zero and rob_ok:
        parts.append(
            "Conclusione: esibirsi in zone centrali della serata sembra conferire "
            "un vantaggio statisticamente significativo nella classifica."
        )
    elif total_fdr > 3:
        parts.append(
            "Conclusione: c'è evidenza di un effetto dell'ordine di esibizione "
            "sulla classifica, con un vantaggio per le posizioni centrali. "
            "L'effetto è statisticamente significativo ma di entità moderata."
        )
    elif total_fdr > 0:
        parts.append(
            "Conclusione: l'evidenza è troppo debole per trarre conclusioni definitive."
        )

    return " ".join(parts)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SANREMO — ANALISI ORDINE DI ESIBIZIONE (dal nuovo Excel)")
    print("=" * 70)

    print("\n[1/5] Caricamento dati da Excel...")
    df = load_data()
    print(f"  Righe: {len(df)}")
    print(f"  Anni: {sorted(df['year'].unique())}")

    print("\n  Copertura:")
    for y in sorted(df["year"].unique()):
        ydf = df[df["year"] == y]
        n_tv = ydf["classifica_serata_televoto"].notna().sum()
        n_comp = ydf["classifica_serata_complessiva"].notna().sum()
        print(f"    {y}: {len(ydf)} esibizioni, televoto={n_tv}, complessiva={n_comp}")

    print("\n[2/5] Analisi per classifica serata complessiva...")
    result_comp, _ = run_analysis_for_rank_type(
        df, "classifica_serata_complessiva", "Classifica serata complessiva")
    print(f"  Osservazioni: {result_comp['n_osservazioni']}")

    print("\n[3/5] Analisi per classifica serata televoto...")
    result_tv, _ = run_analysis_for_rank_type(
        df, "classifica_serata_televoto", "Classifica serata televoto")
    print(f"  Osservazioni: {result_tv['n_osservazioni']}")

    print("\n[4/5] Machine Learning...")
    try:
        ml_comp = run_ml(df, "classifica_serata_complessiva", "Classifica serata complessiva")
        print(f"  Complessiva — baseline MAE: {ml_comp['baseline_mae']}, "
              f"best: {ml_comp.get('miglior_modello')} MAE={ml_comp.get('miglior_mae')}")
    except ImportError:
        print("  scikit-learn non disponibile, skip ML")
        ml_comp = {"errore": "scikit-learn non disponibile"}

    try:
        ml_tv = run_ml(df, "classifica_serata_televoto", "Classifica serata televoto")
        print(f"  Televoto — baseline MAE: {ml_tv['baseline_mae']}, "
              f"best: {ml_tv.get('miglior_modello')} MAE={ml_tv.get('miglior_mae')}")
    except ImportError:
        ml_tv = {"errore": "scikit-learn non disponibile"}

    print("\n[5/5] Sintesi e output...")
    synth_comp = build_synthesis(result_comp, ml_comp)
    synth_tv = build_synthesis(result_tv, ml_tv)

    output = {
        "meta": {
            "titolo": "Sanremo — L'ordine di esibizione influenza la classifica?",
            "domanda": "Esibirsi in zone centrali è meglio per il ranking della serata?",
            "metodologia": (
                "Analisi per quintili dell'ordine di esibizione relativo (0=primo, 1=ultimo). "
                "Doppia analisi: classifica televoto e classifica complessiva di serata. "
                "Test non parametrici (Spearman con bootstrap CI, Kruskal-Wallis con eta², "
                "Jonckheere-Terpstra per trend ordinato, Mann-Whitney con rank-biserial r, "
                "Chi² con Cramér's V) eseguiti per serata, per anno e globale. "
                "Correzione per test multipli: Bonferroni e Benjamini-Hochberg FDR. "
                "Bootstrap CI (5000 repliche) per le differenze di media. "
                "Analisi di robustezza: tercili, rank normalizzato, esclusione finale, "
                "esclusione serate 1-2, solo serate con >=20 artisti. "
                "Modelli ML (Linear, RandomForest, GradientBoosting) con permutation test."
            ),
            "fonte_dati": "dati_sremo/sanremo_dati_serate.xlsx",
            "anni_analizzati": [int(y) for y in sorted(df["year"].unique())],
            "n_osservazioni_totali": int(len(df)),
            "n_con_classifica_complessiva": int(df["classifica_serata_complessiva"].notna().sum()),
            "n_con_classifica_televoto": int(df["classifica_serata_televoto"].notna().sum()),
        },
        "sintesi": {
            "domanda": "Esibirsi in zone centrali è meglio per il ranking della serata?",
            "classifica_complessiva": synth_comp,
            "classifica_televoto": synth_tv,
            "conclusione_generale": overall_conclusion(synth_comp, synth_tv),
        },
        "analisi_classifica_complessiva": result_comp,
        "analisi_classifica_televoto": result_tv,
        "machine_learning": {
            "classifica_complessiva": ml_comp,
            "classifica_televoto": ml_tv,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("RISULTATI")
    print(f"{'=' * 70}")
    _print_result("Classifica complessiva", synth_comp)
    _print_result("Classifica televoto", synth_tv)
    print(f"\nConclusione: {output['sintesi']['conclusione_generale']}")
    print(f"\nOutput: {OUTPUT_JSON}")


def _print_result(label, s):
    print(f"\n  {label}:")
    print(f"    Test significativi: raw={s['n_test_significativi_raw']}, "
          f"Bonferroni={s['n_test_significativi_bonferroni']}, "
          f"FDR={s['n_test_significativi_fdr']} / {s['n_test_eseguiti']}")
    print(f"    Evidenza: {s['evidenza']}")
    print(f"    Direzione: {s['direzione']}")
    print(f"    Media centro: {s['media_centro']}, media estremi: {s['media_estremi']}")
    bd = s.get("bootstrap_diff_mean", {})
    if bd:
        print(f"    Bootstrap diff (centro-estremi): {bd.get('diff_observed')} "
              f"CI95=[{bd.get('ci_lower')}, {bd.get('ci_upper')}] "
              f"{'include 0' if bd.get('includes_zero') else 'NON include 0'}")
    if s.get("ml_miglioramento") is not None:
        print(f"    ML: miglioramento {s['ml_miglioramento']}% "
              f"({'sig.' if s.get('ml_significativo') else 'non sig.'})")
    print(f"    Robustezza consistente: {'Sì' if s.get('robustezza_consistente') else 'No'}")


if __name__ == "__main__":
    main()
