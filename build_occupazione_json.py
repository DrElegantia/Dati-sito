"""
Fetches Italian labor market data from ISTAT SDMX REST API
and produces docs/occupazione_dashboard.json.

Datasets used (with fallback references):
  - Occupati:              150_875, DCCV_OCCUPATI1, DCCV_OCCUPATIMENS1
  - Inattivi:              150_879, 150_880, DCCV_INATTIVI1, DCCV_INATTIV1
  - Tasso inattività:      150_882, 150_883, DCCV_TAXINATT1, DCCV_TASSOINATT1
  - Tasso occupazione:     150_878, 150_881, DCCV_TAXOCCU1, DCCV_TASSOOCCU1
  - Reddito famiglie:      175_634 (DSD: DCCN_ISTITUZ_TNA)

Note: ISTAT often exposes labour-force dataflows as broader datasets;
monthly series are filtered locally using FREQ == "M" when available.
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── API settings ────────────────────────────────────────────────────
BASE = "https://esploradati.istat.it/SDMXWS/rest/data"
CSV_ACCEPT = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
START_PERIOD = "2018-01"
PAUSE = 13  # seconds between API calls (ISTAT limit: 5 req/min)

# ── base‑index reference ────────────────────────────────────────────
BASE_PERIOD = "2023-01"  # January 2023 = 100


# ── helpers ─────────────────────────────────────────────────────────
def _fetch_csv(dataflow_refs: list[str], start: str = START_PERIOD) -> pd.DataFrame:
    """Download a dataflow in SDMX‑CSV, trying multiple flow references."""
    errors: list[str] = []
    for dataflow in dataflow_refs:
        refs = [f"IT1,{dataflow},1.0", dataflow]
        for ref in refs:
            url = f"{BASE}/{ref}/all?startPeriod={start}"
            print(f"  GET {url}")
            try:
                r = requests.get(url, headers=CSV_ACCEPT, timeout=180)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False)
                print(f"    → {len(df)} rows, columns: {list(df.columns)}")
                return df
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                msg = f"HTTP {status} {url}"
                print(f"    ✗ {msg} — trying next form")
                errors.append(msg)
                continue
            except requests.RequestException as exc:
                msg = f"REQ_ERR {url} {exc}"
                print(f"    ✗ {msg} — trying next form")
                errors.append(msg)
                continue
    raise RuntimeError(
        "All URL forms failed for dataflows "
        + ", ".join(dataflow_refs)
        + "\n  "
        + "\n  ".join(errors[-8:])
    )

def _empty_frame() -> pd.DataFrame:
    """Return an empty frame with canonical columns expected by builders."""
    cols = [
        "FREQ", "TIME_PERIOD", "OBS_VALUE",
        "ITTER107", "SESSO", "CLASSE_ETA", "POSIZIONE_PROF",
        "DURATA", "CARATTERE_OCC",
        "SETTORE_ISTIT", "SECTOR", "TIPO_DATO", "TIPO_DATO_CNT", "AGGREGATO",
    ]
    return pd.DataFrame(columns=cols)


def _assert_no_merge_markers() -> None:
    """Fail fast if this file contains unresolved git merge markers."""
    src = Path(__file__).read_text(encoding="utf-8")
    if re.search(r"^(<<<<<<<|=======|>>>>>>>)", src, flags=re.MULTILINE):
        raise RuntimeError("Merge conflict markers detected in build_occupazione_json.py")


def _safe(x):
    """Convert numpy types to native Python; NaN → None."""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return round(float(x), 2)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _safe(v) for k, v in row.items()} for row in df.to_dict("records")]


def _base100(series: pd.Series, periods: pd.Series, base: str) -> pd.Series:
    """Index a numeric series to 100 at the row whose period == base."""
    mask = periods.astype(str) == base
    if not mask.any():
        return series
    base_val = series.loc[mask].iloc[0]
    if pd.isna(base_val) or base_val == 0:
        return series
    return series / base_val * 100.0


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase columns, coerce OBS_VALUE, stringify dimension cols."""
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    # Harmonise ISTAT old/new dimension names to canonical ones used below
    rename_map = {
        "REF_AREA": "ITTER107",
        "SEX": "SESSO",
        "AGE": "CLASSE_ETA",
        "POSIZ_PROF": "POSIZIONE_PROF",
        "PERM_TEMP_EMPLOYEES": "DURATA",
        "DATA_TYPE": "TIPO_DATO",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    if "FREQ" in df.columns:
        month_mask = df["FREQ"].astype(str).str.strip().eq("M")
        if month_mask.any():
            df = df[month_mask].copy()
    # SESSO may arrive as int or str — normalise to str
    if "SESSO" in df.columns:
        df["SESSO"] = df["SESSO"].astype(str).str.strip()
    if "ITTER107" in df.columns:
        df["ITTER107"] = df["ITTER107"].astype(str).str.strip()
    if "CLASSE_ETA" in df.columns:
        df["CLASSE_ETA"] = df["CLASSE_ETA"].astype(str).str.strip()
    if "POSIZIONE_PROF" in df.columns:
        df["POSIZIONE_PROF"] = df["POSIZIONE_PROF"].astype(str).str.strip()
    return df


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _dedup_total(df: pd.DataFrame, period_col: str = "TIME_PERIOD") -> pd.DataFrame:
    """Keep one row per period.  When duplicates exist (e.g. unfiltered DURATA
    dimension), keep the row with the *largest* OBS_VALUE — the aggregate
    'total' is always >= any component."""
    return (
        df.sort_values([period_col, "OBS_VALUE"], ascending=[True, False])
        .drop_duplicates(period_col)
    )


def _find_extra_dim(df: pd.DataFrame, known_cols: set[str]) -> str | None:
    """Find a dimension column that still varies after all known filters.

    This detects hidden dimensions like DURATA / CARATTERE_OCC that cause
    duplicate rows per TIME_PERIOD.
    """
    skip = known_cols | {
        "DATAFLOW", "FREQ", "TIME_PERIOD", "OBS_VALUE",
        "OBS_STATUS", "OBS_FLAG", "UNIT_MEASURE", "UNIT_MULT",
        "CONF_STATUS", "ACTION",
    }
    for col in df.columns:
        if col in skip:
            continue
        unique = set(df[col].dropna().astype(str).unique())
        if len(unique) > 1:
            return col
    return None


_TOTAL_TOKENS = {"TOTAL", "_T", "9", "T", "TOT"}


def _diag(df: pd.DataFrame, label: str):
    """Print unique values for key dimensions — useful for first‑run debugging."""
    print(f"  [{label}] shape={df.shape}  columns={list(df.columns)}")
    for c in ["SESSO", "CLASSE_ETA", "POSIZIONE_PROF", "ITTER107",
              "TIPO_DATO", "DURATA", "CARATTERE_OCC", "TIPO_CONTRATTO",
              "SETTORE_ISTIT", "SECTOR", "TIPO_DATO_CNT", "AGGREGATO",
              "SETTORE", "SETTORE_ISTITUZIONALE", "TIPO_CONTO", "FREQ"]:
        if c in df.columns:
            vals = sorted(df[c].dropna().unique().tolist())
            short = vals[:20]
            suffix = f" … (+{len(vals)-20})" if len(vals) > 20 else ""
            print(f"    {c}: {short}{suffix}")


# ── 1. OCCUPATI ─────────────────────────────────────────────────────
def build_occupati(df_raw: pd.DataFrame) -> dict:
    df = _norm(df_raw)
    _diag(df, "OCCUPATI")

    # Territory = Italy
    df_it = df[df["ITTER107"] == "IT"].copy()

    # Detect which code is used for "total" position
    pos_col = _col(df, "POSIZIONE_PROF")
    if pos_col is None:
        print("  ⚠ POSIZIONE_PROF column missing — skipping position breakdowns")
        pos_total = None
        pos_empl = None
        pos_self = None
    else:
        uniq = set(df_it[pos_col].unique())
        pos_total = "TOTAL" if "TOTAL" in uniq else ("9" if "9" in uniq else None)
        pos_empl = "EMPL" if "EMPL" in uniq else ("1" if "1" in uniq else None)
        pos_self = "SELF" if "SELF" in uniq else ("2" if "2" in uniq else None)
        print(f"  pos_total={pos_total}  pos_empl={pos_empl}  pos_self={pos_self}")

    sex_total = "9"
    age_total = "Y_GE15" if "Y_GE15" in set(df_it["CLASSE_ETA"].unique()) else "Y15-89"

    # ── 1a. Andamento totale occupati ───────────────────────────────
    tot_mask = (df_it["SESSO"] == sex_total) & (df_it["CLASSE_ETA"] == age_total)
    if pos_total and pos_col:
        tot_mask = tot_mask & (df_it[pos_col] == pos_total)
    tot = _dedup_total(df_it[tot_mask])[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "occupati"}
    ).sort_values("periodo")

    # ── Detect the hidden duration dimension (DURATA/CARATTERE_OCC) ─
    # Employees often carry an extra dimension (TOTAL/TEMP/PERM) that
    # must be filtered to avoid picking random sub-categories.
    dur_col = _col(df_it, "DURATA", "CARATTERE_OCC", "CARATTERE_OCCUPAZIONE", "TIPO_CONTRATTO")
    if not dur_col and pos_col and pos_empl:
        # Auto-detect: find the column that varies within EMPL rows
        _probe = df_it[
            (df_it["SESSO"] == sex_total)
            & (df_it["CLASSE_ETA"] == age_total)
            & (df_it[pos_col] == pos_empl)
        ]
        dur_col = _find_extra_dim(
            _probe,
            {"ITTER107", "SESSO", "CLASSE_ETA", pos_col},
        )
        if dur_col:
            print(f"  Auto-detected duration column: {dur_col}  "
                  f"values: {sorted(_probe[dur_col].dropna().unique().tolist())}")

    dur_total = None
    if dur_col and dur_col in df_it.columns:
        dur_vals = set(df_it[dur_col].astype(str).unique())
        dur_total = next((t for t in _TOTAL_TOKENS if t in dur_vals), None)
        print(f"  dur_col={dur_col}  dur_total={dur_total}  dur_vals={sorted(dur_vals)}")

    # ── 1b. Per posizione professionale (base 100 gen 2023) ─────────
    pos_list = []
    if pos_col and pos_empl and pos_self:
        for code, label in [(pos_empl, "Dipendenti"), (pos_self, "Indipendenti")]:
            mask = (
                (df_it["SESSO"] == sex_total)
                & (df_it["CLASSE_ETA"] == age_total)
                & (df_it[pos_col] == code)
            )
            # For employees, filter for aggregate duration
            if code == pos_empl and dur_col and dur_total:
                mask = mask & (df_it[dur_col].astype(str) == dur_total)
            s = _dedup_total(df_it[mask]).copy()
            s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
            for _, r in s.iterrows():
                pos_list.append({
                    "periodo": r["TIME_PERIOD"],
                    "posizione": label,
                    "valore": _safe(r["OBS_VALUE"]),
                    "indice": _safe(r["idx"]),
                })
    pos_df = pd.DataFrame(pos_list).sort_values(["periodo", "posizione"]) if pos_list else pd.DataFrame()

    # ── 1c. Composizione percentuale occupati ───────────────────────
    comp_list = []
    if pos_col and pos_total and pos_empl and pos_self:
        comp_mask = (
            (df_it["SESSO"] == sex_total)
            & (df_it["CLASSE_ETA"] == age_total)
            & (df_it[pos_col].isin([pos_empl, pos_self, pos_total]))
        )
        # Filter for aggregate duration to avoid sub-categories
        if dur_col and dur_total:
            comp_mask = comp_mask & (df_it[dur_col].astype(str) == dur_total)
        comp_src = df_it[comp_mask].copy()
        for per, g in comp_src.groupby("TIME_PERIOD"):
            total_val = g.loc[g[pos_col] == pos_total, "OBS_VALUE"]
            if total_val.empty:
                continue
            tv = total_val.max()  # pick aggregate if duplicates remain
            if tv == 0:
                continue
            for code, label in [(pos_empl, "Dipendenti"), (pos_self, "Indipendenti")]:
                v = g.loc[g[pos_col] == code, "OBS_VALUE"]
                if not v.empty:
                    comp_list.append({
                        "periodo": per,
                        "posizione": label,
                        "quota_pct": round(float(v.max() / tv * 100), 2),
                    })
    comp_df = pd.DataFrame(comp_list).sort_values(["periodo", "posizione"]) if comp_list else pd.DataFrame()

    # ── 1d. Per fascia d'età (base 100 gen 2023) ────────────────────
    age_codes = {"Y15-24": "15-24", "Y25-34": "25-34", "Y35-49": "35-49", "Y50-64": "50-64"}
    eta_available = set(df_it["CLASSE_ETA"].unique())
    age_list = []
    for code, label in age_codes.items():
        if code not in eta_available:
            continue
        mask = (df_it["SESSO"] == sex_total) & (df_it["CLASSE_ETA"] == code)
        if pos_total and pos_col:
            mask = mask & (df_it[pos_col] == pos_total)
        s = _dedup_total(df_it[mask]).copy()
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            age_list.append({
                "periodo": r["TIME_PERIOD"],
                "fascia_eta": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    age_df = pd.DataFrame(age_list).sort_values(["periodo", "fascia_eta"]) if age_list else pd.DataFrame()

    # ── 1e. Per genere ──────────────────────────────────────────────
    gen_list = []
    for sex, label in [("1", "Maschi"), ("2", "Femmine")]:
        mask = (df_it["SESSO"] == sex) & (df_it["CLASSE_ETA"] == age_total)
        if pos_total and pos_col:
            mask = mask & (df_it[pos_col] == pos_total)
        s = _dedup_total(df_it[mask]).copy()
        for _, r in s.iterrows():
            gen_list.append({
                "periodo": r["TIME_PERIOD"],
                "genere": label,
                "valore": _safe(r["OBS_VALUE"]),
            })
    gen_df = pd.DataFrame(gen_list).sort_values(["periodo", "genere"]) if gen_list else pd.DataFrame()

    # ── 1f. Quota dipendenti a termine ──────────────────────────────
    # dur_col was already detected above (either by name or auto-detected)
    term_list = []
    if not dur_col:
        print("  ⚠ No DURATA/CARATTERE_OCC column — quota_termine will be empty")
    if dur_col and pos_empl and pos_col:
        dur_vals = set(df_it[dur_col].astype(str).unique())
        temp_code = next(
            (t for t in ["TEMP", "TD", "2", "FT"] if t in dur_vals), None
        )
        total_code = dur_total  # already computed above
        if temp_code and total_code:
            term_src = df_it[
                (df_it["SESSO"] == sex_total)
                & (df_it["CLASSE_ETA"] == age_total)
                & (df_it[pos_col] == pos_empl)
                & (df_it[dur_col].isin([temp_code, total_code]))
            ].copy()
            for per, g in term_src.groupby("TIME_PERIOD"):
                tot_v = g.loc[g[dur_col] == total_code, "OBS_VALUE"]
                tmp_v = g.loc[g[dur_col] == temp_code, "OBS_VALUE"]
                if not tot_v.empty and not tmp_v.empty and tot_v.iloc[0] > 0:
                    term_list.append({
                        "periodo": per,
                        "quota_termine_pct": round(float(tmp_v.iloc[0] / tot_v.iloc[0] * 100), 2),
                    })
    term_df = pd.DataFrame(term_list).sort_values("periodo") if term_list else pd.DataFrame()

    return {
        "occupati_totali": _records(tot),
        "occupati_posizione": _records(pos_df),
        "composizione_pct": _records(comp_df),
        "occupati_eta": _records(age_df),
        "occupati_genere": _records(gen_df),
        "quota_termine": _records(term_df),
    }


# ── 2. INATTIVI ─────────────────────────────────────────────────────
def build_inattivi(df_raw: pd.DataFrame) -> dict:
    df = _norm(df_raw)
    _diag(df, "INATTIVI")
    df_it = df[df["ITTER107"] == "IT"].copy()

    age_15_64 = "Y15-64"

    # ── 2a. Inattivi totali 15‑64  (base 100 gen 2023) ─────────────
    tot = df_it[
        (df_it["SESSO"] == "9") & (df_it["CLASSE_ETA"] == age_15_64)
    ].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
    tot["idx"] = _base100(tot["OBS_VALUE"], tot["TIME_PERIOD"], BASE_PERIOD)
    tot_out = tot[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "inattivi"}
    ).copy()
    tot_out["indice"] = tot["idx"].values

    # ── 2b. Per fascia d'età (base 100 gen 2023) ────────────────────
    age_codes = {"Y15-24": "15-24", "Y25-34": "25-34", "Y35-49": "35-49", "Y50-64": "50-64"}
    age_list = []
    for code, label in age_codes.items():
        s = df_it[
            (df_it["SESSO"] == "9") & (df_it["CLASSE_ETA"] == code)
        ].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            age_list.append({
                "periodo": r["TIME_PERIOD"],
                "fascia_eta": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    age_df = pd.DataFrame(age_list).sort_values(["periodo", "fascia_eta"]) if age_list else pd.DataFrame()

    # ── 2c. Per genere 15‑64 (base 100 gen 2023) ───────────────────
    gen_list = []
    for sex, label in [("1", "Maschi"), ("2", "Femmine")]:
        s = df_it[
            (df_it["SESSO"] == sex) & (df_it["CLASSE_ETA"] == age_15_64)
        ].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            gen_list.append({
                "periodo": r["TIME_PERIOD"],
                "genere": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    gen_df = pd.DataFrame(gen_list).sort_values(["periodo", "genere"]) if gen_list else pd.DataFrame()

    return {
        "inattivi_totali": _records(tot_out),
        "inattivi_eta": _records(age_df),
        "inattivi_genere": _records(gen_df),
    }


# ── 3. TASSO DI INATTIVITÀ ─────────────────────────────────────────
def build_tasso_inattivita(df_raw: pd.DataFrame) -> dict:
    df = _norm(df_raw)
    _diag(df, "TASSO_INATTIVITA")
    df_it = df[df["ITTER107"] == "IT"].copy()

    rates = df_it[
        (df_it["SESSO"] == "9") & (df_it["CLASSE_ETA"] == "Y15-64")
    ].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
    out = rates[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "tasso_inattivita"}
    )
    return {"tassi_inattivita": _records(out)}


# ── 4. TASSO DI OCCUPAZIONE ────────────────────────────────────────
def build_tasso_occupazione(df_raw: pd.DataFrame) -> dict:
    df = _norm(df_raw)
    _diag(df, "TASSO_OCCUPAZIONE")

    macro = {"ITF+ITG": "Mezzogiorno", "ITC+ITH": "Nord", "ITI": "Centro"}
    nuts1_prefixes = ["ITF", "ITG", "ITC", "ITH", "ITI"]
    all_itter = sorted(df["ITTER107"].unique().tolist())
    print(f"  Available ITTER107: {all_itter[:30]}")

    # Try exact NUTS1 match first; fall back to NUTS2 prefix matching
    df_macro = df[df["ITTER107"].isin(nuts1_prefixes)].copy()
    if df_macro.empty:
        # NUTS2 codes: ITC1, ITC2, ITF1 … — match by prefix
        prefix_mask = df["ITTER107"].apply(
            lambda x: any(str(x).startswith(p) for p in nuts1_prefixes)
            and str(x) != "IT"
        )
        df_macro = df[prefix_mask].copy()
        if not df_macro.empty:
            df_macro["_NUTS1"] = df_macro["ITTER107"].str[:3]
            print(f"  Using NUTS2→NUTS1 prefix matching ({len(df_macro)} rows)")
        else:
            print("  ⚠ No macro-region data — macro breakdowns will be empty")
    else:
        df_macro["_NUTS1"] = df_macro["ITTER107"]

    # Check available age codes for youth
    eta_vals = set(df_macro["CLASSE_ETA"].unique()) if not df_macro.empty else set()
    youth_age = "Y15-29" if "Y15-29" in eta_vals else ("Y15-24" if "Y15-24" in eta_vals else None)

    def _macro_aggregate(g: pd.DataFrame, prefixes: list[str]) -> pd.Series:
        """Select rows matching NUTS1 prefixes and return their values."""
        return g.loc[g["_NUTS1"].isin(prefixes), "OBS_VALUE"]

    # ── 4a. Occupazione giovanile per macroregione ──────────────────
    youth_list = []
    if youth_age and not df_macro.empty:
        youth = df_macro[
            (df_macro["SESSO"] == "9") & (df_macro["CLASSE_ETA"] == youth_age)
        ].copy()
        for per, g in youth.groupby("TIME_PERIOD"):
            for codes, label in macro.items():
                parts = codes.split("+")
                vals = _macro_aggregate(g, parts)
                if not vals.empty:
                    youth_list.append({
                        "periodo": per,
                        "macroregione": label,
                        "tasso_occupazione": _safe(vals.mean()),
                    })
    youth_df = pd.DataFrame(youth_list).sort_values(["periodo", "macroregione"]) if youth_list else pd.DataFrame()

    # ── 4b. Occupazione femminile per macroregione ──────────────────
    fem_list = []
    if not df_macro.empty:
        fem = df_macro[
            (df_macro["SESSO"] == "2") & (df_macro["CLASSE_ETA"] == "Y15-64")
        ].copy()
        for per, g in fem.groupby("TIME_PERIOD"):
            for codes, label in macro.items():
                parts = codes.split("+")
                vals = _macro_aggregate(g, parts)
                if not vals.empty:
                    fem_list.append({
                        "periodo": per,
                        "macroregione": label,
                        "tasso_occupazione": _safe(vals.mean()),
                    })
    fem_df = pd.DataFrame(fem_list).sort_values(["periodo", "macroregione"]) if fem_list else pd.DataFrame()

    return {
        "occupazione_giovanile_macro": _records(youth_df),
        "occupazione_femminile_macro": _records(fem_df),
    }


# ── 5. REDDITO DISPONIBILE FAMIGLIE ────────────────────────────────
def build_reddito(df_raw: pd.DataFrame) -> dict:
    df = _norm(df_raw)
    _diag(df, "REDDITO")

    # Filter for territory = Italy if available
    if "ITTER107" in df.columns:
        df_it = df[df["ITTER107"] == "IT"].copy()
        if df_it.empty:
            print("  ⚠ No ITTER107=IT — using all data")
            df_it = df.copy()
    else:
        df_it = df.copy()

    # Filter for household sector (S14) — prefer exact S14, avoid S14_S15
    sector_col = _col(df_it, "SETTORE_ISTIT", "SECTOR", "SETTORE", "SETTORE_ISTITUZIONALE")
    if sector_col:
        # Try exact S14 first
        exact = df_it[df_it[sector_col].astype(str).str.strip() == "S14"]
        if not exact.empty:
            households = exact.copy()
        else:
            households = df_it[
                df_it[sector_col].astype(str).str.contains("S14", case=False, na=False)
            ].copy()
        if households.empty:
            print("  ⚠ No S14 (household) sector found — using all data")
            households = df_it.copy()
        else:
            print(f"  Sector filter: {sector_col}={sorted(households[sector_col].unique().tolist())}")
    else:
        households = df_it.copy()

    # Find disposable income indicator (B6G = gross disposable income in ESA 2010)
    tipo_col = _col(households, "TIPO_DATO_CNT", "TIPO_DATO", "AGGREGATO", "TIPO_CONTO")
    if tipo_col:
        tipo_vals = sorted(households[tipo_col].unique().tolist())
        print(f"  {tipo_col} values: {tipo_vals[:30]}")
        # Prefer exact B6G match
        reddito = households[households[tipo_col].astype(str).str.strip() == "B6G"].copy()
        if reddito.empty:
            # Fallback to partial match
            reddito = households[
                households[tipo_col].astype(str).str.contains(
                    "B6G|REDD_DISP|RDL", case=False, na=False
                )
            ].copy()
        if reddito.empty:
            print(f"  ⚠ No B6G/REDD_DISP indicator in {tipo_col} — using all rows")
            reddito = households.copy()
        else:
            print(f"  B6G filter → {len(reddito)} rows")
    else:
        reddito = households.copy()

    # Filter remaining dimensions to avoid mixing levels, changes, and prices
    # Look for a "measure type" / "correction" / "prices" dimension and keep
    # only level data at current prices.
    _known = {"ITTER107", "SESSO", "CLASSE_ETA", "TIME_PERIOD", "OBS_VALUE",
              "FREQ", "DATAFLOW", "OBS_STATUS", "OBS_FLAG", "UNIT_MEASURE",
              "UNIT_MULT", "CONF_STATUS", "ACTION"}
    if sector_col:
        _known.add(sector_col)
    if tipo_col:
        _known.add(tipo_col)

    for col in list(reddito.columns):
        if col in _known:
            continue
        unique = sorted(reddito[col].dropna().astype(str).unique().tolist())
        if len(unique) > 1:
            print(f"  Extra dim {col}: {unique}")
            # Try to keep "level / current prices" variant
            # Common ISTAT codes: V = level, PC = current prices, N = no correction
            preferred = None
            for candidate in ["V", "PC", "N", "VALASS", "CP_MEUR", "CP_EUR",
                              "L", "TOTAL", "_T", "9"]:
                if candidate in unique:
                    preferred = candidate
                    break
            if preferred:
                reddito = reddito[reddito[col].astype(str) == preferred].copy()
                print(f"    → filtered {col}={preferred}")

    reddito = _dedup_total(reddito)

    # Base‑100 index (first 2023 period available)
    periods = reddito["TIME_PERIOD"].astype(str)
    base_mask = periods.str.startswith("2023")
    base_per = periods[base_mask].sort_values().iloc[0] if base_mask.any() else periods.iloc[-1]

    reddito = reddito.copy()
    reddito["idx"] = _base100(reddito["OBS_VALUE"], reddito["TIME_PERIOD"], base_per)
    out = reddito[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "reddito"}
    ).copy()
    out["indice"] = reddito["idx"].values

    return {
        "reddito_disponibile": _records(out),
        "base_period_reddito": str(base_per),
    }


# ── MAIN ────────────────────────────────────────────────────────────
def main():
    _assert_no_merge_markers()
    out_path = Path("docs/occupazione_dashboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "meta": {
            "source": "ISTAT – Rilevazione sulle forze di lavoro",
            "base_index": BASE_PERIOD,
            "datasets": [
                "150_875|DCCV_OCCUPATI1",
                "150_879|150_880|DCCV_INATTIVI1",
                "150_882|150_883|DCCV_TAXINATT1",
                "150_878|150_881|DCCV_TAXOCCU1",
                "175_634|DCCN_ISTITUZ_TNA",
            ],
        }
    }

    # ── Fetch datasets (respecting 5 req/min rate limit) ────────────
    # Each entry: (label, [candidate IDs to try], startPeriod)
    # Numeric IDs are for esploradati.istat.it; DSD names are fallbacks.
    dataflows = [
        (["150_875", "DCCV_OCCUPATI1", "DCCV_OCCUPATIMENS1"], START_PERIOD),
        (["150_879", "150_880", "DCCV_INATTIVI1", "DCCV_INATTIV1", "DCCV_INATTIVMENS1"], START_PERIOD),
        (["150_882", "150_883", "DCCV_TAXINATT1", "DCCV_TASSOINATT1", "DCCV_TAXINATTMENS1"], START_PERIOD),
        (["150_878", "150_881", "DCCV_TAXOCCU1", "DCCV_TASSOOCCU1", "DCCV_TAXOCCUMENS1"], START_PERIOD),
        (["175_634", "93_1095", "737_1093"], "2015-01"),
    ]
    frames = []
    failed_dataflows = []
    for i, (refs, start) in enumerate(dataflows, 1):
        print(f"{i}/{len(dataflows)}  Fetching {refs[0]} (fallbacks: {refs[1:]}) …")
        try:
            frames.append(_fetch_csv(refs, start))
        except RuntimeError as exc:
            print(f"  ⚠ fetch failed; continuing with empty dataset: {exc}")
            failed_dataflows.append({"requested": refs, "error": str(exc)})
            frames.append(_empty_frame())
        if i < len(dataflows):
            print(f"  (pausing {PAUSE}s for rate limit)")
            time.sleep(PAUSE)

    if failed_dataflows:
        payload["meta"]["fetch_warnings"] = failed_dataflows

    df_occ, df_ina, df_tin, df_toc, df_red = frames

    # ── Process each section ────────────────────────────────────────
    print("\nProcessing occupati …")
    payload.update(build_occupati(df_occ))

    print("\nProcessing inattivi …")
    payload.update(build_inattivi(df_ina))

    print("\nProcessing tasso inattività …")
    payload.update(build_tasso_inattivita(df_tin))

    print("\nProcessing tasso occupazione …")
    payload.update(build_tasso_occupazione(df_toc))

    print("\nProcessing reddito …")
    payload.update(build_reddito(df_red))

    # ── Summary ─────────────────────────────────────────────────────
    print("\n── Result summary ──")
    for key in ["occupati_totali", "occupati_posizione", "composizione_pct",
                "occupati_eta", "occupati_genere", "quota_termine",
                "inattivi_totali", "inattivi_eta", "inattivi_genere",
                "tassi_inattivita", "occupazione_giovanile_macro",
                "occupazione_femminile_macro", "reddito_disponibile"]:
        data = payload.get(key)
        n = len(data) if isinstance(data, list) else 0
        status = "✓" if n > 0 else "✗ EMPTY"
        print(f"  {key}: {n} records {status}")

    # ── Write JSON ──────────────────────────────────────────────────
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"\nWritten → {out_path.resolve()}")


if __name__ == "__main__":
    main()
