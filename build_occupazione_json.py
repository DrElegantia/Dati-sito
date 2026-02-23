"""
Fetches Italian labor market data from ISTAT SDMX REST API
and produces docs/occupazione_dashboard.json.

Datasets used:
  - 150_876  (DCCV_OCCUPATIMENS1)   Occupati – dati mensili
  - 152_879  (DCCV_INATTIVMENS1)    Inattivi – dati mensili
  - 152_878  (DCCV_TAXINATTMENS1)   Tasso di inattività – dati mensili
  - 150_915  (DCCV_TAXOCCUMENS1)    Tasso di occupazione – dati mensili
  - 39_1005  (DCCN_SEQCONTIRFT)     Reddito disponibile delle famiglie
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── API settings ────────────────────────────────────────────────────
BASE = "https://esploradati.istat.it/SDMXWS/rest/data"
CSV_ACCEPT = {"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"}
START_PERIOD = "2018-01"
PAUSE = 13  # seconds between API calls (5 req/min limit)

# ── base‑index reference ────────────────────────────────────────────
BASE_PERIOD = "2023-01"  # January 2023 = 100


# ── helpers ─────────────────────────────────────────────────────────
def _fetch_csv(dataflow: str, start: str = START_PERIOD) -> pd.DataFrame:
    """Download a full dataflow in CSV, return a DataFrame."""
    url = f"{BASE}/{dataflow}/all?startPeriod={start}"
    print(f"  GET {url}")
    r = requests.get(url, headers=CSV_ACCEPT, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"    → {len(df)} rows")
    return df


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


# ── 1. OCCUPATI  (DCCV_OCCUPATIMENS1 / 150_876) ────────────────────
def build_occupati(df_raw: pd.DataFrame) -> dict:
    """
    From the monthly employed‑persons dataset produce:
      - occupati_totali          trend of total employed
      - occupati_posizione       by professional position  (base 100)
      - composizione_pct         percentage composition
      - occupati_eta             by age group               (base 100)
      - occupati_genere          by gender
      - quota_termine            share of temporary employees
    """
    # Standardise column names (SDMX CSV uses upper‑case)
    df = df_raw.copy()
    cols = {c: c.upper() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Keep only OBS_VALUE as float
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])

    # Common filters
    df_it = df[df["ITTER107"] == "IT"].copy()

    # ── 1a. Andamento totale occupati ───────────────────────────────
    tot = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == "Y_GE15")
        & (df_it["POSIZIONE_PROF"] == "TOTAL")
    ].copy()
    tot = tot[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "occupati"}
    )
    tot = tot.sort_values("periodo").drop_duplicates("periodo")

    # ── 1b. Per posizione professionale (base 100 gen 2023) ─────────
    pos_filter = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == "Y_GE15")
        & (df_it["POSIZIONE_PROF"].isin(["EMPL", "SELF"]))
    ].copy()
    pos_list = []
    for pos_code, pos_label in [("EMPL", "Dipendenti"), ("SELF", "Indipendenti")]:
        s = pos_filter[pos_filter["POSIZIONE_PROF"] == pos_code].copy()
        s = s.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            pos_list.append({
                "periodo": r["TIME_PERIOD"],
                "posizione": pos_label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    pos_df = pd.DataFrame(pos_list).sort_values(["periodo", "posizione"])

    # ── 1c. Composizione percentuale occupati ───────────────────────
    comp_src = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == "Y_GE15")
        & (df_it["POSIZIONE_PROF"].isin(["EMPL", "SELF", "TOTAL"]))
    ].copy()
    comp_list = []
    for per, g in comp_src.groupby("TIME_PERIOD"):
        total_val = g.loc[g["POSIZIONE_PROF"] == "TOTAL", "OBS_VALUE"]
        if total_val.empty or total_val.iloc[0] == 0:
            continue
        tv = total_val.iloc[0]
        for pos_code, pos_label in [("EMPL", "Dipendenti"), ("SELF", "Indipendenti")]:
            v = g.loc[g["POSIZIONE_PROF"] == pos_code, "OBS_VALUE"]
            if not v.empty:
                comp_list.append({
                    "periodo": per,
                    "posizione": pos_label,
                    "quota_pct": round(float(v.iloc[0] / tv * 100), 2),
                })
    comp_df = pd.DataFrame(comp_list).sort_values(["periodo", "posizione"])

    # ── 1d. Per fascia d'età (base 100 gen 2023) ────────────────────
    age_codes = {
        "Y15-24": "15-24",
        "Y25-34": "25-34",
        "Y35-49": "35-49",
        "Y50-64": "50-64",
    }
    age_filter = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"].isin(age_codes.keys()))
        & (df_it["POSIZIONE_PROF"] == "TOTAL")
    ].copy()
    age_list = []
    for code, label in age_codes.items():
        s = age_filter[age_filter["CLASSE_ETA"] == code].copy()
        s = s.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            age_list.append({
                "periodo": r["TIME_PERIOD"],
                "fascia_eta": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    age_df = pd.DataFrame(age_list).sort_values(["periodo", "fascia_eta"])

    # ── 1e. Per genere ──────────────────────────────────────────────
    gen_filter = df_it[
        (df_it["SESSO"].isin([1, 2]))
        & (df_it["CLASSE_ETA"] == "Y_GE15")
        & (df_it["POSIZIONE_PROF"] == "TOTAL")
    ].copy()
    gen_list = []
    for sex_code, sex_label in [(1, "Maschi"), (2, "Femmine")]:
        s = gen_filter[gen_filter["SESSO"] == sex_code].copy()
        s = s.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
        for _, r in s.iterrows():
            gen_list.append({
                "periodo": r["TIME_PERIOD"],
                "genere": sex_label,
                "valore": _safe(r["OBS_VALUE"]),
            })
    gen_df = pd.DataFrame(gen_list).sort_values(["periodo", "genere"])

    # ── 1f. Quota dipendenti a termine ──────────────────────────────
    # DURATA = TEMP (a tempo determinato), PERM (a tempo indeterminato)
    term_filter = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == "Y_GE15")
        & (df_it["POSIZIONE_PROF"] == "EMPL")
    ].copy()
    # Check if DURATA column exists (contract duration)
    term_list = []
    if "DURATA" in term_filter.columns:
        for per, g in term_filter.groupby("TIME_PERIOD"):
            tot_v = g.loc[g["DURATA"] == "TOTAL", "OBS_VALUE"]
            tmp_v = g.loc[g["DURATA"] == "TEMP", "OBS_VALUE"]
            if not tot_v.empty and not tmp_v.empty and tot_v.iloc[0] > 0:
                term_list.append({
                    "periodo": per,
                    "quota_termine_pct": round(
                        float(tmp_v.iloc[0] / tot_v.iloc[0] * 100), 2
                    ),
                })
    elif "CARATTERE_OCCUPAZIONE" in term_filter.columns:
        for per, g in term_filter.groupby("TIME_PERIOD"):
            tot_v = g.loc[
                g["CARATTERE_OCCUPAZIONE"] == "TOTAL", "OBS_VALUE"
            ]
            tmp_v = g.loc[
                g["CARATTERE_OCCUPAZIONE"] == "TEMP", "OBS_VALUE"
            ]
            if not tot_v.empty and not tmp_v.empty and tot_v.iloc[0] > 0:
                term_list.append({
                    "periodo": per,
                    "quota_termine_pct": round(
                        float(tmp_v.iloc[0] / tot_v.iloc[0] * 100), 2
                    ),
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


# ── 2. INATTIVI  (DCCV_INATTIVMENS1 / 152_879) ────────────────────
def build_inattivi(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    cols = {c: c.upper() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df_it = df[df["ITTER107"] == "IT"].copy()

    age_15_64 = "Y15-64"

    # ── 2a. Inattivi totali 15‑64  (base 100 gen 2023) ─────────────
    tot = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == age_15_64)
    ].copy()
    tot = tot.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
    tot["idx"] = _base100(tot["OBS_VALUE"], tot["TIME_PERIOD"], BASE_PERIOD)
    tot_out = tot[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "inattivi"}
    )
    tot_out["indice"] = tot["idx"].values
    tot_out = tot_out.copy()

    # ── 2b. Per fascia d'età (base 100 gen 2023) ────────────────────
    age_codes = {
        "Y15-24": "15-24",
        "Y25-34": "25-34",
        "Y35-49": "35-49",
        "Y50-64": "50-64",
    }
    age_filter = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"].isin(age_codes.keys()))
    ].copy()
    age_list = []
    for code, label in age_codes.items():
        s = age_filter[age_filter["CLASSE_ETA"] == code].copy()
        s = s.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            age_list.append({
                "periodo": r["TIME_PERIOD"],
                "fascia_eta": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    age_df = pd.DataFrame(age_list).sort_values(["periodo", "fascia_eta"])

    # ── 2c. Per genere 15‑64 (base 100 gen 2023) ───────────────────
    gen_filter = df_it[
        (df_it["SESSO"].isin([1, 2]))
        & (df_it["CLASSE_ETA"] == age_15_64)
    ].copy()
    gen_list = []
    for sex, label in [(1, "Maschi"), (2, "Femmine")]:
        s = gen_filter[gen_filter["SESSO"] == sex].copy()
        s = s.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
        s["idx"] = _base100(s["OBS_VALUE"], s["TIME_PERIOD"], BASE_PERIOD)
        for _, r in s.iterrows():
            gen_list.append({
                "periodo": r["TIME_PERIOD"],
                "genere": label,
                "valore": _safe(r["OBS_VALUE"]),
                "indice": _safe(r["idx"]),
            })
    gen_df = pd.DataFrame(gen_list).sort_values(["periodo", "genere"])

    return {
        "inattivi_totali": _records(tot_out),
        "inattivi_eta": _records(age_df),
        "inattivi_genere": _records(gen_df),
    }


# ── 3. TASSO DI INATTIVITÀ  (DCCV_TAXINATTMENS1 / 152_878) ────────
def build_tasso_inattivita(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    cols = {c: c.upper() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    df_it = df[df["ITTER107"] == "IT"].copy()

    age_15_64 = "Y15-64"
    rates = df_it[
        (df_it["SESSO"] == 9)
        & (df_it["CLASSE_ETA"] == age_15_64)
    ].copy()
    rates = rates.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")
    out = rates[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "tasso_inattivita"}
    )
    return {"tassi_inattivita": _records(out)}


# ── 4. TASSO DI OCCUPAZIONE  (DCCV_TAXOCCUMENS1 / 150_915) ─────────
def build_tasso_occupazione(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    cols = {c: c.upper() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])

    macro = {
        "ITF+ITG": "Mezzogiorno",
        "ITC+ITH": "Nord",
        "ITI": "Centro",
    }
    single_codes = ["ITF", "ITG", "ITC", "ITH", "ITI"]
    df_macro = df[df["ITTER107"].isin(single_codes)].copy()

    youth_age = "Y15-29"
    # ── 4a. Occupazione giovanile per macroregione ──────────────────
    youth = df_macro[
        (df_macro["SESSO"] == 9) & (df_macro["CLASSE_ETA"] == youth_age)
    ].copy()
    youth_list = []
    for per, g in youth.groupby("TIME_PERIOD"):
        for (codes, label) in macro.items():
            parts = codes.split("+")
            vals = g.loc[g["ITTER107"].isin(parts), "OBS_VALUE"]
            if not vals.empty:
                youth_list.append({
                    "periodo": per,
                    "macroregione": label,
                    "tasso_occupazione": _safe(vals.mean()),
                })
    youth_df = pd.DataFrame(youth_list).sort_values(["periodo", "macroregione"])

    # ── 4b. Occupazione femminile per macroregione ──────────────────
    fem = df_macro[
        (df_macro["SESSO"] == 2) & (df_macro["CLASSE_ETA"] == "Y15-64")
    ].copy()
    fem_list = []
    for per, g in fem.groupby("TIME_PERIOD"):
        for (codes, label) in macro.items():
            parts = codes.split("+")
            vals = g.loc[g["ITTER107"].isin(parts), "OBS_VALUE"]
            if not vals.empty:
                fem_list.append({
                    "periodo": per,
                    "macroregione": label,
                    "tasso_occupazione": _safe(vals.mean()),
                })
    fem_df = pd.DataFrame(fem_list).sort_values(["periodo", "macroregione"])

    return {
        "occupazione_giovanile_macro": _records(youth_df),
        "occupazione_femminile_macro": _records(fem_df),
    }


# ── 5. REDDITO DISPONIBILE FAMIGLIE  (DCCN_SEQCONTIRFT / 39_1005) ──
def build_reddito(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    cols = {c: c.upper() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])

    # Filter for household sector (S14) or total if available
    sector_col = None
    for c in ["SETTORE_ISTIT", "SECTOR", "SETTORE"]:
        if c in df.columns:
            sector_col = c
            break

    if sector_col:
        households = df[df[sector_col].str.contains("S14", case=False, na=False)].copy()
        if households.empty:
            households = df.copy()
    else:
        households = df.copy()

    # Try to find the right indicator (reddito disponibile lordo)
    tipo_col = None
    for c in ["TIPO_DATO", "TIPO_DATO_CNT", "AGGREGATO"]:
        if c in df.columns:
            tipo_col = c
            break

    if tipo_col:
        # Look for disposable income indicator
        reddito = households[
            households[tipo_col].str.contains("B6G|REDD_DISP|RDL", case=False, na=False)
        ].copy()
        if reddito.empty:
            reddito = households.copy()
    else:
        reddito = households.copy()

    reddito = reddito.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")

    # Build base‑100 index (closest to Jan 2023)
    periods = reddito["TIME_PERIOD"].astype(str)
    base_mask = periods.str.startswith("2023")
    if base_mask.any():
        base_per = periods[base_mask].sort_values().iloc[0]
    else:
        base_per = BASE_PERIOD

    reddito["idx"] = _base100(reddito["OBS_VALUE"], reddito["TIME_PERIOD"], base_per)
    out = reddito[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "reddito"}
    ).copy()
    out["indice"] = reddito["idx"].values

    return {
        "reddito_disponibile": _records(out),
        "base_period": str(base_per),
    }


# ── MAIN ────────────────────────────────────────────────────────────
def main():
    out_path = Path("docs/occupazione_dashboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "meta": {
            "source": "ISTAT – Rilevazione sulle forze di lavoro",
            "base_index": BASE_PERIOD,
            "datasets": [
                "DCCV_OCCUPATIMENS1",
                "DCCV_INATTIVMENS1",
                "DCCV_TAXINATTMENS1",
                "DCCV_TAXOCCUMENS1",
                "DCCN_SEQCONTIRFT",
            ],
        }
    }

    # ── Fetch datasets (respecting 5 req/min rate limit) ────────────
    print("1/5  Fetching DCCV_OCCUPATIMENS1 …")
    df_occ = _fetch_csv("150_876")
    time.sleep(PAUSE)

    print("2/5  Fetching DCCV_INATTIVMENS1 …")
    df_ina = _fetch_csv("152_879")
    time.sleep(PAUSE)

    print("3/5  Fetching DCCV_TAXINATTMENS1 …")
    df_tin = _fetch_csv("152_878")
    time.sleep(PAUSE)

    print("4/5  Fetching DCCV_TAXOCCUMENS1 …")
    df_toc = _fetch_csv("150_915")
    time.sleep(PAUSE)

    print("5/5  Fetching DCCN_SEQCONTIRFT …")
    df_red = _fetch_csv("39_1005", start="2015-01")

    # ── Process each section ────────────────────────────────────────
    print("Processing occupati …")
    payload.update(build_occupati(df_occ))

    print("Processing inattivi …")
    payload.update(build_inattivi(df_ina))

    print("Processing tasso inattività …")
    payload.update(build_tasso_inattivita(df_tin))

    print("Processing tasso occupazione …")
    payload.update(build_tasso_occupazione(df_toc))

    print("Processing reddito …")
    payload.update(build_reddito(df_red))

    # ── Write JSON ──────────────────────────────────────────────────
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Written → {out_path.resolve()}")


if __name__ == "__main__":
    main()
