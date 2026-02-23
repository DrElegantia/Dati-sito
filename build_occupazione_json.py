"""
Fetches Italian labor market data from ISTAT SDMX REST API
and produces docs/occupazione_dashboard.json.

Datasets used (semantic names, more stable than numeric IDs):
  - DCCV_OCCUPATIMENS1   Occupati – dati mensili
  - DCCV_INATTIVMENS1    Inattivi – dati mensili
  - DCCV_TAXINATTMENS1   Tasso di inattività – dati mensili
  - DCCV_TAXOCCUMENS1    Tasso di occupazione – dati mensili
  - DCCN_SEQCONTIRFT     Reddito disponibile delle famiglie
"""

from __future__ import annotations

import io
import json
import sys
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
def _fetch_csv(dataflow: str, start: str = START_PERIOD) -> pd.DataFrame:
    """Download a full dataflow in SDMX‑CSV, return a DataFrame.

    Uses semantic dataflow names (e.g. ``IT1,DCCV_OCCUPATIMENS1,1.0``).
    Falls back to shorter forms if the full triplet fails.
    """
    # Try full triplet first, then just the name
    candidates = [
        f"IT1,{dataflow},1.0",
        dataflow,
    ]
    for ref in candidates:
        url = f"{BASE}/{ref}/all?startPeriod={start}"
        print(f"  GET {url}")
        try:
            r = requests.get(url, headers=CSV_ACCEPT, timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"    → {len(df)} rows, columns: {list(df.columns)}")
            return df
        except requests.HTTPError as exc:
            print(f"    ✗ HTTP {exc.response.status_code} — trying next form")
            continue
    raise RuntimeError(f"All URL forms failed for dataflow {dataflow}")


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
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
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


def _diag(df: pd.DataFrame, label: str):
    """Print unique values for key dimensions — useful for first‑run debugging."""
    print(f"  [{label}] shape={df.shape}")
    for c in ["SESSO", "CLASSE_ETA", "POSIZIONE_PROF", "ITTER107",
              "TIPO_DATO", "DURATA", "CARATTERE_OCC",
              "SETTORE_ISTIT", "SECTOR", "TIPO_DATO_CNT", "AGGREGATO"]:
        if c in df.columns:
            vals = sorted(df[c].dropna().unique().tolist())
            short = vals[:15]
            suffix = f" … (+{len(vals)-15})" if len(vals) > 15 else ""
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
    tot = df_it[tot_mask][["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "periodo", "OBS_VALUE": "occupati"}
    ).sort_values("periodo").drop_duplicates("periodo")

    # ── 1b. Per posizione professionale (base 100 gen 2023) ─────────
    pos_list = []
    if pos_col and pos_empl and pos_self:
        for code, label in [(pos_empl, "Dipendenti"), (pos_self, "Indipendenti")]:
            s = df_it[
                (df_it["SESSO"] == sex_total)
                & (df_it["CLASSE_ETA"] == age_total)
                & (df_it[pos_col] == code)
            ].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
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
        comp_src = df_it[
            (df_it["SESSO"] == sex_total)
            & (df_it["CLASSE_ETA"] == age_total)
            & (df_it[pos_col].isin([pos_empl, pos_self, pos_total]))
        ].copy()
        for per, g in comp_src.groupby("TIME_PERIOD"):
            total_val = g.loc[g[pos_col] == pos_total, "OBS_VALUE"]
            if total_val.empty or total_val.iloc[0] == 0:
                continue
            tv = total_val.iloc[0]
            for code, label in [(pos_empl, "Dipendenti"), (pos_self, "Indipendenti")]:
                v = g.loc[g[pos_col] == code, "OBS_VALUE"]
                if not v.empty:
                    comp_list.append({
                        "periodo": per,
                        "posizione": label,
                        "quota_pct": round(float(v.iloc[0] / tv * 100), 2),
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
        s = df_it[mask].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
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
        s = df_it[mask].sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD").copy()
        for _, r in s.iterrows():
            gen_list.append({
                "periodo": r["TIME_PERIOD"],
                "genere": label,
                "valore": _safe(r["OBS_VALUE"]),
            })
    gen_df = pd.DataFrame(gen_list).sort_values(["periodo", "genere"]) if gen_list else pd.DataFrame()

    # ── 1f. Quota dipendenti a termine ──────────────────────────────
    # Try multiple possible column names for contract type
    dur_col = _col(df_it, "DURATA", "CARATTERE_OCC", "CARATTERE_OCCUPAZIONE", "TIPO_CONTRATTO")
    term_list = []
    if dur_col and pos_empl and pos_col:
        dur_vals = set(df_it[dur_col].astype(str).unique())
        temp_code = "TEMP" if "TEMP" in dur_vals else ("TD" if "TD" in dur_vals else None)
        total_code = "TOTAL" if "TOTAL" in dur_vals else ("9" if "9" in dur_vals else None)
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
    all_parts = ["ITF", "ITG", "ITC", "ITH", "ITI"]
    df_macro = df[df["ITTER107"].isin(all_parts)].copy()

    # Check available age codes for youth
    eta_vals = set(df_macro["CLASSE_ETA"].unique())
    youth_age = "Y15-29" if "Y15-29" in eta_vals else ("Y15-24" if "Y15-24" in eta_vals else None)

    # ── 4a. Occupazione giovanile per macroregione ──────────────────
    youth_list = []
    if youth_age:
        youth = df_macro[
            (df_macro["SESSO"] == "9") & (df_macro["CLASSE_ETA"] == youth_age)
        ].copy()
        for per, g in youth.groupby("TIME_PERIOD"):
            for codes, label in macro.items():
                parts = codes.split("+")
                vals = g.loc[g["ITTER107"].isin(parts), "OBS_VALUE"]
                if not vals.empty:
                    youth_list.append({
                        "periodo": per,
                        "macroregione": label,
                        "tasso_occupazione": _safe(vals.mean()),
                    })
    youth_df = pd.DataFrame(youth_list).sort_values(["periodo", "macroregione"]) if youth_list else pd.DataFrame()

    # ── 4b. Occupazione femminile per macroregione ──────────────────
    fem_list = []
    fem = df_macro[
        (df_macro["SESSO"] == "2") & (df_macro["CLASSE_ETA"] == "Y15-64")
    ].copy()
    for per, g in fem.groupby("TIME_PERIOD"):
        for codes, label in macro.items():
            parts = codes.split("+")
            vals = g.loc[g["ITTER107"].isin(parts), "OBS_VALUE"]
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

    # Filter for household sector (S14)
    sector_col = _col(df, "SETTORE_ISTIT", "SECTOR", "SETTORE", "SETTORE_ISTITUZIONALE")
    if sector_col:
        households = df[df[sector_col].astype(str).str.contains("S14", case=False, na=False)].copy()
        if households.empty:
            print("  ⚠ No S14 (household) sector found — using all data")
            households = df.copy()
    else:
        households = df.copy()

    # Find disposable income indicator
    tipo_col = _col(households, "TIPO_DATO", "TIPO_DATO_CNT", "AGGREGATO", "TIPO_CONTO")
    if tipo_col:
        reddito = households[
            households[tipo_col].astype(str).str.contains("B6G|REDD_DISP|RDL", case=False, na=False)
        ].copy()
        if reddito.empty:
            print(f"  ⚠ No B6G/REDD_DISP indicator found in {tipo_col} — using all rows")
            reddito = households.copy()
    else:
        reddito = households.copy()

    reddito = reddito.sort_values("TIME_PERIOD").drop_duplicates("TIME_PERIOD")

    # Base‑100 index (first 2023 period available)
    periods = reddito["TIME_PERIOD"].astype(str)
    base_mask = periods.str.startswith("2023")
    base_per = periods[base_mask].sort_values().iloc[0] if base_mask.any() else BASE_PERIOD

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
    dataflows = [
        ("DCCV_OCCUPATIMENS1", START_PERIOD),
        ("DCCV_INATTIVMENS1", START_PERIOD),
        ("DCCV_TAXINATTMENS1", START_PERIOD),
        ("DCCV_TAXOCCUMENS1", START_PERIOD),
        ("DCCN_SEQCONTIRFT", "2015-01"),
    ]
    frames = []
    for i, (name, start) in enumerate(dataflows, 1):
        print(f"{i}/{len(dataflows)}  Fetching {name} …")
        frames.append(_fetch_csv(name, start))
        if i < len(dataflows):
            print(f"  (pausing {PAUSE}s for rate limit)")
            time.sleep(PAUSE)

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
