from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import eurostat

PAESI = ["IT", "DE", "FR", "ES", "NL", "PL", "EU27_2020"]
PAESI_LABEL = {
    "IT": "Italia",
    "DE": "Germania",
    "FR": "Francia",
    "ES": "Spagna",
    "NL": "Paesi Bassi",
    "PL": "Polonia",
    "EU27_2020": "Media UE",
}

YEAR_MIN = 1995

DATASET_GDP = "nama_10_gdp"
NA_ITEM_GDP = "B1GQ"

UNIT_REAL_PREFERRED = "CLV10_MEUR"
BASE_YEAR_PREFERRED = 2000


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_year_col(c: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", str(c).strip()))


def _find_geo_time_col(columns) -> str:
    cols = list(columns)
    if "geo\\TIME_PERIOD" in cols:
        return "geo\\TIME_PERIOD"
    for c in cols:
        sc = str(c)
        if ("geo" in sc.lower()) and ("time_period" in sc.lower()):
            return c
    for c in cols:
        sc = str(c)
        if ("geo" in sc.lower()) and ("time" in sc.lower()):
            return c
    for c in cols:
        if str(c).strip().lower() == "geo":
            return c
    raise ValueError("Colonna geo non trovata nel dataset")


def load_gdp_long() -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = eurostat.get_data(DATASET_GDP)
    if not raw or len(raw) < 2:
        raise RuntimeError("Download Eurostat fallito o dataset vuoto")

    header = raw[0]
    rows = raw[1:]
    df = pd.DataFrame(rows, columns=header)

    geo_col = _find_geo_time_col(df.columns)
    if geo_col != "geo\\TIME_PERIOD":
        df = df.rename(columns={geo_col: "geo\\TIME_PERIOD"})

    required = ["unit", "na_item", "geo\\TIME_PERIOD"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Colonna mancante nel dataset: {c}")

    df["na_item"] = df["na_item"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    df["geo\\TIME_PERIOD"] = df["geo\\TIME_PERIOD"].astype(str).str.strip()

    df = df[df["na_item"] == NA_ITEM_GDP].copy()
    df = df[df["geo\\TIME_PERIOD"].isin(PAESI)].copy()

    id_vars = [c for c in ["freq", "unit", "na_item", "geo\\TIME_PERIOD"] if c in df.columns]
    year_cols = [c for c in df.columns if c not in id_vars and _is_year_col(c)]
    if not year_cols:
        raise RuntimeError("Nessuna colonna anno trovata nel dataset dopo i filtri")

    long_df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = long_df["Year"].astype(str).str.strip()
    long_df["Value"] = _safe_num(long_df["Value"])

    long_df["Paese"] = long_df["geo\\TIME_PERIOD"].map(PAESI_LABEL).fillna(long_df["geo\\TIME_PERIOD"])
    long_df = long_df[["Year", "unit", "geo\\TIME_PERIOD", "Paese", "Value"]].copy()

    available_units = sorted(long_df["unit"].dropna().astype(str).unique().tolist())
    available_years = sorted(long_df["Year"].dropna().astype(str).unique().tolist())
    return long_df, available_years, available_units


def pick_unit(long_df: pd.DataFrame, preferred: str) -> str:
    units = sorted(long_df["unit"].dropna().astype(str).unique().tolist())
    if not units:
        raise RuntimeError("Nessuna unita disponibile nel dataset dopo i filtri")
    if preferred in units:
        return preferred
    for u in ["CLV10_MEUR", "CLV15_MEUR", "CLV20_MEUR", "CLV_I10", "CLV_I15", "CLV_I20"]:
        if u in units:
            return u
    return units[0]


def pivot_unit(long_df: pd.DataFrame, unit: str) -> pd.DataFrame:
    d = long_df[long_df["unit"].astype(str) == str(unit)].copy()
    p = d.pivot_table(index=["Year"], columns="geo\\TIME_PERIOD", values="Value", aggfunc="first").reset_index()
    p["Year"] = pd.to_numeric(p["Year"], errors="coerce")
    p = p.sort_values("Year")
    return p


def choose_base_year(p: pd.DataFrame, preferred: int) -> int:
    years = p["Year"].dropna().astype(int).tolist()
    if not years:
        return preferred
    if preferred in years:
        return preferred
    return min(years)


def last_common_year(p: pd.DataFrame, countries: list[str]) -> int | None:
    ok = p.copy()
    ok["ok"] = True
    for c in countries:
        if c not in ok.columns:
            ok["ok"] = False
        else:
            ok["ok"] = ok["ok"] & ok[c].notna()
    ok2 = ok[ok["ok"]].copy()
    if ok2.empty:
        yrs = p["Year"].dropna().astype(int).tolist()
        return max(yrs) if yrs else None
    return int(ok2["Year"].max())


def to_index_base100(p: pd.DataFrame, countries: list[str], base_year: int) -> pd.DataFrame:
    out = p.copy()
    base_row = out[out["Year"].astype(int) == int(base_year)]
    if base_row.empty:
        base_year = int(out["Year"].dropna().astype(int).min())
        base_row = out[out["Year"].astype(int) == int(base_year)]
        if base_row.empty:
            raise RuntimeError("Impossibile determinare base year per indice")

    base = base_row.iloc[0]
    for c in countries:
        if c in out.columns:
            b = base[c]
            out[c] = np.where(pd.notna(b) and b != 0, out[c] / b * 100.0, np.nan)
    return out


def yoy_growth(p: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    out = p.copy()
    out = out.sort_values("Year")
    for c in countries:
        if c in out.columns:
            out[c] = out[c].pct_change() * 100.0
    return out


def _val_at(df, col, year):
    r = df[df["Year"].astype(int) == int(year)]
    if r.empty or col not in r.columns:
        return np.nan
    v = r.iloc[0][col]
    return float(v) if pd.notna(v) else np.nan


def _nan_to_none(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def df_to_records(p: pd.DataFrame) -> list[dict]:
    out = []
    cols = [c for c in p.columns if c in ["Year"] + PAESI]
    p2 = p[cols].copy()
    for _, r in p2.iterrows():
        rec = {}
        for c in cols:
            rec[c] = _nan_to_none(r[c])
        out.append(rec)
    return out


def main():
    out_path = Path("docs/pil_dashboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    long_df, _, _ = load_gdp_long()
    unit_real = pick_unit(long_df, UNIT_REAL_PREFERRED)

    p_real = pivot_unit(long_df, unit_real)
    p_real = p_real[p_real["Year"].notna() & (p_real["Year"].astype(int) >= YEAR_MIN)].copy()

    base_year = choose_base_year(p_real, BASE_YEAR_PREFERRED)

    last_year = last_common_year(p_real, PAESI)
    if last_year is None:
        raise RuntimeError("Impossibile determinare ultimo anno")

    p_idx = to_index_base100(p_real, PAESI, base_year)
    p_yoy = yoy_growth(p_real, PAESI)

    it_last_real = _val_at(p_real, "IT", last_year)
    it_last_idx = _val_at(p_idx, "IT", last_year)
    it_last_yoy = _val_at(p_yoy, "IT", last_year)

    de_last_idx = _val_at(p_idx, "DE", last_year)
    es_last_idx = _val_at(p_idx, "ES", last_year)

    kpi = {
        "last_year": int(last_year),
        "it_last_real": _nan_to_none(it_last_real),
        "it_last_idx": _nan_to_none(it_last_idx),
        "it_last_yoy": _nan_to_none(it_last_yoy),
        "gap_de_it": _nan_to_none(de_last_idx - it_last_idx) if pd.notna(de_last_idx) and pd.notna(it_last_idx) else None,
        "gap_es_it": _nan_to_none(es_last_idx - it_last_idx) if pd.notna(es_last_idx) and pd.notna(it_last_idx) else None,
    }

    payload = {
        "meta": {
            "dataset": DATASET_GDP,
            "na_item": NA_ITEM_GDP,
            "unit_real": unit_real,
            "base_year": int(base_year),
            "year_min": int(YEAR_MIN),
            "last_year": int(last_year),
            "paesi": PAESI,
            "paesi_label": PAESI_LABEL,
        },
        "kpi": kpi,
        "real": df_to_records(p_real),
        "index": df_to_records(p_idx),
        "yoy": df_to_records(p_yoy),
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
