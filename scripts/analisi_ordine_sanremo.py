#!/usr/bin/env python3
"""
Sanremo — L'ordine di uscita influenza la classifica?

Workflow:
1) Scrape Wikipedia: ordine di uscita per serata (Campioni)
2) Integra con dati verificati locali (sanremo_verified_data.json)
3) Salva CSV: dati_sremo/sanremo_ordine_serate.csv
4) Analisi statistica

Uso:
  python3 scripts/analisi_ordine_sanremo.py
  python3 scripts/analisi_ordine_sanremo.py --skip-scrape
  python3 scripts/analisi_ordine_sanremo.py --years 2019 2020 2021
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dati_sremo")
VERIFIED_JSON = os.path.join(DATA_DIR, "sanremo_verified_data.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "sanremo_ordine_serate.csv")

UA = {"User-Agent": "Mozilla/5.0 (compatible; SanremoAnalysis/2.0)"}


# ============================================================
# Normalizzazione / matching artisti
# ============================================================

def norm_artist(name: str) -> str:
    s = str(name).lower().strip()
    s = re.sub(r"\s+feat\.?\s+", " ", s)
    s = re.sub(r"\s*&\s*", " e ", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def artist_match(a: str, b: str) -> bool:
    na, nb = norm_artist(a), norm_artist(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    overlap = wa & wb
    if overlap and len(overlap) / max(1, min(len(wa), len(wb))) >= 0.5:
        return True
    return False


def find_rank(artist: str, rankings: list[dict]) -> int | None:
    for r in rankings:
        if artist_match(artist, r.get("artist", "")):
            return r.get("rank")
    return None


# ============================================================
# Dati verificati locali (JSON)
# ============================================================

def load_verified_serate() -> list[dict]:
    if not os.path.exists(VERIFIED_JSON):
        return []

    with open(VERIFIED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for year_str, ydata in data.items():
        year = int(year_str)
        rankings = ydata.get("rankings", [])

        for serata_key, sdata in ydata.get("serate", {}).items():
            serata = int(serata_key)
            totale = sdata["num_performers"]

            for perf in sdata["performances"]:
                artist = perf["artist"]
                rank = find_rank(artist, rankings)

                rows.append({
                    "year": year,
                    "serata": serata,
                    "serata_name": sdata.get("name", f"Serata {serata}"),
                    "artist": artist,
                    "song": "",
                    "ordine": perf["order"],
                    "totale_serata": totale,
                    "classifica_serata": None,
                    "classifica_finale": rank,
                })
    return rows


def load_rankings() -> dict[int, list[dict]]:
    if not os.path.exists(VERIFIED_JSON):
        return {}

    with open(VERIFIED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: dict[int, list[dict]] = {}
    for year_str, ydata in data.items():
        result[int(year_str)] = ydata.get("rankings", [])
    return result


def load_expected_performers_by_year_serata() -> dict[tuple[int, int], int]:
    """
    Usa il JSON verificato solo come "prior" sul numero atteso di artisti in gara per serata.
    Non è obbligatorio: se manca, lo scraping va lo stesso.
    """
    exp: dict[tuple[int, int], int] = {}
    if not os.path.exists(VERIFIED_JSON):
        return exp

    with open(VERIFIED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for year_str, ydata in data.items():
        year = int(year_str)
        for serata_key, sdata in ydata.get("serate", {}).items():
            serata = int(serata_key)
            n = int(sdata.get("num_performers") or 0)
            if n > 0:
                exp[(year, serata)] = n
    return exp


# ============================================================
# Scraping Wikipedia (robusto + deterministico)
# ============================================================

SERATA_PATTERNS = [
    (re.compile(r"\bprima\s+serata\b", re.I), 1),
    (re.compile(r"\bseconda\s+serata\b", re.I), 2),
    (re.compile(r"\bterza\s+serata\b", re.I), 3),
    (re.compile(r"\bquarta\s+serata\b", re.I), 4),
    (re.compile(r"\bquinta\s+serata\b", re.I), 5),
    (re.compile(r"\bserata\s+finale\b", re.I), 5),
    (re.compile(r"\bquinta\s+serata\s*-\s*finale\b", re.I), 5),
    (re.compile(r"\bfinale\b", re.I), 5),
]

BAD_CONTEXT_KEYWORDS = (
    "nuove proposte", "giovani", "sanremo giovani",
    "ospiti", "orchestra", "premi",
    "cover", "duetti", "medley",
    "finale a tre", "finale a 3",
    "classifica provvisoria", "classifica parziale",
    "ripesc", "semifinal", "finalissima",
)

ARTIST_KEYS = ("interprete", "interpreti", "artista", "artisti", "cantante", "cantanti")
SONG_KEYS = ("brano", "canzone", "titolo", "titolo del brano")
ORDER_KEYS = ("ordine di uscita", "ordine", "n.", "n°", "n", "ord", "#")
RANK_KEYS = ("classifica", "pos.", "posizione", "pos", "posto", "piazz", "graduatoria")


def _t(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _serata_from_heading(text: str) -> int | None:
    t = _t(text).lower()
    for rx, n in SERATA_PATTERNS:
        if rx.search(t):
            return n
    m = re.search(r"\bserata\b\s*(\d)\b", t)
    if m:
        k = int(m.group(1))
        if 1 <= k <= 5:
            return k
    return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(str(c) for c in col).strip() for col in df.columns]
    return df


def _pick_col(df: pd.DataFrame, keys: tuple[str, ...]) -> str | None:
    cols = [str(c).strip().lower() for c in df.columns]
    for j, c in enumerate(cols):
        if any(k in c for k in keys):
            return df.columns[j]
    return None


def _looks_like_candidate(df: pd.DataFrame) -> bool:
    df = _flatten_columns(df)
    cols = [str(c).strip().lower() for c in df.columns]
    has_artist = any(any(k in c for k in ARTIST_KEYS) for c in cols)
    has_song = any(any(k in c for k in SONG_KEYS) for c in cols)
    return len(df) >= 3 and has_artist and has_song


def _first_col_seems_order(df: pd.DataFrame) -> bool:
    """
    True se la prima colonna sembra un ordine 1..n (anche se l'header è strano).
    """
    if df.empty:
        return False
    first = df.iloc[:, 0].astype(str).str.replace(r"[^\d]", "", regex=True)
    nums = pd.to_numeric(first, errors="coerce")
    nums = nums.dropna()
    if len(nums) < max(5, int(0.5 * len(df))):
        return False
    # sequenza che parte da 1 con pochi buchi
    s = set(int(x) for x in nums.tolist() if int(x) > 0)
    if 1 not in s:
        return False
    top = max(s)
    # copertura sufficiente
    coverage = len(s.intersection(set(range(1, top + 1)))) / max(1, top)
    return coverage >= 0.75


def _context_penalty(text: str) -> int:
    t = _t(text).lower()
    return sum(1 for k in BAD_CONTEXT_KEYWORDS if k in t)


def _score_table(df: pd.DataFrame, html_ctx: str, expected_n: int | None) -> int:
    """
    Score deterministico: più alto = più probabile "Ordine di uscita Campioni" della serata.
    """
    df = _flatten_columns(df)
    cols = [str(c).strip().lower() for c in df.columns]

    score = 0

    # base: artista + brano
    has_artist = any(any(k in c for k in ARTIST_KEYS) for c in cols)
    has_song = any(any(k in c for k in SONG_KEYS) for c in cols)
    if has_artist:
        score += 10
    if has_song:
        score += 8

    # colonna ordine esplicita
    has_order_col = any(any(k in c for k in ORDER_KEYS) for c in cols)
    if has_order_col:
        score += 8

    # prima colonna sembra 1..n
    if _first_col_seems_order(df):
        score += 6

    # dimensione: vicino all'atteso
    n = len(df)
    if expected_n is not None and expected_n > 0:
        diff = abs(n - expected_n)
        if diff == 0:
            score += 10
        elif diff <= 1:
            score += 7
        elif diff <= 3:
            score += 4
        else:
            score -= min(10, diff)
    else:
        # range plausibile (serate spezzate + serate piene)
        if 8 <= n <= 45:
            score += 2
        else:
            score -= 6

    # penalità contesto (ospiti/nuove proposte/finale a tre/cover/...)
    score -= 6 * _context_penalty(html_ctx)

    return score


def scrape_year(year: int, expected: dict[tuple[int, int], int]) -> list[dict]:
    """
    Strategia robusta:
    - segmenta la pagina per heading "Prima/Seconda/.../Finale"
    - dentro ogni serata: valuta TUTTE le tabelle e prende la migliore per score
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://it.wikipedia.org/wiki/Festival_di_Sanremo_{year}"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  [Wikipedia] Errore fetch {year}: {e}")
        return []

    html = r.text
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Trova tutti gli heading e costruisci le "sezioni serata"
    headings = soup.find_all(["h2", "h3", "h4", "h5"])
    serata_heads: list[tuple[int, str, object]] = []
    for h in headings:
        txt = _t(h.get_text(" ", strip=True))
        s = _serata_from_heading(txt)
        if s is not None:
            serata_heads.append((s, txt, h))

    # Dedup: spesso Wikipedia ripete "Finale" o simili
    # teniamo il primo heading per (serata, testo normalizzato)
    uniq = []
    seen = set()
    for s, txt, h in serata_heads:
        key = (s, txt.lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((s, txt, h))
    serata_heads = uniq

    # Se per qualche motivo non trovi heading serata, fallback: scansiona tutte le tabelle (ma con score)
    if not serata_heads:
        rows = _fallback_scan_all_tables_for_year(year, soup, expected)
        print(f"  [Wikipedia] {year}: {len(rows)} esibizioni trovate (fallback)")
        return rows

    rows: list[dict] = []
    used_keys = set()

    # Per ogni serata: raccogli tabelle finché non arriva un altro heading di serata
    for idx, (serata_num, serata_name, hnode) in enumerate(serata_heads):
        next_h = serata_heads[idx + 1][2] if idx + 1 < len(serata_heads) else None

        section_tables = []
        cur = hnode
        while True:
            cur = cur.find_next()  # type: ignore
            if cur is None:
                break
            if next_h is not None and cur == next_h:
                break
            if getattr(cur, "name", None) == "table":
                section_tables.append(cur)

        if not section_tables:
            continue

        # scegli la tabella migliore per score
        best = None  # (score, df, tbl_ctx_html)
        expected_n = expected.get((year, serata_num))

        for tbl in section_tables:
            # contesto locale: caption + 2 heading non-serata prima della tabella
            ctx_parts = []
            cap = tbl.find("caption")
            if cap:
                ctx_parts.append(cap.get_text(" ", strip=True))

            # heading non-serata precedenti (max 2)
            hprev = tbl.find_previous(["h2", "h3", "h4", "h5"])
            count = 0
            while hprev is not None and count < 6 and len(ctx_parts) < 3:
                ttxt = _t(hprev.get_text(" ", strip=True))
                if _serata_from_heading(ttxt) is None:
                    ctx_parts.append(ttxt)
                count += 1
                hprev = hprev.find_previous(["h2", "h3", "h4", "h5"])
            ctx = " ".join(ctx_parts)

            tbl_html = str(tbl)
            try:
                dfs = pd.read_html(io.StringIO(tbl_html))
            except Exception:
                continue

            for df in dfs:
                if not _looks_like_candidate(df):
                    continue
                df = _flatten_columns(df)
                sc = _score_table(df, ctx, expected_n)
                if best is None or sc > best[0]:
                    best = (sc, df, ctx)

        if best is None:
            continue

        _, df_best, _ = best
        df_best = _flatten_columns(df_best)

        artist_col = _pick_col(df_best, ARTIST_KEYS)
        song_col = _pick_col(df_best, SONG_KEYS)
        order_col = _pick_col(df_best, ORDER_KEYS)
        rank_col = _pick_col(df_best, RANK_KEYS)

        if artist_col is None:
            continue

        totale = len(df_best)

        for irow, row in df_best.iterrows():
            artist_raw = str(row.get(artist_col, "")).strip()
            if not artist_raw or artist_raw.lower() == "nan":
                continue
            artist = re.sub(r"\[.*?\]", "", artist_raw).strip()

            song = ""
            if song_col is not None:
                song = re.sub(r"\[.*?\]", "", str(row.get(song_col, "")).strip())
                if song.lower() == "nan":
                    song = ""

            ordine = irow + 1
            if order_col is not None:
                try:
                    ordine = int(float(str(row.get(order_col, "")).strip()))
                except (ValueError, TypeError):
                    pass
            else:
                # fallback: prova dalla prima colonna se sembra numerica
                try:
                    v = str(row.iloc[0])
                    v = re.sub(r"[^\d]", "", v)
                    if v:
                        ordine = int(v)
                except Exception:
                    pass

            serata_rank = None
            if rank_col is not None:
                try:
                    serata_rank = int(float(str(row.get(rank_col, "")).strip()))
                except (ValueError, TypeError):
                    pass

            key = (year, serata_num, norm_artist(artist), ordine)
            if key in used_keys:
                continue
            used_keys.add(key)

            rows.append({
                "year": year,
                "serata": serata_num,
                "serata_name": serata_name or f"Serata {serata_num}",
                "artist": artist,
                "song": song,
                "ordine": ordine,
                "totale_serata": totale,
                "classifica_serata": serata_rank,
            })

    print(f"  [Wikipedia] {year}: {len(rows)} esibizioni trovate")
    return rows


def _fallback_scan_all_tables_for_year(year: int, soup, expected: dict[tuple[int, int], int]) -> list[dict]:
    """
    Fallback raro: se non trovi heading serate, scansiona tutte le tabelle e prova ad assegnare serata
    dal contesto vicino. È meno affidabile, ma sempre deterministico.
    """
    rows: list[dict] = []
    used = set()

    for tbl in soup.find_all("table"):
        # prova a dedurre serata dai heading precedenti
        serata_num = None
        serata_name = None
        h = tbl.find_previous(["h2", "h3", "h4", "h5"])
        while h is not None and serata_num is None:
            txt = _t(h.get_text(" ", strip=True))
            s = _serata_from_heading(txt)
            if s is not None:
                serata_num = s
                serata_name = txt
                break
            h = h.find_previous(["h2", "h3", "h4", "h5"])
        if serata_num is None:
            continue

        ctx = ""
        cap = tbl.find("caption")
        if cap:
            ctx += " " + cap.get_text(" ", strip=True)

        expected_n = expected.get((year, serata_num))

        try:
            dfs = pd.read_html(io.StringIO(str(tbl)))
        except Exception:
            continue

        best = None
        for df in dfs:
            if not _looks_like_candidate(df):
                continue
            df = _flatten_columns(df)
            sc = _score_table(df, ctx, expected_n)
            if best is None or sc > best[0]:
                best = (sc, df)
        if best is None:
            continue

        df_best = _flatten_columns(best[1])
        artist_col = _pick_col(df_best, ARTIST_KEYS)
        song_col = _pick_col(df_best, SONG_KEYS)
        order_col = _pick_col(df_best, ORDER_KEYS)

        if artist_col is None:
            continue

        totale = len(df_best)
        for irow, row in df_best.iterrows():
            artist_raw = str(row.get(artist_col, "")).strip()
            if not artist_raw or artist_raw.lower() == "nan":
                continue
            artist = re.sub(r"\[.*?\]", "", artist_raw).strip()

            song = ""
            if song_col is not None:
                song = re.sub(r"\[.*?\]", "", str(row.get(song_col, "")).strip())
                if song.lower() == "nan":
                    song = ""

            ordine = irow + 1
            if order_col is not None:
                try:
                    ordine = int(float(str(row.get(order_col, "")).strip()))
                except (ValueError, TypeError):
                    pass

            key = (year, serata_num, norm_artist(artist), ordine)
            if key in used:
                continue
            used.add(key)

            rows.append({
                "year": year,
                "serata": serata_num,
                "serata_name": serata_name or f"Serata {serata_num}",
                "artist": artist,
                "song": song,
                "ordine": ordine,
                "totale_serata": totale,
                "classifica_serata": None,
            })

    return rows


# ============================================================
# Merge: Wikipedia + Verified
# ============================================================

def merge_data(
    wiki_rows: list[dict],
    verified_rows: list[dict],
    rankings_by_year: dict[int, list[dict]],
) -> list[dict]:
    verified_idx = set()
    all_rows: list[dict] = []

    for row in verified_rows:
        key = (row["year"], row["serata"], norm_artist(row["artist"]))
        verified_idx.add(key)
        all_rows.append(row)

    for row in wiki_rows:
        key = (row["year"], row["serata"], norm_artist(row["artist"]))
        if key not in verified_idx:
            rankings = rankings_by_year.get(row["year"], [])
            row["classifica_finale"] = find_rank(row["artist"], rankings)
            all_rows.append(row)
            verified_idx.add(key)

    # riempi eventuali classifica_finale mancanti
    for row in all_rows:
        if row.get("classifica_finale") is None:
            rankings = rankings_by_year.get(row["year"], [])
            row["classifica_finale"] = find_rank(row["artist"], rankings)

    return all_rows


# ============================================================
# CSV output + coverage report
# ============================================================

def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print("Nessun dato da salvare.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    cols = [
        "year",
        "serata",
        "serata_name",
        "artist",
        "song",
        "ordine",
        "totale_serata",
        "posizione_relativa",
        "classifica_serata",
        "classifica_finale",
    ]

    for row in rows:
        tot = int(row.get("totale_serata") or 0)
        ord_ = int(row.get("ordine") or 0)
        row["posizione_relativa"] = round((ord_ - 1) / max(tot - 1, 1), 4)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["year"], r["serata"], r["ordine"], norm_artist(r["artist"]))))

    print(f"\n  Salvato: {path} ({len(rows)} righe)")


def print_coverage(df: pd.DataFrame, years: list[int]) -> None:
    print("\n" + "=" * 70)
    print("COPERTURA SERATE (da CSV)")
    print("=" * 70)

    g = df.groupby(["year", "serata"]).size().sort_index()
    print(g)

    for y in years:
        present = set(int(s) for s in df[df["year"] == y]["serata"].unique())
        missing = [s for s in [1, 2, 3, 4, 5] if s not in present]
        if missing:
            print(f"  ⚠️ {y}: mancano serate {missing}")


# ============================================================
# Analisi
# ============================================================

def run_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("ANALISI: POSIZIONE CENTRALE NELL'ORDINE DI USCITA → CLASSIFICA MIGLIORE?")
    print("=" * 70)

    usable = df[df["classifica_finale"].notna()].copy()
    usable["classifica_finale"] = usable["classifica_finale"].astype(int)

    years = sorted(int(y) for y in usable["year"].unique())
    print(f"\nAnni con dati: {years}")
    print(f"Esibizioni totali con classifica: {len(usable)}")
    for y in years:
        ydf = usable[usable["year"] == y]
        serate = sorted(int(s) for s in ydf["serata"].unique())
        print(f"  {y}: {len(ydf)} esibizioni, serate {serate}")

    print("\n" + "─" * 60)
    print("A) POSIZIONE NELL'ORDINE DI USCITA: INIZIO vs CENTRO vs FINE")
    print("─" * 60)

    usable["fascia"] = pd.cut(
        usable["posizione_relativa"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Inizio (1/3)", "Centro (2/3)", "Fine (3/3)"],
    )

    print(f"\n  Anno   {'Fascia':<16} {'n':>4}  {'Rank medio':>11}  {'Mediana':>8}  {'Top 5':>6}  {'Top 10':>7}")
    print(f"  {'─' * 72}")

    for y in years:
        ydf = usable[usable["year"] == y]
        for fascia in ["Inizio (1/3)", "Centro (2/3)", "Fine (3/3)"]:
            g = ydf[ydf["fascia"] == fascia]
            if len(g) == 0:
                continue
            avg = g["classifica_finale"].mean()
            med = g["classifica_finale"].median()
            top5 = (g["classifica_finale"] <= 5).mean() * 100
            top10 = (g["classifica_finale"] <= 10).mean() * 100
            print(f"  {y}   {fascia:<16} {len(g):>4}  {avg:>11.1f}  {med:>8.1f}  {top5:>5.0f}%  {top10:>6.0f}%")
        print()

    print("  TOTALE")
    for fascia in ["Inizio (1/3)", "Centro (2/3)", "Fine (3/3)"]:
        g = usable[usable["fascia"] == fascia]
        if len(g) == 0:
            continue
        avg = g["classifica_finale"].mean()
        med = g["classifica_finale"].median()
        top5 = (g["classifica_finale"] <= 5).mean() * 100
        top10 = (g["classifica_finale"] <= 10).mean() * 100
        print(f"  TUTTI  {fascia:<16} {len(g):>4}  {avg:>11.1f}  {med:>8.1f}  {top5:>5.0f}%  {top10:>6.0f}%")

    print("\n" + "─" * 60)
    print("B) CORRELAZIONE: POSIZIONE RELATIVA vs CLASSIFICA FINALE")
    print("─" * 60)

    for y in years:
        ydf = usable[usable["year"] == y]
        for serata in sorted(ydf["serata"].unique()):
            sdf = ydf[ydf["serata"] == serata]
            if len(sdf) < 5:
                continue
            rho, p = stats.spearmanr(sdf["posizione_relativa"], sdf["classifica_finale"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {y} S{serata} (n={len(sdf):>2}):  Spearman rho = {rho:+.3f}  p = {p:.4f} {sig}")

    print()
    rho, p = stats.spearmanr(usable["posizione_relativa"], usable["classifica_finale"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  GLOBALE (n={len(usable)}):  Spearman rho = {rho:+.3f}  p = {p:.4f} {sig}")

    if p < 0.05:
        direction = "più basse (peggiori)" if rho > 0 else "più alte (migliori)"
        print(f"  → Significativo: esibirsi più tardi → posizioni {direction}")
    else:
        print("  → Non significativo: l'ordine di uscita NON correla con la classifica")

    print("\n" + "─" * 60)
    print("C) TEST: CENTRO vs ESTREMI (INIZIO + FINE)")
    print("─" * 60)

    centro = usable[usable["fascia"] == "Centro (2/3)"]["classifica_finale"]
    estremi = usable[usable["fascia"] != "Centro (2/3)"]["classifica_finale"]

    if len(centro) >= 3 and len(estremi) >= 3:
        stat_u, p_mw = stats.mannwhitneyu(centro, estremi, alternative="two-sided")
        print(f"\n  Centro:  n={len(centro):>3}, media={centro.mean():.1f}, mediana={centro.median():.1f}")
        print(f"  Estremi: n={len(estremi):>3}, media={estremi.mean():.1f}, mediana={estremi.median():.1f}")
        print(f"\n  Mann-Whitney U = {stat_u:.1f},  p = {p_mw:.4f}")
        if p_mw < 0.05:
            if centro.mean() < estremi.mean():
                print("  → SIGNIFICATIVO: chi si esibisce al centro ha rank MIGLIORE")
            else:
                print("  → SIGNIFICATIVO: chi si esibisce al centro ha rank PEGGIORE")
        else:
            print("  → NON significativo (p > 0.05): nessuna differenza centro/estremi")

        top5_centro = (centro <= 5).mean()
        top5_estremi = (estremi <= 5).mean()
        print(f"\n  % Top 5:  Centro {top5_centro*100:.1f}% vs Estremi {top5_estremi*100:.1f}%")

        ct = np.array(
            [
                [(centro <= 5).sum(), (centro > 5).sum()],
                [(estremi <= 5).sum(), (estremi > 5).sum()],
            ]
        )
        if ct.min() > 0:
            chi2, p_chi, _, _ = stats.chi2_contingency(ct)
            print(f"  Chi² (Top 5 centro vs estremi) = {chi2:.3f}, p = {p_chi:.4f}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Sanremo: ordine di uscita vs classifica")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2019, 2027)),
        help="Anni da analizzare (default: 2019-2026)",
    )
    parser.add_argument("--skip-scrape", action="store_true", help="Salta lo scraping Wikipedia")
    args = parser.parse_args()

    print("=" * 70)
    print("SANREMO — L'ORDINE DI USCITA INFLUENZA LA CLASSIFICA?")
    print("=" * 70)

    print("\n[1/4] Caricamento dati locali verificati...")
    verified_rows = load_verified_serate()
    rankings = load_rankings()
    expected = load_expected_performers_by_year_serata()
    print(f"  Esibizioni da JSON verificato: {len(verified_rows)}")
    print(f"  Anni con classifica: {sorted(rankings.keys())}")

    wiki_rows: list[dict] = []
    if not args.skip_scrape:
        print(f"\n[2/4] Scraping Wikipedia per anni: {args.years}...")
        for year in args.years:
            wiki_rows.extend(scrape_year(year, expected))
        print(f"  Totale da Wikipedia: {len(wiki_rows)} esibizioni")
    else:
        print("\n[2/4] Scraping Wikipedia saltato (--skip-scrape)")

    print("\n[3/4] Unione dati e salvataggio CSV...")
    all_rows = merge_data(wiki_rows, verified_rows, rankings)
    if not all_rows:
        print("ERRORE: Nessun dato disponibile!")
        sys.exit(1)

    save_csv(all_rows, OUTPUT_CSV)
    with_rank = sum(1 for r in all_rows if r.get("classifica_finale") is not None)
    print(f"  Esibizioni con classifica finale: {with_rank}")

    df = pd.read_csv(OUTPUT_CSV)
    print_coverage(df, args.years)

    print("\n[4/4] Analisi statistica...")
    run_analysis(df)

    print("\n" + "=" * 70)
    print("FATTO")
    print("=" * 70)
    print(f"CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()