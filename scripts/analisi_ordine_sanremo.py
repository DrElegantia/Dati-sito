#!/usr/bin/env python3
"""
Sanremo — L'ordine di uscita influenza la classifica?

Ipotesi: esibirsi nelle posizioni centrali della scaletta dà più
probabilità di piazzarsi in alto in classifica.

Workflow:
  1. Scarica da Wikipedia ordine di uscita + classifica per ogni serata
     di ogni edizione (2020-2026).
  2. Se Wikipedia non è raggiungibile, usa i dati da
     dati_sremo/sanremo_verified_data.json + dati hardcoded.
  3. Salva tutto in un CSV (dati_sremo/sanremo_ordine_serate.csv).
  4. Esegue l'analisi statistica.

Uso:
    python3 scripts/analisi_ordine_sanremo.py
    python3 scripts/analisi_ordine_sanremo.py --skip-scrape   # usa solo dati locali
    python3 scripts/analisi_ordine_sanremo.py --years 2023 2024 2025
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dati_sremo')
VERIFIED_JSON = os.path.join(DATA_DIR, 'sanremo_verified_data.json')
OUTPUT_CSV = os.path.join(DATA_DIR, 'sanremo_ordine_serate.csv')

UA = {'User-Agent': 'Mozilla/5.0 (compatible; SanremoAnalysis/1.0)'}


# ============================================================
# 1. Scraping Wikipedia
# ============================================================

def norm_artist(name):
    """Normalize artist name for matching."""
    s = name.lower().strip()
    s = re.sub(r'\s+feat\.?\s+', ' ', s)
    s = re.sub(r'\s*&\s*', ' e ', s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def artist_match(a, b):
    na, nb = norm_artist(a), norm_artist(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    overlap = wa & wb
    if overlap and len(overlap) / min(len(wa), len(wb)) >= 0.5:
        return True
    return False


def find_rank(artist, rankings):
    """Find final rank for artist from rankings list."""
    for r in rankings:
        if artist_match(artist, r['artist']):
            return r['rank']
    return None


def scrape_year(year):
    """Scrape per-serata performance order from Italian Wikipedia.

    Returns list of dicts: {year, serata, artist, song, ordine, totale, classifica_serata}
    """
    import requests

    url = f'https://it.wikipedia.org/wiki/Festival_di_Sanremo_{year}'
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f'  [Wikipedia] Errore fetch {year}: {e}')
        return []

    html = r.text
    rows = []

    # Strategy: parse ALL tables, then identify serata tables by:
    # 1. Column names containing "interprete"/"artista" AND position-like column
    # 2. Context from surrounding HTML (section headings)

    # Extract section headings and their positions to map tables to serate
    serata_sections = []
    for m in re.finditer(
        r'<h[23][^>]*>\s*<span[^>]*>([^<]*(?:serata|finale)[^<]*)</span>',
        html, re.IGNORECASE
    ):
        name = re.sub(r'\s+', ' ', m.group(1)).strip()
        serata_sections.append((m.start(), name))

    # Also match alternative heading formats
    for m in re.finditer(
        r'<h[23][^>]*>\s*(?:<span[^>]*>)*\s*'
        r'((?:Prima|Seconda|Terza|Quarta|Quinta)\s+serata'
        r'|Serata\s+finale)',
        html, re.IGNORECASE
    ):
        name = re.sub(r'\s+', ' ', m.group(1)).strip()
        # Avoid duplicates
        if not any(abs(m.start() - pos) < 100 for pos, _ in serata_sections):
            serata_sections.append((m.start(), name))

    serata_sections.sort()

    # Parse all tables -- use StringIO so pandas doesn't misinterpret the
    # raw HTML string as a URL/path.  Catch ImportError (missing lxml) and
    # any other parsing failure.
    try:
        all_tables = pd.read_html(io.StringIO(html))
    except (ValueError, ImportError) as e:
        print(f'  [Wikipedia] Nessuna tabella trovata per {year}: {e}')
        return []
    except Exception as e:
        print(f'  [Wikipedia] Errore parsing tabelle {year}: {e}')
        return []

    # Find table positions in HTML for mapping to sections
    table_positions = [m.start() for m in re.finditer(r'<table', html, re.IGNORECASE)]

    # Map serata name to number
    SERATA_MAP = {
        'prima': 1, 'seconda': 2, 'terza': 3,
        'quarta': 4, 'quinta': 5, 'finale': 5,
    }

    def get_serata_num(section_name):
        for key, num in SERATA_MAP.items():
            if key in section_name.lower():
                return num
        return None

    def get_serata_for_table(table_idx):
        """Find which serata section a table belongs to."""
        if table_idx >= len(table_positions):
            return None, None
        tpos = table_positions[table_idx]
        best = None
        for spos, sname in serata_sections:
            if spos < tpos:
                best = sname
        if best:
            return get_serata_num(best), best
        return None, None

    # Identify and parse performance order tables
    for i, df in enumerate(all_tables):
        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(str(c) for c in col).strip() for col in df.columns]

        cols_lower = [str(c).strip().lower() for c in df.columns]

        # Check if this looks like a performance order table
        has_artist = any(c in cols_lower for c in
                         ('interprete', 'artista', 'cantante', 'interpreti'))
        has_song = any(c in cols_lower for c in
                       ('brano', 'canzone', 'titolo'))

        if not (has_artist and has_song):
            continue
        if len(df) < 3:
            continue

        serata_num, serata_name = get_serata_for_table(i)
        if serata_num is None:
            continue

        # Find the right columns
        artist_col = None
        song_col = None
        order_col = None
        rank_col = None

        for j, c in enumerate(cols_lower):
            if c in ('interprete', 'artista', 'cantante', 'interpreti'):
                artist_col = df.columns[j]
            elif c in ('brano', 'canzone', 'titolo'):
                song_col = df.columns[j]
            elif c in ('n.', 'n', 'ordine', '#', 'ord.'):
                order_col = df.columns[j]
            elif c in ('classifica', 'pos.', 'posizione', 'class.'):
                rank_col = df.columns[j]

        if artist_col is None:
            continue

        totale = len(df)
        for idx, row in df.iterrows():
            artist_raw = str(row[artist_col]).strip()
            if not artist_raw or artist_raw == 'nan':
                continue
            # Clean artist name
            artist = re.sub(r'\[.*?\]', '', artist_raw).strip()
            song = ''
            if song_col is not None:
                song = re.sub(r'\[.*?\]', '', str(row[song_col])).strip()
                if song == 'nan':
                    song = ''

            ordine = idx + 1  # default: row order = performance order
            if order_col is not None:
                try:
                    ordine = int(float(str(row[order_col]).strip()))
                except (ValueError, TypeError):
                    pass

            serata_rank = None
            if rank_col is not None:
                try:
                    serata_rank = int(float(str(row[rank_col]).strip()))
                except (ValueError, TypeError):
                    pass

            rows.append({
                'year': year,
                'serata': serata_num,
                'serata_name': serata_name or f'Serata {serata_num}',
                'artist': artist,
                'song': song,
                'ordine': ordine,
                'totale_serata': totale,
                'classifica_serata': serata_rank,
            })

    print(f'  [Wikipedia] {year}: {len(rows)} esibizioni trovate')
    return rows


# ============================================================
# 2. Dati dal JSON verificato (locale)
# ============================================================

def load_verified_serate():
    """Load per-serata data from sanremo_verified_data.json."""
    if not os.path.exists(VERIFIED_JSON):
        return []

    with open(VERIFIED_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    for year_str, ydata in data.items():
        year = int(year_str)
        rankings = ydata.get('rankings', [])

        for serata_key, sdata in ydata.get('serate', {}).items():
            serata = int(serata_key)
            totale = sdata['num_performers']

            for perf in sdata['performances']:
                artist = perf['artist']
                rank = find_rank(artist, rankings)

                rows.append({
                    'year': year,
                    'serata': serata,
                    'serata_name': sdata.get('name', f'Serata {serata}'),
                    'artist': artist,
                    'song': '',
                    'ordine': perf['order'],
                    'totale_serata': totale,
                    'classifica_serata': None,
                    'classifica_finale': rank,
                })

    return rows


def load_rankings():
    """Load final rankings from verified JSON."""
    if not os.path.exists(VERIFIED_JSON):
        return {}

    with open(VERIFIED_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for year_str, ydata in data.items():
        result[int(year_str)] = ydata.get('rankings', [])
    return result


# ============================================================
# 3. Merge e salvataggio
# ============================================================

def merge_data(wiki_rows, verified_rows, rankings_by_year):
    """Merge Wikipedia scraped data with verified local data.
    Verified data takes priority. Add final rankings where missing."""

    # Index verified data by (year, serata, norm_artist)
    verified_idx = set()
    all_rows = []

    for row in verified_rows:
        key = (row['year'], row['serata'], norm_artist(row['artist']))
        verified_idx.add(key)
        all_rows.append(row)

    # Add wiki data that's not already in verified
    for row in wiki_rows:
        key = (row['year'], row['serata'], norm_artist(row['artist']))
        if key not in verified_idx:
            # Add final ranking
            rankings = rankings_by_year.get(row['year'], [])
            row['classifica_finale'] = find_rank(row['artist'], rankings)
            all_rows.append(row)
            verified_idx.add(key)

    # Ensure all rows have classifica_finale
    for row in all_rows:
        if 'classifica_finale' not in row or row['classifica_finale'] is None:
            rankings = rankings_by_year.get(row['year'], [])
            row['classifica_finale'] = find_rank(row['artist'], rankings)

    return all_rows


def save_csv(rows, path):
    """Save rows to CSV."""
    if not rows:
        print('Nessun dato da salvare.')
        return

    cols = ['year', 'serata', 'serata_name', 'artist', 'song',
            'ordine', 'totale_serata', 'posizione_relativa',
            'classifica_serata', 'classifica_finale']

    # Compute relative position
    for row in rows:
        tot = row['totale_serata']
        ord_ = row['ordine']
        row['posizione_relativa'] = round((ord_ - 1) / max(tot - 1, 1), 4)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['year'], r['serata'], r['ordine'])))

    print(f'\n  Salvato: {path} ({len(rows)} righe)')


# ============================================================
# 4. Analisi statistica
# ============================================================

def run_analysis(df):
    """Test: posizioni centrali → migliore classifica?"""

    print(f"\n{'='*70}")
    print("ANALISI: POSIZIONE CENTRALE NELL'ORDINE DI USCITA → CLASSIFICA MIGLIORE?")
    print(f"{'='*70}")

    # Filter: only rows with final ranking
    usable = df[df['classifica_finale'].notna()].copy()
    usable['classifica_finale'] = usable['classifica_finale'].astype(int)

    years = sorted(int(y) for y in usable['year'].unique())
    print(f"\nAnni con dati: {years}")
    print(f"Esibizioni totali con classifica: {len(usable)}")

    for y in years:
        ydf = usable[usable['year'] == y]
        serate = sorted(int(s) for s in ydf['serata'].unique())
        print(f"  {y}: {len(ydf)} esibizioni, serate {serate}")

    # === A) Dividere in terzi: inizio / centro / fine ===
    print(f"\n{'─'*60}")
    print("A) POSIZIONE NELL'ORDINE DI USCITA: INIZIO vs CENTRO vs FINE")
    print(f"{'─'*60}")

    usable['fascia'] = pd.cut(
        usable['posizione_relativa'],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=['Inizio (1/3)', 'Centro (2/3)', 'Fine (3/3)']
    )

    # Per ogni anno, calcola il rank medio per fascia
    print(f"\n  Anno   {'Fascia':<16} {'n':>4}  {'Rank medio':>11}  {'Mediana':>8}  {'Top 5':>6}  {'Top 10':>7}")
    print(f"  {'─'*72}")

    for y in years:
        ydf = usable[usable['year'] == y]
        n_artists = ydf['classifica_finale'].max()
        for fascia in ['Inizio (1/3)', 'Centro (2/3)', 'Fine (3/3)']:
            g = ydf[ydf['fascia'] == fascia]
            if len(g) == 0:
                continue
            avg = g['classifica_finale'].mean()
            med = g['classifica_finale'].median()
            top5 = (g['classifica_finale'] <= 5).mean() * 100
            top10 = (g['classifica_finale'] <= 10).mean() * 100
            print(f"  {y}   {fascia:<16} {len(g):>4}  {avg:>11.1f}  {med:>8.1f}  {top5:>5.0f}%  {top10:>6.0f}%")
        print()

    # Aggregato su tutti gli anni
    print(f"  {'TOTALE':<7}", end='')
    print()
    for fascia in ['Inizio (1/3)', 'Centro (2/3)', 'Fine (3/3)']:
        g = usable[usable['fascia'] == fascia]
        if len(g) == 0:
            continue
        avg = g['classifica_finale'].mean()
        med = g['classifica_finale'].median()
        top5 = (g['classifica_finale'] <= 5).mean() * 100
        top10 = (g['classifica_finale'] <= 10).mean() * 100
        print(f"  TUTTI  {fascia:<16} {len(g):>4}  {avg:>11.1f}  {med:>8.1f}  {top5:>5.0f}%  {top10:>6.0f}%")

    # === B) Correlazione ordine vs classifica ===
    print(f"\n{'─'*60}")
    print("B) CORRELAZIONE: POSIZIONE RELATIVA vs CLASSIFICA FINALE")
    print(f"{'─'*60}")

    # Per serata
    for y in years:
        ydf = usable[usable['year'] == y]
        for serata in sorted(ydf['serata'].unique()):
            sdf = ydf[ydf['serata'] == serata]
            if len(sdf) < 5:
                continue
            rho, p = stats.spearmanr(sdf['posizione_relativa'], sdf['classifica_finale'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {y} S{serata} (n={len(sdf):>2}):  "
                  f"Spearman rho = {rho:+.3f}  p = {p:.4f} {sig}")

    # Globale
    print()
    rho, p = stats.spearmanr(usable['posizione_relativa'], usable['classifica_finale'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  GLOBALE (n={len(usable)}):  Spearman rho = {rho:+.3f}  p = {p:.4f} {sig}")

    if p < 0.05:
        direction = 'più basse (peggiori)' if rho > 0 else 'più alte (migliori)'
        print(f"  → Significativo: esibirsi più tardi → posizioni {direction}")
    else:
        print("  → Non significativo: l'ordine di uscita NON correla con la classifica")

    # === C) Test specifico: centro vs estremi ===
    print(f"\n{'─'*60}")
    print("C) TEST: CENTRO vs ESTREMI (INIZIO + FINE)")
    print(f"{'─'*60}")

    centro = usable[usable['fascia'] == 'Centro (2/3)']['classifica_finale']
    estremi = usable[usable['fascia'] != 'Centro (2/3)']['classifica_finale']

    if len(centro) >= 3 and len(estremi) >= 3:
        # Mann-Whitney U test (non-parametric)
        stat, p = stats.mannwhitneyu(centro, estremi, alternative='two-sided')
        print(f"\n  Centro:  n={len(centro):>3}, media={centro.mean():.1f}, mediana={centro.median():.1f}")
        print(f"  Estremi: n={len(estremi):>3}, media={estremi.mean():.1f}, mediana={estremi.median():.1f}")
        print(f"\n  Mann-Whitney U = {stat:.1f},  p = {p:.4f}")
        if p < 0.05:
            if centro.mean() < estremi.mean():
                print(f"  → SIGNIFICATIVO: chi si esibisce al centro ha rank MIGLIORE")
            else:
                print(f"  → SIGNIFICATIVO: chi si esibisce al centro ha rank PEGGIORE")
        else:
            print(f"  → NON significativo (p > 0.05): nessuna differenza centro/estremi")

        # Proporzione top 5
        top5_centro = (centro <= 5).mean()
        top5_estremi = (estremi <= 5).mean()
        print(f"\n  % Top 5:  Centro {top5_centro*100:.1f}% vs Estremi {top5_estremi*100:.1f}%")

        # Chi-square test on top 5 vs not-top-5
        # contingency table: [centro_top5, centro_notop5], [estremi_top5, estremi_notop5]
        ct = np.array([
            [(centro <= 5).sum(), (centro > 5).sum()],
            [(estremi <= 5).sum(), (estremi > 5).sum()]
        ])
        if ct.min() > 0:
            chi2, p_chi, _, _ = stats.chi2_contingency(ct)
            print(f"  Chi² (Top 5 centro vs estremi) = {chi2:.3f}, p = {p_chi:.4f}")

    # === D) Solo serata finale (la più rilevante) ===
    finale = usable[usable['serata'] == usable['serata'].max()].copy()
    if len(finale) >= 10:
        print(f"\n{'─'*60}")
        print("D) SOLO SERATA FINALE")
        print(f"{'─'*60}")

        finale['fascia'] = pd.cut(
            finale['posizione_relativa'],
            bins=[-0.01, 0.33, 0.66, 1.01],
            labels=['Inizio', 'Centro', 'Fine']
        )

        print(f"\n  {'Fascia':<10} {'n':>4}  {'Rank medio':>11}  {'Mediana':>8}  {'Top 5':>6}  {'Top 10':>7}")
        print(f"  {'─'*52}")
        for fascia in ['Inizio', 'Centro', 'Fine']:
            g = finale[finale['fascia'] == fascia]
            if len(g) == 0:
                continue
            avg = g['classifica_finale'].mean()
            med = g['classifica_finale'].median()
            t5 = (g['classifica_finale'] <= 5).mean() * 100
            t10 = (g['classifica_finale'] <= 10).mean() * 100
            print(f"  {fascia:<10} {len(g):>4}  {avg:>11.1f}  {med:>8.1f}  {t5:>5.0f}%  {t10:>6.0f}%")

        rho, p = stats.spearmanr(finale['posizione_relativa'], finale['classifica_finale'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"\n  Spearman rho = {rho:+.3f}  p = {p:.4f} {sig}")

    # === E) Riepilogo ===
    print(f"\n{'='*70}")
    print("CONCLUSIONE")
    print(f"{'='*70}")

    centro = usable[usable['fascia'] == 'Centro (2/3)']['classifica_finale']
    estremi = usable[usable['fascia'] != 'Centro (2/3)']['classifica_finale']
    centro_mean = centro.mean()
    estremi_mean = estremi.mean()
    diff = estremi_mean - centro_mean

    # Test centro vs estremi (il test più rilevante per l'ipotesi)
    _, p_mw = stats.mannwhitneyu(centro, estremi, alternative='two-sided') \
        if len(centro) >= 3 and len(estremi) >= 3 else (0, 1.0)

    # Correlazione lineare
    rho_all, p_rho = stats.spearmanr(usable['posizione_relativa'], usable['classifica_finale'])

    print(f"\n  Rank medio - Centro (2/3): {centro_mean:.1f}")
    print(f"  Rank medio - Estremi:      {estremi_mean:.1f}")
    print(f"  Differenza: {diff:+.1f} posizioni {'a favore del centro' if diff > 0 else 'a favore degli estremi'}")

    if p_mw < 0.05 and diff > 0:
        print(f"\n  RISULTATO: I dati SUPPORTANO l'ipotesi (Mann-Whitney p = {p_mw:.4f}).")
        print("  Chi si esibisce nelle posizioni centrali tende ad avere")
        print("  una classifica finale migliore rispetto a chi apre o chiude.")
    elif p_mw < 0.10 and diff > 0:
        print(f"\n  RISULTATO: Tendenza a favore del centro (p = {p_mw:.4f}),")
        print("  ma non statisticamente significativa al livello 0.05.")
        print("  Servono piu edizioni per confermare.")
    else:
        print(f"\n  RISULTATO: I dati NON supportano l'ipotesi (p = {p_mw:.4f}).")
        print("  L'ordine di uscita non ha un effetto significativo")
        print("  sulla classifica finale.")

    if abs(rho_all) > 0.1 and p_rho < 0.05:
        direction = "tardi" if rho_all < 0 else "presto"
        print(f"\n  Nota: c'e anche un effetto lineare (rho = {rho_all:+.3f}, p = {p_rho:.4f}):")
        print(f"  esibirsi piu {direction} correla con classifica migliore.")
    else:
        print(f"\n  Nota: nessun effetto lineare (rho = {rho_all:+.3f}, p = {p_rho:.4f}):")
        print("  non e 'prima = meglio' o 'dopo = meglio', ma specificamente")
        print("  le posizioni centrali sembrano avvantaggiate.")

    print(f"\n  Basato su {len(usable)} esibizioni, anni: {years}")
    if len(years) < 3:
        print("  Nota: servono piu edizioni per risultati robusti.")
        print("  Lancia senza --skip-scrape per scaricare dati da Wikipedia.")


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sanremo: ordine di uscita vs classifica')
    parser.add_argument('--years', type=int, nargs='+',
                        default=list(range(2020, 2027)),
                        help='Anni da analizzare (default: 2020-2026)')
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Salta lo scraping Wikipedia, usa solo dati locali')
    parser.add_argument('--force-scrape', action='store_true',
                        help='Forza ri-scraping anche se il CSV esiste')
    args = parser.parse_args()

    print('=' * 70)
    print("SANREMO — L'ORDINE DI USCITA INFLUENZA LA CLASSIFICA?")
    print('=' * 70)

    # Step 1: Load verified local data
    print('\n[1/4] Caricamento dati locali verificati...')
    verified_rows = load_verified_serate()
    rankings = load_rankings()
    print(f'  Esibizioni da JSON verificato: {len(verified_rows)}')
    print(f'  Anni con classifica: {sorted(rankings.keys())}')

    # Step 2: Scrape Wikipedia (if not skipped)
    wiki_rows = []
    if not args.skip_scrape:
        print(f'\n[2/4] Scraping Wikipedia per anni: {args.years}...')
        for year in args.years:
            wiki_rows.extend(scrape_year(year))
        print(f'  Totale da Wikipedia: {len(wiki_rows)} esibizioni')
    else:
        print('\n[2/4] Scraping Wikipedia saltato (--skip-scrape)')

    # Step 3: Merge and save
    print('\n[3/4] Unione dati e salvataggio CSV...')
    all_rows = merge_data(wiki_rows, verified_rows, rankings)

    if not all_rows:
        print('ERRORE: Nessun dato disponibile!')
        print('Lancia senza --skip-scrape per scaricare da Wikipedia,')
        print('oppure verifica che dati_sremo/sanremo_verified_data.json esista.')
        sys.exit(1)

    save_csv(all_rows, OUTPUT_CSV)

    # Count usable data
    with_rank = sum(1 for r in all_rows if r.get('classifica_finale') is not None)
    print(f'  Esibizioni con classifica finale: {with_rank}')

    # Step 4: Analysis
    print('\n[4/4] Analisi statistica...')
    df = pd.read_csv(OUTPUT_CSV)
    run_analysis(df)

    print(f"\n{'='*70}")
    print('FATTO')
    print(f"{'='*70}")
    print(f'CSV:  {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
