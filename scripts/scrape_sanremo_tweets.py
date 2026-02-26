#!/usr/bin/env python3
"""
Scarica tutti i tweet di @SanremoRai durante il Festival via gallery-dl,
poi filtra quelli che menzionano artisti in gara.

Prerequisiti:
  pip install gallery-dl requests pandas lxml
  Essere loggati su X/Twitter nel browser (Chrome di default)

Uso:
  python3 scripts/scrape_sanremo_tweets.py [--year 2025] [--browser chrome]
  python3 scripts/scrape_sanremo_tweets.py --all  # 2015-2026

  Opzioni per resilienza al rate-limiting:
  --delay 30        Pausa in secondi tra una richiesta e l'altra (default: 30)
  --max-retries 3   Tentativi per ogni giorno prima di saltare (default: 3)
  --resume          Salta i giorni che hanno già dati in raw/
  --cookies FILE    Usa un file cookies.txt (formato Netscape) invece del browser
  --merge-repo      Unisci i tweet scaricati con quelli già nel repo (dati_sremo/SanremoRai/)

Output in dati_sremo/scrape_output/<ANNO>/:
  - raw/*.json         (metadati gallery-dl)
  - tweets_<ANNO>.csv  (tutti i tweet del periodo)
  - tweets_<ANNO>_match.csv  (solo tweet con menzione artisti)
  - artisti_<ANNO>.txt (lista artisti da Wikipedia)
  - meta.json          (metadati festival: date, hashtag, ecc.)
"""

import os
import re
import sys
import json
import csv
import glob
import signal
import argparse
import subprocess
import urllib.parse
import time
import random
from datetime import date, timedelta
from shutil import which

import requests
import pandas as pd

# ============================================================
# Graceful shutdown on Ctrl+C
# ============================================================

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        print('\n  Secondo Ctrl+C, uscita immediata.')
        sys.exit(1)
    _shutdown_requested = True
    print('\n  Ctrl+C ricevuto. Finisco il giorno corrente e mi fermo...')


signal.signal(signal.SIGINT, _signal_handler)

# ============================================================
# Config
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(BASE_DIR, 'dati_sremo', 'scrape_output')
ACCOUNT = 'SanremoRai'
UA = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

MONTHS_IT = {
    'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4,
    'maggio': 5, 'giugno': 6, 'luglio': 7, 'agosto': 8,
    'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12,
}

# Dati hardcoded come fallback quando Wikipedia non è raggiungibile
FALLBACK_FESTIVAL_DATA = {
    2015: {'dates': ('2015-02-10', '2015-02-14'), 'artists': [
        'Il Volo', 'Nek', 'Malika Ayane', 'Dear Jack', 'Chiara',
        'Nesli', 'Annalisa', 'Lorenzo Fragola', 'Nina Zilli',
        'Gianluca Grignani', 'Marco Masini', 'Moreno', 'Alex Britti',
        'Grazia Di Michele e Mauro Coruzzi', 'Bianca Atzei', 'Lara Fabian',
        'Raf', 'Stadio', 'Nomadi', 'Federica Carta',
    ]},
    2016: {'dates': ('2016-02-09', '2016-02-13'), 'artists': [
        'Stadio', 'Francesca Michielin', 'Deborah Iurato e Giovanni Caccamo',
        'Enrico Ruggeri', 'Arisa', 'Noemi', 'Alessio Bernabei', 'Annalisa',
        'Lorenzo Fragola', 'Rocco Hunt', 'Dear Jack', 'Elodie', 'Bluvertigo',
        'Dolcenera', 'Valerio Scanu', 'Neffa', 'Patty Pravo',
        'Irene Fornaciari', 'Clementino', 'Zero Assoluto',
    ]},
    2017: {'dates': ('2017-02-07', '2017-02-11'), 'artists': [
        'Francesco Gabbani', 'Fiorella Mannoia', 'Ermal Meta',
        'Michele Bravi', 'Fabrizio Moro', 'Elodie', 'Paola Turci',
        'Sergio Sylvestre', 'Ron', 'Gigi D\'Alessio', 'Al Bano',
        'Bianca Atzei', 'Clementino', 'Samuel', 'Giusy Ferreri',
        'Lodovica Comello', 'Raige e Giulia Luzi', 'Nesli',
        'Michele Zarrillo', 'Chiara', 'Marco Masini', 'Lele',
    ]},
    2018: {'dates': ('2018-02-06', '2018-02-10'), 'artists': [
        'Ermal Meta e Fabrizio Moro', 'Lo Stato Sociale', 'Annalisa',
        'Ultimo', 'Max Gazzè', 'Ron', 'Nina Zilli', 'Noemi',
        'Le Vibrazioni', 'Diodato e Roy Paci', 'Ornella Vanoni',
        'Gio Evan', 'Luca Barbarossa', 'Decibel', 'Roby Facchinetti e Riccardo Fogli',
        'Red Canzian', 'Renzo Rubino', 'Mario Biondi',
    ]},
    2019: {'dates': ('2019-02-05', '2019-02-09'), 'artists': [
        'Mahmood', 'Ultimo', 'Il Volo', 'Irama', 'Loredana Bertè',
        'Simone Cristicchi', 'Daniele Silvestri', 'Arisa', 'Achille Lauro',
        'Boomdabash', 'Federica Carta e Shade', 'Ex-Otago', 'Ghemon',
        'Zen Circus', 'Enrico Nigiotti', 'Paola Turci', 'Francesco Renga',
        'Anna Tatangelo', 'Nek', 'Negrita', 'Motta', 'Einar',
        'Nino D\'Angelo e Livio Cori', 'Patty Pravo e Briga',
    ]},
    2020: {'dates': ('2020-02-04', '2020-02-08'), 'artists': [
        'Diodato', 'Francesco Gabbani', 'Pinguini Tattici Nucleari',
        'Le Vibrazioni', 'Piero Pelù', 'Tosca', 'Elodie', 'Achille Lauro',
        'Irene Grandi', 'Rancore', 'Raphael Gualazzi', 'Levante',
        'Anastasio', 'Alberto Urso', 'Marco Masini', 'Paolo Jannacci',
        'Rita Pavone', 'Michele Zarrillo', 'Enrico Nigiotti', 'Giordana Angi',
        'Elettra Lamborghini', 'Junior Cally', 'Riki', 'Bugo e Morgan',
    ]},
    2021: {'dates': ('2021-03-02', '2021-03-06'), 'artists': [
        'Maneskin', 'Fedez e Francesca Michielin', 'Ermal Meta',
        'Colapesce Dimartino', 'Irama', 'Annalisa', 'Willie Peyote',
        'Madame', 'Arisa', 'Lo Stato Sociale', 'Malika Ayane', 'Noemi',
        'Coma_Cose', 'La Rappresentante di Lista', 'Extraliscio feat. Davide Toffolo',
        'Gaia', 'Fasma', 'Fulminacci', 'Orietta Berti', 'Max Gazzè',
        'Ghemon', 'Gio Evan', 'Francesco Renga', 'Bugo', 'Random', 'Aiello',
    ]},
    2022: {'dates': ('2022-02-01', '2022-02-05'), 'artists': [
        'Mahmood e Blanco', 'Elisa', 'Gianni Morandi', 'Irama',
        'Sangiovanni', 'Emma', 'La Rappresentante di Lista', 'Massimo Ranieri',
        'Dargen D\'Amico', 'Michele Bravi', 'Rkomi', 'Fabrizio Moro',
        'Achille Lauro', 'Aka 7even', 'Noemi', 'Ditonellapiaga e Rettore',
        'Matteo Romano', 'Highsnob e Hu', 'Le Vibrazioni', 'Giusy Ferreri',
        'Iva Zanicchi', 'Giovanni Truppi', 'Yuman', 'Ana Mena', 'Tananai',
    ]},
    2023: {'dates': ('2023-02-07', '2023-02-11'), 'artists': [
        'Marco Mengoni', 'Lazza', 'Mr.Rain', 'Ultimo', 'Tananai',
        'Madame', 'Giorgia', 'Colapesce Dimartino', 'Elodie', 'Articolo 31',
        'Levante', 'Modà', 'Gianluca Grignani', 'Rosa Chemical',
        'Paola e Chiara', 'Ariete', 'LDA', 'Leo Gassman', 'Colla Zio',
        'gIANMARIA', 'Mara Sattei', 'Olly', 'Cugini di Campagna',
        'Anna Oxa', 'Shari', 'Will', 'Sethu',
    ]},
    2024: {'dates': ('2024-02-06', '2024-02-10'), 'artists': [
        'Angelina Mango', 'Geolier', 'Annalisa', 'Ghali', 'Irama',
        'Negramaro', 'The Kolors', 'Mahmood', 'Il Volo', 'Emma',
        'Renga Nek', 'Loredana Bertè', 'Alessandra Amoroso',
        'Ricchi e Poveri', 'Fiorella Mannoia', 'Dargen D\'Amico', 'Mr.Rain',
        'Clara', 'La Sad', 'Rose Villain', 'Sangiovanni', 'Diodato',
        'Santi Francesi', 'BigMama', 'Fred De Palma', 'Alfa', 'Bnkr44',
        'Petit', 'gIANMARIA', 'Il Tre',
    ]},
    2025: {'dates': ('2025-02-11', '2025-02-15'), 'artists': [
        'Olly', 'Lucio Corsi', 'Brunori Sas', 'Fedez', 'Simone Cristicchi',
        'Giorgia', 'Achille Lauro', 'Francesco Gabbani', 'Irama', 'Coma_Cose',
        'Bresh', 'Elodie', 'Noemi', 'The Kolors', 'Rocco Hunt',
        'Willie Peyote', 'Sarah Toscano', 'Shablo feat. Gue, Joshua e Tormento',
        'Rose Villain', 'Joan Thiele', 'Francesca Michielin', 'Modà',
        'Massimo Ranieri', 'Serena Brancale', 'Tony Effe', 'Gaia', 'Clara',
        'Rkomi', 'Marcella Bella',
    ]},
    2026: {'dates': ('2026-02-24', '2026-02-28'), 'artists': [
        'Patty Pravo', 'LDA', 'Aka 7even', 'Enrico Nigiotti',
        'Tommaso Paradiso', 'Elettra Lamborghini', 'Ermal Meta', 'Levante',
        'Bambole di pezza', 'Chiello', 'J-Ax', 'Nayt', 'Fulminacci',
        'Fedez', 'Marco Masini', 'Dargen D\'Amico', 'Ditonellapiaga',
    ]},
}


# ============================================================
# Wikipedia: date festival + lista artisti
# ============================================================

def norm_text(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def parse_periodo(year, periodo):
    """Parse 'X-Y mese' or 'X mese - Y mese' into (date, date)."""
    p = norm_text(periodo).lower().replace('º', '').replace('°', '')
    p = p.replace('–', '-').replace('—', '-')
    # "X-Y mese"
    m = re.search(r'(\d{1,2})\s*-\s*(\d{1,2})\s+([a-zà]+)$', p)
    if m:
        d1, d2, mon = int(m.group(1)), int(m.group(2)), m.group(3)
        return date(year, MONTHS_IT[mon], d1), date(year, MONTHS_IT[mon], d2)
    # "X mese - Y mese"
    m = re.search(r'(\d{1,2})\s+([a-zà]+)\s*-\s*(\d{1,2})\s+([a-zà]+)$', p)
    if m:
        d1, mon1, d2, mon2 = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        return date(year, MONTHS_IT[mon1], d1), date(year, MONTHS_IT[mon2], d2)
    raise ValueError(f'Formato periodo non gestito: {periodo}')


def clean_artist_name(s):
    s = norm_text(s)
    s = re.sub(r'\s*\(.*?\)\s*', '', s).strip()
    s = re.sub(r'\[.*?\]', '', s).strip()
    return s


def get_wikipedia_data(year):
    """Fetch festival dates and artist list from Wikipedia."""
    url = f'https://it.wikipedia.org/wiki/Festival_di_Sanremo_{year}'
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    html = r.text

    # Strategy 1: find the infobox by content ("Periodo")
    # This avoids lxml XPath exact-match issues with multi-class attributes
    # (e.g. class="sinottico tematica" != "sinottico")
    infoboxes = []
    try:
        infoboxes = pd.read_html(html, match='Periodo')
    except ValueError:
        pass

    # Strategy 2: try class-based selectors
    if not infoboxes:
        for cls in ('sinottico', 'infobox', 'infobox vevent'):
            try:
                infoboxes = pd.read_html(html, attrs={'class': cls})
                if infoboxes:
                    break
            except ValueError:
                continue

    # Strategy 3: scan all tables
    if not infoboxes:
        try:
            infoboxes = pd.read_html(html)
        except ValueError:
            raise RuntimeError(f'Nessuna tabella trovata su Wikipedia per {year}')

    periodo = None
    for df in infoboxes[:10]:
        if df.shape[1] >= 2:
            for _, row in df.iterrows():
                k = norm_text(row.iloc[0]).lower()
                if 'periodo' in k:
                    periodo = norm_text(row.iloc[1])
                    break
        if periodo:
            break
    if not periodo:
        raise RuntimeError(f'Campo "Periodo" non trovato su Wikipedia per {year}')

    start, end = parse_periodo(year, periodo)

    # Parse artists from tables with "Interprete" column
    try:
        tables = pd.read_html(html)
    except ValueError:
        tables = []
    artists = set()
    for df in tables:
        # Flatten MultiIndex columns (merged header rows on Wikipedia)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(str(c) for c in col).strip() for col in df.columns]
        cols = [str(c).strip().lower() for c in df.columns]
        for col_name in ('interprete', 'artista', 'cantante'):
            if col_name in cols:
                col = df.columns[cols.index(col_name)]
                for x in df[col].dropna().tolist():
                    name = clean_artist_name(x)
                    if name and len(name) < 80 and not name[0].isdigit():
                        artists.add(name)

    return start, end, sorted(artists)


def get_fallback_data(year):
    """Return hardcoded festival data as fallback."""
    if year not in FALLBACK_FESTIVAL_DATA:
        return None, None, []
    data = FALLBACK_FESTIVAL_DATA[year]
    start_str, end_str = data['dates']
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    return start, end, list(data.get('artists', []))


# ============================================================
# gallery-dl download
# ============================================================

def find_gallery_dl():
    """Find gallery-dl executable."""
    # Try PATH first
    gdl = which('gallery-dl')
    if gdl:
        return gdl
    # Common locations
    for path in [
        os.path.expanduser('~/Library/Python/3.9/bin/gallery-dl'),
        os.path.expanduser('~/Library/Python/3.11/bin/gallery-dl'),
        os.path.expanduser('~/.local/bin/gallery-dl'),
        '/usr/local/bin/gallery-dl',
    ]:
        if os.path.isfile(path):
            return path
    return None


def daterange(d1, d2):
    d = d1
    while d <= d2:
        yield d
        d += timedelta(days=1)


def _count_jsons_in(directory):
    """Count JSON files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    return len(glob.glob(os.path.join(directory, '**', '*.json'), recursive=True))


def _day_has_data(rawdir, day):
    """Check if we already have data for a given day (for --resume)."""
    day_str = day.isoformat()
    for jf in glob.glob(os.path.join(rawdir, '**', '*.json'), recursive=True):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            dt = str(d.get('date', ''))
            if dt.startswith(day_str):
                return True
        except Exception:
            continue
    return False


def scrape_day(gdl, browser, account, hashtag, day, outdir,
               max_retries=3, delay=30, cookies_file=None, resume=False):
    """Scrape tweets for one day using gallery-dl, with retry and backoff."""
    if resume and _day_has_data(outdir, day):
        print(f'    {day}: [SKIP] dati già presenti (--resume)')
        return True

    since = day.isoformat()
    until = (day + timedelta(days=1)).isoformat()
    query = f'from:{account} since:{since} until:{until} {hashtag}'
    url = 'https://x.com/search?q=' + urllib.parse.quote_plus(query) + '&f=live'

    cmd = [gdl]
    if cookies_file:
        cmd += ['--cookies', cookies_file]
    else:
        cmd += ['--cookies-from-browser', browser]
    cmd += [
        '--write-metadata',
        '--no-download',
        '--directory', outdir,
        url,
    ]

    before_count = _count_jsons_in(outdir)

    for attempt in range(1, max_retries + 1):
        if _shutdown_requested:
            return False

        label = f'(tentativo {attempt}/{max_retries})' if max_retries > 1 else ''
        print(f'    {day}: {query} {label}')

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        after_count = _count_jsons_in(outdir)
        new_files = after_count - before_count

        # Check for rate limit / error signals in stderr
        stderr = result.stderr or ''
        is_rate_limited = any(kw in stderr.lower() for kw in [
            'rate limit', '429', 'too many requests', 'forbidden',
        ])

        if is_rate_limited and attempt < max_retries:
            backoff = delay * (2 ** (attempt - 1)) + random.uniform(0, 5)
            print(f'           Rate limit rilevato. Attendo {backoff:.0f}s...')
            time.sleep(backoff)
            continue

        if result.returncode != 0 and new_files == 0 and attempt < max_retries:
            backoff = delay * attempt + random.uniform(0, 5)
            print(f'           Errore (rc={result.returncode}). Riprovo tra {backoff:.0f}s...')
            time.sleep(backoff)
            continue

        if new_files > 0:
            print(f'           +{new_files} file JSON')
        elif result.returncode == 0:
            print(f'           Nessun tweet trovato per questo giorno')
        break

    # Delay between days to avoid rate limiting
    if delay > 0 and not _shutdown_requested:
        jitter = random.uniform(0, delay * 0.3)
        pause = delay + jitter
        print(f'           Pausa {pause:.0f}s prima del prossimo giorno...')
        time.sleep(pause)

    return True


# ============================================================
# Consolidation
# ============================================================

def consolidate_jsons(rawdir):
    """Read all gallery-dl JSON metadata and deduplicate by tweet_id."""
    rows = []
    seen = set()
    for jf in glob.glob(os.path.join(rawdir, '**', '*.json'), recursive=True):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        tid = d.get('tweet_id')
        dt = d.get('date')
        tx = d.get('content')
        if not tid or not dt or not tx:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        rows.append({
            'date': str(dt),
            'tweet_id': str(tid),
            'text': tx.replace('\n', ' ').replace('\r', ' ').strip(),
        })
    rows.sort(key=lambda x: x['date'])
    return rows


def filter_by_artists(rows, artists):
    """Filter tweets that mention at least one artist."""
    if not artists:
        return []
    pattern = re.compile(
        '|'.join(re.escape(a) for a in sorted(artists, key=len, reverse=True)),
        re.IGNORECASE,
    )
    return [r for r in rows if pattern.search(r['text'])]


def write_csv(rows, path, fieldnames=None):
    if not fieldnames:
        fieldnames = ['date', 'tweet_id', 'text']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Also read existing repo JSONs (fallback if no gallery-dl)
# ============================================================

def load_existing_tweets(year, start, end):
    """Load tweets from existing repo data (dati_sremo/SanremoRai/*.json)."""
    tweet_dir = os.path.join(BASE_DIR, 'dati_sremo', 'SanremoRai')
    rows = []
    seen = set()
    hashtag = f'Sanremo{year}'
    for fp in glob.glob(os.path.join(tweet_dir, '*.json')):
        try:
            with open(fp, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        tid = d.get('tweet_id')
        if tid in seen:
            continue
        # Check hashtag match
        tags = d.get('hashtags', [])
        if not any(hashtag.lower() in t.lower() for t in tags):
            continue
        dt_str = d.get('date', '')
        tx = d.get('content', '')
        if not dt_str or not tx:
            continue
        seen.add(tid)
        rows.append({
            'date': dt_str,
            'tweet_id': str(tid),
            'text': tx.replace('\n', ' ').replace('\r', ' ').strip(),
        })
    rows.sort(key=lambda x: x['date'])
    return rows


# ============================================================
# Main
# ============================================================

def process_year(year, browser='chrome', gdl_path=None,
                 delay=30, max_retries=3, cookies_file=None,
                 resume=False, merge_repo=False):
    print(f'\n{"="*60}')
    print(f'SANREMO {year}')
    print(f'{"="*60}')

    yeardir = os.path.join(OUTPUT_ROOT, str(year))
    os.makedirs(yeardir, exist_ok=True)

    # 1) Festival data: Wikipedia → fallback hardcoded
    print(f'  [1] Fetching dati festival...')
    start = end = None
    artists = []
    try:
        start, end, artists = get_wikipedia_data(year)
        print(f'      [Wikipedia] Date: {start} → {end}')
        print(f'      [Wikipedia] Artisti trovati: {len(artists)}')
    except Exception as e:
        print(f'      ⚠ Wikipedia non disponibile: {e}')
        start, end, artists = get_fallback_data(year)
        if start:
            print(f'      [Fallback] Date: {start} → {end}')
            print(f'      [Fallback] Artisti: {len(artists)}')
        else:
            print(f'      ✗ Nessun dato disponibile per {year} (né Wikipedia né fallback)')

    # Save metadata
    meta = {
        'year': year,
        'start': start.isoformat() if start else None,
        'end': end.isoformat() if end else None,
        'hashtag': f'#Sanremo{year}',
        'artists_count': len(artists),
        'artists': artists,
    }
    with open(os.path.join(yeardir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Save artist list
    with open(os.path.join(yeardir, f'artisti_{year}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(artists) + '\n')

    # 2) Scrape tweets
    rawdir = os.path.join(yeardir, 'raw')
    rows = []

    if gdl_path and start and end:
        print(f'  [2] Scraping tweet con gallery-dl...')
        if resume:
            print(f'      Modalità resume: salto giorni già scaricati')
        hashtag = f'#Sanremo{year}'
        for day in daterange(start, end):
            if _shutdown_requested:
                print(f'      Interruzione richiesta. Salvo i dati raccolti fin qui.')
                break
            scrape_day(gdl_path, browser, ACCOUNT, hashtag, day, rawdir,
                       max_retries=max_retries, delay=delay,
                       cookies_file=cookies_file, resume=resume)
        rows = consolidate_jsons(rawdir)
        print(f'      Tweet scaricati: {len(rows)}')
    else:
        print(f'  [2] gallery-dl non disponibile, uso dati repo...')
        rows = load_existing_tweets(year, start, end)
        print(f'      Tweet dal repo: {len(rows)}')

    # 2b) Merge with existing repo data if requested
    if merge_repo:
        print(f'  [2b] Merge con dati esistenti nel repo...')
        repo_rows = load_existing_tweets(year, start, end)
        seen_ids = {r['tweet_id'] for r in rows}
        added = 0
        for r in repo_rows:
            if r['tweet_id'] not in seen_ids:
                rows.append(r)
                seen_ids.add(r['tweet_id'])
                added += 1
        rows.sort(key=lambda x: x['date'])
        if added > 0:
            print(f'       +{added} tweet dal repo (totale: {len(rows)})')
        else:
            print(f'       Nessun tweet aggiuntivo dal repo')

    # 3) Export CSV
    all_csv = os.path.join(yeardir, f'tweets_{year}.csv')
    write_csv(rows, all_csv)
    print(f'  [3] → {all_csv} ({len(rows)} tweet)')

    match_rows = filter_by_artists(rows, artists)
    match_csv = os.path.join(yeardir, f'tweets_{year}_match.csv')
    write_csv(match_rows, match_csv)
    print(f'      → {match_csv} ({len(match_rows)} match artisti)')

    return {
        'year': year,
        'start': str(start) if start else '',
        'end': str(end) if end else '',
        'artists': len(artists),
        'tweets_total': len(rows),
        'tweets_match': len(match_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Scrape Sanremo tweet da @SanremoRai',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Test veloce su 2025
  python3 scripts/scrape_sanremo_tweets.py --year 2025

  # Tutti gli anni, con pausa lunga e resume
  python3 scripts/scrape_sanremo_tweets.py --all --delay 45 --resume

  # Usa file cookies invece del browser
  python3 scripts/scrape_sanremo_tweets.py --year 2024 --cookies cookies.txt

  # Unisci scraping con dati già nel repo
  python3 scripts/scrape_sanremo_tweets.py --all --merge-repo
""",
    )
    parser.add_argument('--year', type=int, help='Anno specifico (es. 2025)')
    parser.add_argument('--all', action='store_true', help='Tutti gli anni 2015-2026')
    parser.add_argument('--browser', default='chrome',
                        help='Browser per cookies (default: chrome)')
    parser.add_argument('--delay', type=int, default=30,
                        help='Pausa in secondi tra le richieste (default: 30)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Tentativi per ogni giorno (default: 3)')
    parser.add_argument('--cookies', type=str, default=None,
                        help='File cookies.txt (formato Netscape) al posto del browser')
    parser.add_argument('--resume', action='store_true',
                        help='Salta giorni che hanno già dati scaricati')
    parser.add_argument('--merge-repo', action='store_true',
                        help='Unisci tweet scaricati con quelli in dati_sremo/SanremoRai/')
    args = parser.parse_args()

    gdl = find_gallery_dl()
    if gdl:
        print(f'gallery-dl trovato: {gdl}')
    else:
        print('gallery-dl NON trovato. Uso solo dati esistenti nel repo.')
        print('Per installarlo: pip install gallery-dl')

    if args.cookies and not os.path.isfile(args.cookies):
        print(f'ERRORE: file cookies non trovato: {args.cookies}')
        sys.exit(1)

    if args.all:
        years = list(range(2015, 2027))
    elif args.year:
        years = [args.year]
    else:
        years = [2025]  # default: test con 2025
        print('Nessun anno specificato, uso 2025 come test.')

    print(f'Configurazione: delay={args.delay}s, max_retries={args.max_retries}, '
          f'resume={args.resume}, merge_repo={args.merge_repo}')

    results = []
    for y in years:
        if _shutdown_requested:
            print(f'\nInterruzione richiesta, salto gli anni rimanenti.')
            break
        try:
            r = process_year(
                y,
                browser=args.browser,
                gdl_path=gdl,
                delay=args.delay,
                max_retries=args.max_retries,
                cookies_file=args.cookies,
                resume=args.resume,
                merge_repo=args.merge_repo,
            )
            results.append(r)
        except Exception as e:
            print(f'  ERRORE per {y}: {e}')

    # Summary
    print(f'\n{"="*60}')
    print('RIEPILOGO')
    print(f'{"="*60}')
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        summary_csv = os.path.join(OUTPUT_ROOT, 'scrape_summary.csv')
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f'\n→ {summary_csv}')
    else:
        print('Nessun risultato.')


if __name__ == '__main__':
    main()
