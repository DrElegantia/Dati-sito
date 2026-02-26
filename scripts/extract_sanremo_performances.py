#!/usr/bin/env python3
"""
Sanremo Festival — Performance Time vs Ranking Analysis

Reads verified performance data from dati_sremo/sanremo_verified_data.json,
enriches with tweet timestamps where available, and builds ML models
to test whether performance time/order correlates with final ranking.

Usage:
    python3 scripts/extract_sanremo_performances.py
"""

import json
import glob
import os
import re
import csv
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='R.*score is not well-defined')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dati_sremo')
TWEET_DIR = os.path.join(DATA_DIR, 'SanremoRai')
VERIFIED_JSON = os.path.join(DATA_DIR, 'sanremo_verified_data.json')


# ============================================================
# 1. Load verified data
# ============================================================

def load_verified_data():
    with open(VERIFIED_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 2. Parse tweet timestamps (supplement verified data)
# ============================================================

def parse_tweets():
    """Parse tweet JSONs for per-performance timestamps."""
    results = []
    files = sorted(glob.glob(os.path.join(TWEET_DIR, '*.json')))
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        content = d.get('content', '')
        artist_m = re.search(r'🎤\s*(.+?)(?:\n|$)', content)
        song_m = re.search(r'🎧\s*(.+?)(?:\n|$)', content)
        if not artist_m or not song_m:
            continue
        year = None
        for h in d.get('hashtags', []):
            m = re.search(r'[Ss]anremo(\d{4})', h)
            if m:
                year = int(m.group(1))
                break
        serata_m = re.search(r'[Ss]erata\s*(\d+)', content)
        serata = int(serata_m.group(1)) if serata_m else None
        if year and serata and d.get('date'):
            results.append({
                'year': year,
                'serata': serata,
                'artist': artist_m.group(1).strip(),
                'song': song_m.group(1).strip(),
                'tweet_datetime': d['date'],
            })
    return results


# ============================================================
# 3. Normalize artist names for matching
# ============================================================

def norm(name):
    s = name.lower().strip()
    s = re.sub(r'\s+feat\.?\s+', ' ', s)
    s = re.sub(r'\s*&\s*', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def artist_match(a, b):
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    overlap = wa & wb
    if overlap and len(overlap) / min(len(wa), len(wb)) >= 0.5:
        return True
    return False


# ============================================================
# 4. Build dataset
# ============================================================

def time_to_minutes(time_str, date_str):
    """Convert HH:MM to minutes-since-20:00 on the given date.
    Handles times after midnight (00:xx-05:xx) as continuation of the evening."""
    if not time_str:
        return None
    h, m = map(int, time_str.split(':'))
    # If before 06:00, it's "the next day" of the evening
    if h < 6:
        h += 24
    return (h - 20) * 60 + m


def build_dataset(verified_data, tweets):
    rows = []
    for year_str, ydata in verified_data.items():
        year = int(year_str)
        # Build ranking lookup
        rank_map = {}
        for r in ydata['rankings']:
            rank_map[norm(r['artist'])] = (r['rank'], r['artist'], r['song'])

        for serata_key, sdata in ydata['serate'].items():
            serata = int(serata_key)
            serata_date = sdata['date']
            n_performers = sdata['num_performers']

            for perf in sdata['performances']:
                artist = perf['artist']
                order = perf['order']
                time_str = perf.get('time')

                # Find ranking
                rank_info = None
                for rn, rv in rank_map.items():
                    if artist_match(artist, rv[1]):
                        rank_info = rv
                        break

                # Check tweets for additional timestamp
                tweet_time = None
                for tw in tweets:
                    if tw['year'] == year and tw['serata'] == serata:
                        if artist_match(artist, tw['artist']):
                            tweet_time = tw['tweet_datetime']
                            break

                minutes = time_to_minutes(time_str, serata_date)

                rows.append({
                    'year': year,
                    'serata': serata,
                    'serata_name': sdata['name'],
                    'serata_date': serata_date,
                    'artist': rank_info[1] if rank_info else artist,
                    'song': rank_info[2] if rank_info else '',
                    'rank': rank_info[0] if rank_info else None,
                    'performance_order': order,
                    'total_in_serata': n_performers,
                    'relative_position': round((order - 1) / max(n_performers - 1, 1), 4),
                    'time': time_str if time_str else '',
                    'minutes_since_2000': minutes,
                    'has_exact_time': time_str is not None,
                    'tweet_datetime': tweet_time if tweet_time else '',
                })

    return pd.DataFrame(rows)


# ============================================================
# 5. ML Analysis
# ============================================================

def run_analysis(df):
    """Run complete statistical and ML analysis."""

    print(f"\n{'='*70}")
    print("ANALISI: L'ORARIO DI ESIBIZIONE INFLUISCE SULLA CLASSIFICA?")
    print(f"{'='*70}")

    # --- Dataset overview ---
    total = len(df)
    with_time = df['has_exact_time'].sum()
    with_rank = df['rank'].notna().sum()
    usable = df[df['rank'].notna()].copy()

    print(f"\nDataset: {total} esibizioni totali")
    print(f"  Con orario esatto: {with_time}")
    print(f"  Con classifica finale: {with_rank}")

    for serata in sorted(usable['serata'].unique()):
        s_df = usable[usable['serata'] == serata]
        has_time = s_df['has_exact_time'].sum()
        print(f"  Serata {serata} ({s_df['serata_name'].iloc[0]}): "
              f"{len(s_df)} artisti, {has_time} con orario esatto")

    # --- A) Correlation: order vs rank (all serate with rankings) ---
    print(f"\n{'─'*50}")
    print("A) CORRELAZIONE: ORDINE DI ESIBIZIONE vs CLASSIFICA FINALE")
    print(f"{'─'*50}")

    # For each serata, compute correlation
    for serata in sorted(usable['serata'].unique()):
        s_df = usable[usable['serata'] == serata]
        if len(s_df) < 5:
            continue
        r_spearman, p_spearman = stats.spearmanr(s_df['performance_order'], s_df['rank'])
        r_pearson, p_pearson = stats.pearsonr(s_df['performance_order'], s_df['rank'])
        print(f"\n  Serata {serata} (n={len(s_df)}):")
        print(f"    Spearman rho = {r_spearman:+.4f}  (p = {p_spearman:.4f})")
        print(f"    Pearson  r   = {r_pearson:+.4f}  (p = {p_pearson:.4f})")
        if p_spearman < 0.05:
            direction = "peggiore" if r_spearman > 0 else "migliore"
            print(f"    → Significativo: esibirsi dopo → classifica {direction}")
        else:
            print(f"    → NON significativo (p > 0.05)")

    # All serate combined (use relative_position for comparability)
    print(f"\n  Tutte le serate combinate (n={len(usable)}):")
    r_sp, p_sp = stats.spearmanr(usable['relative_position'], usable['rank'])
    print(f"    Spearman rho(posizione_relativa, rank) = {r_sp:+.4f}  (p = {p_sp:.4f})")

    # --- B) Time analysis (only where we have exact times) ---
    timed = usable[usable['has_exact_time'] & usable['minutes_since_2000'].notna()].copy()
    if len(timed) >= 5:
        print(f"\n{'─'*50}")
        print("B) CORRELAZIONE: ORARIO ESATTO vs CLASSIFICA FINALE")
        print(f"{'─'*50}")
        print(f"  (n={len(timed)} esibizioni con orario esatto)")

        r_sp, p_sp = stats.spearmanr(timed['minutes_since_2000'], timed['rank'])
        print(f"  Spearman rho(minuti_da_20:00, rank) = {r_sp:+.4f}  (p = {p_sp:.4f})")

        for serata in sorted(timed['serata'].unique()):
            s_df = timed[timed['serata'] == serata]
            if len(s_df) < 5:
                continue
            r_s, p_s = stats.spearmanr(s_df['minutes_since_2000'], s_df['rank'])
            print(f"  Serata {serata} (n={len(s_df)}): rho = {r_s:+.4f}  (p = {p_s:.4f})")

    # --- C) Position group analysis ---
    print(f"\n{'─'*50}")
    print("C) ANALISI PER FASCE DI POSIZIONE")
    print(f"{'─'*50}")

    usable = usable.copy()
    usable['position_group'] = pd.cut(
        usable['relative_position'],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=['Inizio (1/3)', 'Centro (2/3)', 'Fine (3/3)']
    )

    print(f"\n  {'Fascia':<20} {'n':>4} {'Rank medio':>12} {'Top 5 %':>10} {'Top 10 %':>10}")
    print(f"  {'─'*60}")
    for grp in ['Inizio (1/3)', 'Centro (2/3)', 'Fine (3/3)']:
        g = usable[usable['position_group'] == grp]
        if len(g) == 0:
            continue
        avg_rank = g['rank'].mean()
        top5 = (g['rank'] <= 5).mean() * 100
        top10 = (g['rank'] <= 10).mean() * 100
        print(f"  {grp:<20} {len(g):>4} {avg_rank:>12.1f} {top5:>9.1f}% {top10:>9.1f}%")

    # --- D) ML Models ---
    print(f"\n{'─'*50}")
    print("D) MODELLI ML: PREDIRE LA CLASSIFICA DALL'ORDINE DI ESIBIZIONE")
    print(f"{'─'*50}")

    feature_cols = ['performance_order', 'relative_position', 'total_in_serata', 'serata']
    X = usable[feature_cols].copy()
    y = usable['rank'].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=42),
    }

    # Baseline: always predict mean rank
    mean_rank = y.mean()
    baseline_mae = np.abs(y - mean_rank).mean()
    print(f"\n  Baseline (predici sempre la media): MAE = {baseline_mae:.2f}")
    print(f"  Features: {feature_cols}")

    results = []
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    for name, model in models.items():
        mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
        mae = -mae_scores.mean()
        r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        r2 = r2_scores.mean()
        print(f"\n  {name}:")
        print(f"    CV MAE   = {mae:.2f}  (baseline: {baseline_mae:.2f})")
        print(f"    CV R²    = {r2:.4f}")
        improvement = (baseline_mae - mae) / baseline_mae * 100
        print(f"    Miglioramento su baseline: {improvement:+.1f}%")
        results.append({
            'model': name,
            'cv_mae': round(mae, 4),
            'cv_r2': round(r2, 4),
            'baseline_mae': round(baseline_mae, 4),
            'improvement_pct': round(improvement, 2),
        })

    # Feature importance from best tree model
    rf = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)
    rf.fit(X_scaled, y)
    print(f"\n  Feature Importance (RandomForest):")
    for feat, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    # --- E) Summary ---
    print(f"\n{'='*70}")
    print("CONCLUSIONE")
    print(f"{'='*70}")
    best = min(results, key=lambda x: x['cv_mae'])
    if best['improvement_pct'] > 5:
        print(f"\n  Il modello {best['model']} migliora del {best['improvement_pct']:.1f}%")
        print(f"  rispetto alla baseline → L'ordine ha un effetto DEBOLE ma misurabile.")
    elif best['improvement_pct'] > 0:
        print(f"\n  Miglioramento minimo ({best['improvement_pct']:.1f}%) rispetto alla baseline.")
        print(f"  → L'ordine di esibizione ha un effetto MOLTO DEBOLE sulla classifica.")
    else:
        print(f"\n  Nessun miglioramento rispetto alla baseline.")
        print(f"  → L'ordine di esibizione NON sembra influire sulla classifica.")
    print(f"\n  Nota: analisi basata su Sanremo 2025 ({len(usable)} esibizioni).")
    print(f"  Aggiungendo piu edizioni (2021-2026) il modello sara piu robusto.")

    return pd.DataFrame(results)


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 70)
    print("SANREMO — PIPELINE ESTRAZIONE E ANALISI ESIBIZIONI")
    print("=" * 70)

    # 1) Load verified data
    print("\n[1/5] Caricamento dati verificati...")
    vdata = load_verified_data()
    years = list(vdata.keys())
    print(f"  Anni disponibili: {years}")

    # 2) Parse tweets
    print("\n[2/5] Parsing tweet per timestamp aggiuntivi...")
    tweets = parse_tweets()
    print(f"  Tweet con dati esibizione: {len(tweets)}")

    # 3) Build dataset
    print("\n[3/5] Costruzione dataset...")
    df = build_dataset(vdata, tweets)
    print(f"  Righe totali: {len(df)}")
    print(f"  Con orario esatto: {df['has_exact_time'].sum()}")
    print(f"  Con classifica: {df['rank'].notna().sum()}")

    # 4) Export CSV
    print("\n[4/5] Export CSV...")
    merged_csv = os.path.join(DATA_DIR, 'sanremo_esibizioni.csv')
    df.to_csv(merged_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  → {merged_csv} ({len(df)} righe)")

    # Rankings summary
    rankings_rows = []
    for year_str, ydata in vdata.items():
        for r in ydata['rankings']:
            rankings_rows.append({
                'year': int(year_str),
                'rank': r['rank'],
                'artist': r['artist'],
                'song': r['song'],
            })
    rankings_csv = os.path.join(DATA_DIR, 'sanremo_classifiche.csv')
    pd.DataFrame(rankings_rows).to_csv(rankings_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  → {rankings_csv}")

    # Pipeline summary
    summary_rows = []
    for year_str, ydata in vdata.items():
        year = int(year_str)
        ydf = df[df['year'] == year]
        timed = ydf[ydf['has_exact_time']].shape[0]
        serate = sorted(ydf['serata'].unique())
        summary_rows.append({
            'year': year,
            'winner': ydata['winner'],
            'num_artists': ydata['num_artists'],
            'serate_available': ', '.join(map(str, serate)),
            'performances_total': len(ydf),
            'with_exact_time': timed,
            'with_ranking': ydf['rank'].notna().sum(),
            'data_quality': 'buona' if timed > 20 else 'parziale' if timed > 0 else 'solo_ordine',
        })
    summary_csv = os.path.join(DATA_DIR, 'sanremo_pipeline_summary.csv')
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  → {summary_csv}")
    print(f"\n  Riepilogo:")
    print(summary_df.to_string(index=False))

    # 5) ML Analysis
    print("\n[5/5] Analisi ML...")
    ml_results = run_analysis(df)
    ml_csv = os.path.join(DATA_DIR, 'sanremo_ml_results.csv')
    ml_results.to_csv(ml_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n  → {ml_csv}")

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETATA")
    print(f"{'='*70}")
    print(f"\nFile generati in {DATA_DIR}:")
    print(f"  sanremo_verified_data.json    — Dati verificati (input)")
    print(f"  sanremo_esibizioni.csv        — Dataset completo esibizioni")
    print(f"  sanremo_classifiche.csv       — Classifiche finali")
    print(f"  sanremo_ml_results.csv        — Risultati modelli ML")
    print(f"  sanremo_pipeline_summary.csv  — Riepilogo pipeline")


if __name__ == '__main__':
    main()
