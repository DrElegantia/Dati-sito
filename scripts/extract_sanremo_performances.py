#!/usr/bin/env python3
"""
Extract Sanremo performance data from tweets and Wikipedia,
build a dataset and ML model to analyze if performance order/time affects ranking.

Data sources:
- Tweet JSONs from dati_sremo/SanremoRai/ (performance timestamps for 2026)
- Wikipedia & web sources (performance orders and final rankings for 2021-2026)
"""

import json
import glob
import os
import re
import csv
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

TWEET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dati_sremo', 'SanremoRai')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dati_sremo')


# ============================================================
# PART 1: Extract performance timestamps from tweets
# ============================================================

def parse_tweets():
    """Parse tweet JSON files and extract structured performance data."""
    performances = []
    files = sorted(glob.glob(os.path.join(TWEET_DIR, '*.json')))
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = data.get('content', '')
        hashtags = data.get('hashtags', [])
        date_str = data.get('date', '')

        artist_match = re.search(r'🎤\s*(.+?)(?:\n|$)', content)
        song_match = re.search(r'🎧\s*(.+?)(?:\n|$)', content)
        if not artist_match or not song_match:
            continue

        artist = artist_match.group(1).strip()
        song = song_match.group(1).strip()

        year = None
        for h in hashtags:
            m = re.search(r'[Ss]anremo(\d{4})', h)
            if m:
                year = int(m.group(1))
                break
        if not year:
            m = re.search(r'#[Ss]anremo(\d{4})', content)
            if m:
                year = int(m.group(1))

        serata_match = re.search(r'[Ss]erata\s*(\d+)', content)
        serata = int(serata_match.group(1)) if serata_match else None

        if year and serata and date_str:
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            performances.append({
                'year': year,
                'serata': serata,
                'artist': artist,
                'song': song,
                'tweet_datetime': date_str,
                'tweet_time': dt.strftime('%H:%M:%S'),
                'tweet_hour': dt.hour,
                'tweet_minute': dt.minute,
                'tweet_id': data.get('tweet_id', ''),
            })
    return performances


# ============================================================
# PART 2: Wikipedia / Web data — rankings + performance orders
# ============================================================

# Final rankings from Wikipedia (complete for 2021-2025)
RANKINGS = {
    2021: [
        (1, "Maneskin", "Zitti e buoni"),
        (2, "Francesca Michielin e Fedez", "Chiamami per nome"),
        (3, "Ermal Meta", "Un milione di cose da dirti"),
        (4, "Colapesce e Dimartino", "Musica leggerissima"),
        (5, "Irama", "La genesi del tuo colore"),
        (6, "Willie Peyote", "Mai dire mai (La locura)"),
        (7, "Annalisa", "Dieci"),
        (8, "Madame", "Voce"),
        (9, "Orietta Berti", "Quando ti sei innamorato"),
        (10, "Arisa", "Potevi fare di piu"),
        (11, "La Rappresentante di Lista", "Amare"),
        (12, "Extraliscio feat. Davide Toffolo", "Bianca luce nera"),
        (13, "Lo Stato Sociale", "Combat Pop"),
        (14, "Noemi", "Glicine"),
        (15, "Malika Ayane", "Ti piaci cosi"),
        (16, "Fulminacci", "Santa Marinella"),
        (17, "Max Gazze", "Il farmacista"),
        (18, "Fasma", "Parlami"),
        (19, "Gaia", "Cuore amaro"),
        (20, "Coma_Cose", "Fiamme negli occhi"),
        (21, "Ghemon", "Momento perfetto"),
        (22, "Francesco Renga", "Quando trovo te"),
        (23, "Gio Evan", "Arnica"),
        (24, "Bugo", "E invece si"),
        (25, "Aiello", "Ora"),
        (26, "Random", "Torno a te"),
    ],
    2022: [
        (1, "Mahmood e Blanco", "Brividi"),
        (2, "Elisa", "O forse sei tu"),
        (3, "Gianni Morandi", "Apri tutte le porte"),
        (4, "Irama", "Ovunque sarai"),
        (5, "Sangiovanni", "Farfalle"),
        (6, "Emma", "Ogni volta e cosi"),
        (7, "La Rappresentante di Lista", "Ciao ciao"),
        (8, "Massimo Ranieri", "Lettera di la dal mare"),
        (9, "Dargen D'Amico", "Dove si balla"),
        (10, "Michele Bravi", "Inverno dei fiori"),
        (11, "Matteo Romano", "Virale"),
        (12, "Fabrizio Moro", "Sei tu"),
        (13, "Aka 7even", "Perfetta cosi"),
        (14, "Achille Lauro", "Domenica"),
        (15, "Noemi", "Ti amo non lo so dire"),
        (16, "Ditonellapiaga e Rettore", "Chimica"),
        (17, "Rkomi", "Insuperabile"),
        (18, "Iva Zanicchi", "Voglio amarti"),
        (19, "Giovanni Truppi", "Tuo padre, mia madre, Lucia"),
        (20, "Highsnob e Hu", "Abbi cura di te"),
        (21, "Yuman", "Ora e qui"),
        (22, "Le Vibrazioni", "Tantissimo"),
        (23, "Giusy Ferreri", "Miele"),
        (24, "Ana Mena", "Duecentomila ore"),
        (25, "Tananai", "Sesso occasionale"),
    ],
    2023: [
        (1, "Marco Mengoni", "Due vite"),
        (2, "Lazza", "Cenere"),
        (3, "Mr. Rain", "Supereroi"),
        (4, "Ultimo", "Alba"),
        (5, "Tananai", "Tango"),
        (6, "Giorgia", "Parole dette male"),
        (7, "Madame", "Il bene nel male"),
        (8, "Rosa Chemical", "Made in Italy"),
        (9, "Elodie", "Due"),
        (10, "Colapesce Dimartino", "Splash"),
        (11, "Moda", "Lasciami"),
        (12, "Gianluca Grignani", "Quando ti manca il fiato"),
        (13, "Coma_Cose", "L'addio"),
        (14, "Ariete", "Mare di guai"),
        (15, "LDA", "Se poi domani"),
        (16, "Articolo 31", "Un bel viaggio"),
        (17, "Paola e Chiara", "Furore"),
        (18, "Leo Gassmann", "Terzo cuore"),
        (19, "Mara Sattei", "Duemilaminuti"),
        (20, "Colla Zio", "Non mi va"),
        (21, "I Cugini di Campagna", "Lettera 22"),
        (22, "gIANMARIA", "Mostro"),
        (23, "Levante", "Vivo"),
        (24, "Olly", "Polvere"),
        (25, "Anna Oxa", "Sali (canto dell'anima)"),
        (26, "Will", "Stupido"),
        (27, "Shari", "Egoista"),
        (28, "Sethu", "Cause perse"),
    ],
    2024: [
        (1, "Angelina Mango", "La noia"),
        (2, "Geolier", "I p' me, tu p' te"),
        (3, "Annalisa", "Sinceramente"),
        (4, "Ghali", "Casa mia"),
        (5, "Irama", "Tu no"),
        (6, "Mahmood", "Tuta gold"),
        (7, "Loredana Berte", "Pazza"),
        (8, "Il Volo", "Capolavoro"),
        (9, "Alessandra Amoroso", "Fino a qui"),
        (10, "Alfa", "Vai!"),
        (11, "Gazzelle", "Tutto qui"),
        (12, "Il Tre", "Fragili"),
        (13, "Diodato", "Ti muovi"),
        (14, "Emma", "Apnea"),
        (15, "Fiorella Mannoia", "Mariposa"),
        (16, "The Kolors", "Un ragazzo una ragazza"),
        (17, "Mr. Rain", "Due altalene"),
        (18, "Santi Francesi", "L'amore in bocca"),
        (19, "Negramaro", "Ricominciamo tutto"),
        (20, "Dargen D'Amico", "Onda alta"),
        (21, "Ricchi e Poveri", "Ma non tutta la vita"),
        (22, "BigMama", "La rabbia non ti basta"),
        (23, "Rose Villain", "Click boom!"),
        (24, "Clara", "Diamanti grezzi"),
        (25, "Renga e Nek", "Pazzo di te"),
        (26, "Maninni", "Spigoli"),
        (27, "La Sad", "Autodistruttivo"),
        (28, "BNKR44", "Governo punk"),
        (29, "Sangiovanni", "Finiscimi"),
        (30, "Fred De Palma", "Il cielo non ci vuole"),
    ],
    2025: [
        (1, "Olly", "Balorda nostalgia"),
        (2, "Lucio Corsi", "Volevo essere un duro"),
        (3, "Brunori Sas", "L'albero delle noci"),
        (4, "Fedez", "Battito"),
        (5, "Simone Cristicchi", "Quando sarai piccola"),
        (6, "Giorgia", "La cura per me"),
        (7, "Achille Lauro", "Incoscienti giovani"),
        (8, "Francesco Gabbani", "Viva la vita"),
        (9, "Irama", "Lentamente"),
        (10, "Coma_Cose", "Cuoricini"),
        (11, "Bresh", "La tana del granchio"),
        (12, "Elodie", "Dimenticarsi alle 7"),
        (13, "Noemi", "Se t'innamori muori"),
        (14, "The Kolors", "Tu con chi fai l'amore"),
        (15, "Rocco Hunt", "Mille vote ancora"),
        (16, "Willie Peyote", "Grazie ma no grazie"),
        (17, "Sarah Toscano", "Amarcord"),
        (18, "Shablo feat. Gue, Joshua e Tormento", "La mia parola"),
        (19, "Rose Villain", "Fuorilegge"),
        (20, "Joan Thiele", "Eco"),
        (21, "Francesca Michielin", "Fango in paradiso"),
        (22, "Moda", "Non ti dimentico"),
        (23, "Massimo Ranieri", "Tra le mani un cuore"),
        (24, "Serena Brancale", "Anema e core"),
        (25, "Tony Effe", "Damme 'na mano"),
        (26, "Gaia", "Chiamo io chiami tu"),
        (27, "Clara", "Febbre"),
        (28, "Rkomi", "Il ritmo delle cose"),
        (29, "Marcella Bella", "Pelle diamante"),
    ],
}

# Performance orders by (year, serata) — from official scalette
# Each list contains artist names in order of performance
PERFORMANCE_ORDERS = {
    # === SANREMO 2023 ===
    # Serata 1 (Feb 7) — 14 of 28 artists
    (2023, 1): [
        "Anna Oxa", "gIANMARIA", "Mr. Rain", "Marco Mengoni", "Ariete",
        "Coma_Cose", "Gianluca Grignani", "Elodie", "Colapesce Dimartino",
        "Ultimo", "Madame", "Tananai", "Leo Gassmann", "I Cugini di Campagna",
    ],
    # Serata 2 (Feb 8) — the other 14 artists (complementary to serata 1)
    (2023, 2): [
        "Lazza", "Giorgia", "LDA", "Levante", "Rosa Chemical",
        "Articolo 31", "Moda", "Paola e Chiara", "Mara Sattei",
        "Olly", "Colla Zio", "Will", "Shari", "Sethu",
    ],

    # === SANREMO 2024 ===
    # Serata 1 (Feb 6) — ALL 30 artists performed
    (2024, 1): [
        "Clara", "Sangiovanni", "Fiorella Mannoia", "La Sad", "Irama",
        "Ghali", "Negramaro", "Annalisa", "Mahmood", "Diodato",
        "Loredana Berte", "Geolier", "Alessandra Amoroso", "The Kolors",
        "Angelina Mango", "Il Volo", "BigMama", "Ricchi e Poveri", "Emma",
        "Renga e Nek", "Mr. Rain", "BNKR44", "Gazzelle", "Dargen D'Amico",
        "Rose Villain", "Santi Francesi", "Fred De Palma", "Maninni",
        "Alfa", "Il Tre",
    ],

    # === SANREMO 2025 ===
    # Serata 1 (Feb 11) — ALL 29 artists performed
    (2025, 1): [
        "Gaia", "Francesco Gabbani", "Rkomi", "Noemi", "Irama",
        "Coma_Cose", "Simone Cristicchi", "Marcella Bella", "Achille Lauro",
        "Giorgia", "Willie Peyote", "Rose Villain", "Olly", "Elodie",
        "Shablo feat. Gue, Joshua e Tormento", "Massimo Ranieri", "Tony Effe",
        "Serena Brancale", "Brunori Sas", "Moda", "Clara", "Fedez",
        "Lucio Corsi", "Bresh", "Sarah Toscano", "Joan Thiele",
        "Rocco Hunt", "Francesca Michielin", "The Kolors",
    ],

    # === SANREMO 2026 ===
    # Serata 1 (Feb 24) — ALL 30 artists performed
    (2026, 1): [
        "Ditonellapiaga", "Michele Bravi", "Sayf", "Mara Sattei",
        "Dargen D'Amico", "Arisa", "Luche", "Tommaso Paradiso",
        "Elettra Lamborghini", "Patty Pravo", "Samurai Jay", "Raf",
        "J-Ax", "Fulminacci", "Levante", "Fedez e Masini", "Ermal Meta",
        "Serena Brancale", "Nayt", "Malika Ayane", "Eddie Brock",
        "Sal Da Vinci", "Enrico Nigiotti", "Tredici Pietro",
        "Bambole di Pezza", "Chiello", "Maria Antonietta e Colombre",
        "Leo Gassmann", "Francesco Renga", "LDA e Aka 7even",
    ],
    # Serata 2 (Feb 25) — 15 of 30 artists (from tweet timestamps)
    (2026, 2): [
        "Patty Pravo", "LDA e Aka 7even", "Enrico Nigiotti",
        "Tommaso Paradiso", "Elettra Lamborghini", "Ermal Meta",
        "Levante", "Bambole di Pezza", "Chiello", "J-Ax",
        "Nayt", "Fulminacci", "Fedez e Masini", "Dargen D'Amico",
        "Ditonellapiaga",
    ],
}

SANREMO_INFO = {
    2021: {'winner': 'Maneskin', 'dates': '2021-03-02/2021-03-06'},
    2022: {'winner': 'Mahmood e Blanco', 'dates': '2022-02-01/2022-02-05'},
    2023: {'winner': 'Marco Mengoni', 'dates': '2023-02-07/2023-02-11'},
    2024: {'winner': 'Angelina Mango', 'dates': '2024-02-06/2024-02-10'},
    2025: {'winner': 'Olly', 'dates': '2025-02-11/2025-02-15'},
    2026: {'winner': 'TBD (in corso)', 'dates': '2026-02-24/2026-02-28'},
}


# ============================================================
# PART 3: Normalize and match artist names
# ============================================================

def normalize(name):
    """Normalize artist name for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r'\s+feat\.?\s+', ' e ', name)
    name = re.sub(r'\s*&\s*', ' e ', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def match_artist(a, b):
    """Check if two artist names refer to the same artist."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    common = wa & wb
    if len(common) >= 1 and (len(common) / min(len(wa), len(wb))) >= 0.5:
        return True
    return False


def find_rank(artist, year):
    """Find the ranking for an artist in a given year."""
    if year not in RANKINGS:
        return None
    for rank, name, song in RANKINGS[year]:
        if match_artist(artist, name):
            return rank, name, song
    return None


# ============================================================
# PART 4: Build the complete dataset
# ============================================================

def build_dataset():
    """
    Build the full dataset combining:
    - Performance order from official scalette
    - Final rankings from Wikipedia
    - Tweet timestamps (where available, for 2026)
    """
    # Parse tweets for timestamps
    tweet_perfs = parse_tweets()
    tweet_lookup = {}
    for tp in tweet_perfs:
        key = (tp['year'], tp['serata'], normalize(tp['artist']))
        tweet_lookup[key] = tp

    rows = []
    for (year, serata), order in PERFORMANCE_ORDERS.items():
        total_in_serata = len(order)
        for pos_idx, artist in enumerate(order):
            performance_order = pos_idx + 1  # 1-indexed

            # Find ranking
            rank_result = find_rank(artist, year)
            rank = rank_result[0] if rank_result else None
            artist_wiki = rank_result[1] if rank_result else artist
            song_wiki = rank_result[2] if rank_result else ''

            # Check for tweet timestamp
            tweet_key = (year, serata, normalize(artist))
            tweet_data = tweet_lookup.get(tweet_key, {})

            row = {
                'year': year,
                'serata': serata,
                'artist': artist_wiki if rank_result else artist,
                'song': song_wiki,
                'rank': rank if rank else '',
                'performance_order': performance_order,
                'total_in_serata': total_in_serata,
                'relative_position': round((performance_order - 1) / max(total_in_serata - 1, 1), 4),
                'tweet_datetime': tweet_data.get('tweet_datetime', ''),
                'tweet_time': tweet_data.get('tweet_time', ''),
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PART 5: ML Model
# ============================================================

def build_ml_model(df):
    """Build and evaluate ML models to predict top placement from performance order."""
    # Keep only rows with valid rankings (exclude 2026 which is ongoing)
    df_ml = df[df['rank'] != ''].copy()
    df_ml['rank'] = df_ml['rank'].astype(int)

    # Remove duplicate artists within the same year (keep first serata appearance)
    df_ml = df_ml.sort_values(['year', 'serata']).drop_duplicates(subset=['year', 'artist'], keep='first')

    n = len(df_ml)
    print(f"\n{'='*60}")
    print("ML MODEL: Performance Order vs Final Ranking")
    print(f"{'='*60}")
    print(f"\nDataset: {n} artist-performances with known rankings")
    print(f"Years: {sorted(df_ml['year'].unique())}")

    # Features
    feature_cols = [
        'performance_order',
        'relative_position',
        'total_in_serata',
        'serata',
        'is_first_half',
        'is_last_quarter',
    ]

    df_ml['is_first_half'] = (df_ml['relative_position'] <= 0.5).astype(int)
    df_ml['is_last_quarter'] = (df_ml['relative_position'] >= 0.75).astype(int)

    X = df_ml[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Target: Top 5
    y_top5 = (df_ml['rank'] <= 5).astype(int)
    # Target: Top 10
    y_top10 = (df_ml['rank'] <= 10).astype(int)

    targets = {'Top 5': y_top5, 'Top 10': y_top10}
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=2),
    }

    results = []
    for target_name, y in targets.items():
        print(f"\n--- Target: {target_name} (positive={y.sum()}, negative={len(y)-y.sum()}) ---")

        for model_name, model in models.items():
            cv = StratifiedKFold(n_splits=min(5, n), shuffle=True, random_state=42)
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                print(f"  {model_name}: accuracy={scores.mean():.3f} (+/-{scores.std():.3f})")
                results.append({
                    'target': target_name,
                    'model': model_name,
                    'mean_accuracy': round(scores.mean(), 4),
                    'std_accuracy': round(scores.std(), 4),
                })
            except Exception as e:
                print(f"  {model_name}: error - {e}")

    # Feature importance
    print(f"\n--- Feature Importance (RandomForest, Top 5) ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=3)
    rf.fit(X_scaled, y_top5)
    for feat, imp in sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Correlation analysis
    print(f"\n--- Spearman Correlation: Features vs Rank ---")
    for col in ['performance_order', 'relative_position']:
        corr = df_ml[col].corr(df_ml['rank'], method='spearman')
        print(f"  {col} <-> rank: {corr:.4f} ({'higher order -> worse rank' if corr > 0 else 'higher order -> better rank'})")

    # Descriptive analysis by position groups
    print(f"\n--- Average Rank by Position Group ---")
    df_ml['position_group'] = pd.cut(
        df_ml['relative_position'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['1st quarter', '2nd quarter', '3rd quarter', '4th quarter'],
        include_lowest=True
    )
    for group, grp_df in df_ml.groupby('position_group', observed=True):
        avg_rank = grp_df['rank'].mean()
        top5_pct = (grp_df['rank'] <= 5).mean() * 100
        print(f"  {group}: avg_rank={avg_rank:.1f}, top5_rate={top5_pct:.1f}% (n={len(grp_df)})")

    return pd.DataFrame(results)


# ============================================================
# PART 6: Main pipeline
# ============================================================

def main():
    print("=" * 60)
    print("SANREMO PERFORMANCE TIME/ORDER ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Parse tweets
    print("\n[1/6] Parsing tweet data for timestamps...")
    tweet_perfs = parse_tweets()
    print(f"  Found {len(tweet_perfs)} performance tweets (from Sanremo 2026)")

    # Step 2: Build combined dataset
    print("\n[2/6] Building combined dataset (orders + rankings)...")
    df = build_dataset()
    print(f"  Total rows: {len(df)}")
    ranked = df[df['rank'] != '']
    print(f"  With rankings: {len(ranked)}")
    with_tweets = df[df['tweet_datetime'] != '']
    print(f"  With tweet timestamps: {len(with_tweets)}")

    # Step 3: Export tweet performance times
    print("\n[3/6] Exporting tweet performance times...")
    tweet_csv = os.path.join(OUTPUT_DIR, 'sanremo_tweet_performance_times.csv')
    pd.DataFrame(tweet_perfs).to_csv(tweet_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  -> {tweet_csv} ({len(tweet_perfs)} rows)")

    # Step 4: Export rankings
    print("\n[4/6] Exporting rankings...")
    rankings_rows = []
    for year, entries in RANKINGS.items():
        for rank, artist, song in entries:
            rankings_rows.append({'year': year, 'rank': rank, 'artist': artist, 'song': song})
    rankings_df = pd.DataFrame(rankings_rows)
    rankings_csv = os.path.join(OUTPUT_DIR, 'sanremo_rankings.csv')
    rankings_df.to_csv(rankings_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  -> {rankings_csv} ({len(rankings_df)} rows)")

    # Step 5: Export merged dataset
    print("\n[5/6] Exporting merged dataset...")
    merged_csv = os.path.join(OUTPUT_DIR, 'sanremo_merged_dataset.csv')
    df.to_csv(merged_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"  -> {merged_csv} ({len(df)} rows)")

    # Step 6: ML Model
    print("\n[6/6] Running ML models...")
    ml_results = build_ml_model(df)

    # Export ML results
    ml_csv = os.path.join(OUTPUT_DIR, 'sanremo_ml_results.csv')
    ml_results.to_csv(ml_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n  -> {ml_csv}")

    # Pipeline summary
    summary_rows = []
    for year in sorted(set(list(RANKINGS.keys()) + [2026])):
        year_data = df[df['year'] == year]
        year_ranked = year_data[year_data['rank'] != '']
        year_tweets = year_data[year_data['tweet_datetime'] != '']
        serate = sorted(year_data['serata'].unique())
        info = SANREMO_INFO.get(year, {})
        summary_rows.append({
            'year': year,
            'winner': info.get('winner', 'N/A'),
            'dates': info.get('dates', 'N/A'),
            'total_performances': len(year_data),
            'with_ranking': len(year_ranked),
            'with_tweet_time': len(year_tweets),
            'serate': ', '.join(map(str, serate)),
            'status': 'OK' if len(year_ranked) > 0 else ('in_corso' if year == 2026 else 'no_order_data')
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, 'sanremo_pipeline_summary.csv')
    summary_df.to_csv(summary_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n  -> {summary_csv}")
    print(f"\n  Pipeline Summary:")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print(f"  1. sanremo_tweet_performance_times.csv  - Tweet timestamps (2026)")
    print(f"  2. sanremo_rankings.csv                 - Final rankings (2021-2025)")
    print(f"  3. sanremo_merged_dataset.csv            - Full merged dataset")
    print(f"  4. sanremo_ml_results.csv                - ML model accuracy results")
    print(f"  5. sanremo_pipeline_summary.csv           - Pipeline verification")

    return df, ml_results


if __name__ == '__main__':
    main()
