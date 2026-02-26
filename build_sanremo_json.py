#!/usr/bin/env python3
"""
Sanremo Festival — Build dashboard JSON for the website.

Reads:
  - dati_sremo/sanremo_verified_data.json  (rankings, serate, performances)
  - dati_sremo/SanremoRai/*.json           (tweet timestamps)

Writes:
  - docs/sanremo_dashboard.json

Usage:
    python3 build_sanremo_json.py
"""

import json
import glob
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dati_sremo")
TWEET_DIR = os.path.join(DATA_DIR, "SanremoRai")
VERIFIED_JSON = os.path.join(DATA_DIR, "sanremo_verified_data.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "docs", "sanremo_dashboard.json")


def load_verified():
    with open(VERIFIED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_tweets():
    """Parse tweet JSONs and return list of dicts with metadata."""
    results = []
    for fp in sorted(glob.glob(os.path.join(TWEET_DIR, "*.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        year = None
        for h in d.get("hashtags", []):
            m = re.search(r"[Ss]anremo(\d{4})", h)
            if m:
                year = int(m.group(1))
                break
        if not year:
            continue
        content = d.get("content", "")
        artist_m = re.search(r"🎤\s*(.+?)(?:\n|$)", content)
        song_m = re.search(r"🎧\s*(.+?)(?:\n|$)", content)
        serata_m = re.search(r"[Ss]erata\s*(\d+)", content)
        results.append({
            "year": year,
            "serata": int(serata_m.group(1)) if serata_m else None,
            "artist": artist_m.group(1).strip() if artist_m else None,
            "song": song_m.group(1).strip() if song_m else None,
            "tweet_datetime": d.get("date", ""),
            "has_performance": bool(artist_m and song_m),
        })
    return results


def time_to_minutes(time_str):
    """Convert HH:MM to minutes-since-20:00."""
    if not time_str:
        return None
    h, m = map(int, time_str.split(":"))
    if h < 6:
        h += 24
    return (h - 20) * 60 + m


def build_dashboard(verified, tweets):
    """Build the complete dashboard JSON object."""

    # --- meta ---
    years = sorted(verified.keys())
    completed_years = [y for y in years if verified[y].get("winner")]
    meta = {
        "source": "Festival di Sanremo — dati raccolti da Wikipedia, tweet @SanremoRai, fonti giornalistiche",
        "last_update": datetime.now().strftime("%Y-%m-%d"),
        "years": years,
        "completed_years": completed_years,
        "total_editions": len(years),
        "description": "Classifiche, ordini di esibizione e orari per le edizioni 2021-2026 del Festival di Sanremo",
    }

    # --- editions (summary per year) ---
    editions = []
    for y in years:
        ed = verified[y]
        n_serate = len(ed.get("serate", {}))
        n_performances = sum(
            len(s["performances"])
            for s in ed.get("serate", {}).values()
        )
        n_with_time = sum(
            1
            for s in ed.get("serate", {}).values()
            for p in s["performances"]
            if p.get("time")
        )
        editions.append({
            "year": int(y),
            "edition": ed["edition"],
            "dates": ed["dates"],
            "location": ed["location"],
            "host": ed["host"],
            "num_artists": ed["num_artists"],
            "winner": ed.get("winner"),
            "winner_song": ed.get("winner_song"),
            "num_serate_with_data": n_serate,
            "total_performances": n_performances,
            "performances_with_time": n_with_time,
            "has_rankings": len(ed.get("rankings", [])) > 0,
            "status": "in_corso" if not ed.get("winner") else "completato",
        })

    # --- kpi ---
    all_artists = set()
    for y in years:
        for r in verified[y].get("rankings", []):
            all_artists.add(r["artist"])
    latest_complete = max(completed_years) if completed_years else None
    kpi = {
        "total_editions_tracked": len(years),
        "total_unique_artists": len(all_artists),
        "latest_winner": verified[latest_complete]["winner"] if latest_complete else None,
        "latest_winner_song": verified[latest_complete]["winner_song"] if latest_complete else None,
        "latest_winner_year": int(latest_complete) if latest_complete else None,
    }

    # --- rankings (all years, flat list) ---
    rankings = []
    for y in years:
        for r in verified[y].get("rankings", []):
            rankings.append({
                "year": int(y),
                "rank": r["rank"],
                "artist": r["artist"],
                "song": r["song"],
            })

    # --- performances (all serate, flat list with computed fields) ---
    performances = []
    for y in years:
        ed = verified[y]
        rank_map = {}
        for r in ed.get("rankings", []):
            rank_map[r["artist"].lower().strip()] = r["rank"]

        for sk, sdata in ed.get("serate", {}).items():
            n = sdata["num_performers"]
            for perf in sdata["performances"]:
                order = perf["order"]
                time_str = perf.get("time")
                minutes = time_to_minutes(time_str)
                rel_pos = round((order - 1) / max(n - 1, 1), 4)

                # Try to find rank
                artist_lower = perf["artist"].lower().strip()
                rank = rank_map.get(artist_lower)
                if rank is None:
                    for rk, rv in rank_map.items():
                        if artist_lower in rk or rk in artist_lower:
                            rank = rv
                            break

                performances.append({
                    "year": int(y),
                    "serata": int(sk),
                    "serata_name": sdata["name"],
                    "serata_date": sdata["date"],
                    "artist": perf["artist"],
                    "performance_order": order,
                    "total_in_serata": n,
                    "relative_position": rel_pos,
                    "time": time_str,
                    "minutes_since_2000": minutes,
                    "has_exact_time": time_str is not None,
                    "final_rank": rank,
                })

    # --- tweet_activity (tweets per year, per type) ---
    tweet_counts = {}
    for tw in tweets:
        y = tw["year"]
        if y not in tweet_counts:
            tweet_counts[y] = {"total": 0, "with_performance": 0}
        tweet_counts[y]["total"] += 1
        if tw["has_performance"]:
            tweet_counts[y]["with_performance"] += 1

    tweet_activity = [
        {
            "year": y,
            "total_tweets": v["total"],
            "performance_tweets": v["with_performance"],
        }
        for y, v in sorted(tweet_counts.items())
    ]

    # --- winners_timeline ---
    winners_timeline = []
    for y in completed_years:
        ed = verified[y]
        winners_timeline.append({
            "year": int(y),
            "edition": ed["edition"],
            "winner": ed["winner"],
            "song": ed["winner_song"],
            "host": ed["host"],
            "num_artists": ed["num_artists"],
        })

    # --- artist_appearances (cross-year presence) ---
    artist_years = {}
    for y in years:
        for r in verified[y].get("rankings", []):
            a = r["artist"]
            if a not in artist_years:
                artist_years[a] = []
            artist_years[a].append({
                "year": int(y),
                "rank": r["rank"],
                "song": r["song"],
            })
    # Only keep artists with >1 appearance
    returning_artists = [
        {"artist": a, "appearances": apps, "num_editions": len(apps)}
        for a, apps in sorted(artist_years.items())
        if len(apps) > 1
    ]

    # Assemble dashboard
    dashboard = {
        "meta": meta,
        "kpi": kpi,
        "editions": editions,
        "winners_timeline": winners_timeline,
        "rankings": rankings,
        "performances": performances,
        "returning_artists": returning_artists,
        "tweet_activity": tweet_activity,
    }

    return dashboard


def main():
    print("Caricamento dati verificati...")
    verified = load_verified()
    print(f"  Anni: {list(verified.keys())}")

    print("Parsing tweet...")
    tweets = parse_tweets()
    print(f"  Tweet totali: {len(tweets)}")

    print("Costruzione dashboard JSON...")
    dashboard = build_dashboard(verified, tweets)

    print(f"Scrittura {OUTPUT_JSON}...")
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\nDashboard JSON generato:")
    print(f"  Edizioni:     {len(dashboard['editions'])}")
    print(f"  Classifiche:  {len(dashboard['rankings'])} artisti")
    print(f"  Esibizioni:   {len(dashboard['performances'])} performance")
    print(f"  Artisti multi-edizione: {len(dashboard['returning_artists'])}")
    print(f"  Tweet activity: {len(dashboard['tweet_activity'])} anni")
    print(f"\n  Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
