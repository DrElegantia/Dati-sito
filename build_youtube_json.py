#!/usr/bin/env python3
"""build_youtube_json.py – Pipeline giornaliera dati YouTube EconomiaItalia.

Genera docs/youtube_dashboard.json con:
1. Views mensili (shorts vs longs)
2. Iscritti (snapshot giornaliero + gain mensile)
3. Numero video pubblicati al mese
4. Media views per video per mese
5. Dati boxplot views per mese (ultimi 12 mesi)

Shorts = video con durata <= 60 secondi.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import isodate
import numpy as np
from googleapiclient.discovery import build

CHANNEL_ID = "UCq5fU1zDj0KcdWM6wjI0jhA"  # EconomiaItalia
SHORTS_MAX_SECS = 60
OUTPUT = Path("docs/youtube_dashboard.json")
HISTORICAL_CSV = Path("yt/Totali.csv")


def yt_client():
    key = os.environ.get("YT_API_KEY")
    if not key:
        sys.exit("ERROR: variabile d'ambiente YT_API_KEY non impostata")
    return build("youtube", "v3", developerKey=key)


def channel_stats(yt):
    resp = yt.channels().list(
        part="snippet,contentDetails,statistics", id=CHANNEL_ID
    ).execute()
    if not resp.get("items"):
        sys.exit(f"Canale {CHANNEL_ID} non trovato")
    it = resp["items"][0]
    return {
        "name": it["snippet"]["title"],
        "subscribers": int(it["statistics"].get("subscriberCount", 0)),
        "totalViews": int(it["statistics"].get("viewCount", 0)),
        "totalVideos": int(it["statistics"].get("videoCount", 0)),
        "playlistId": it["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def all_video_ids(yt, playlist_id):
    ids, token = [], None
    while True:
        kw = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if token:
            kw["pageToken"] = token
        resp = yt.playlistItems().list(**kw).execute()
        ids.extend(item["contentDetails"]["videoId"] for item in resp["items"])
        token = resp.get("nextPageToken")
        if not token:
            break
    return ids


def video_details(yt, ids):
    videos = []
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(ids[i : i + 50]),
        ).execute()
        for v in resp["items"]:
            try:
                dur = isodate.parse_duration(v["contentDetails"]["duration"]).total_seconds()
            except Exception:
                dur = 0
            videos.append(
                {
                    "id": v["id"],
                    "title": v["snippet"]["title"],
                    "publishedAt": v["snippet"]["publishedAt"],
                    "viewCount": int(v["statistics"].get("viewCount", 0)),
                    "likeCount": int(v["statistics"].get("likeCount", 0)),
                    "commentCount": int(v["statistics"].get("commentCount", 0)),
                    "durationSecs": dur,
                    "isShort": 0 < dur <= SHORTS_MAX_SECS,
                }
            )
    return videos


def monthly_aggregations(videos):
    buckets = defaultdict(lambda: {"shorts": [], "longs": []})
    for v in videos:
        m = v["publishedAt"][:7]  # YYYY-MM
        k = "shorts" if v["isShort"] else "longs"
        buckets[m][k].append(v["viewCount"])

    months = sorted(buckets)

    views = {
        t: [{"month": m, "views": int(sum(buckets[m][t]))} for m in months]
        for t in ("shorts", "longs")
    }
    count = {
        t: [{"month": m, "count": len(buckets[m][t])} for m in months]
        for t in ("shorts", "longs")
    }
    avg = {
        t: [
            {
                "month": m,
                "avg": round(float(np.mean(buckets[m][t])), 1)
                if buckets[m][t]
                else 0,
            }
            for m in months
        ]
        for t in ("shorts", "longs")
    }

    # Boxplot: ultimi 12 mesi
    now = datetime.now(timezone.utc)
    y, mo = now.year, now.month - 12
    while mo <= 0:
        mo += 12
        y -= 1
    cutoff = f"{y}-{mo:02d}"
    recent = [m for m in months if m >= cutoff]

    bp = {}
    for t in ("shorts", "longs"):
        bp[t] = []
        for m in recent:
            arr = np.array(buckets[m][t], dtype=float)
            if arr.size == 0:
                bp[t].append(
                    {"month": m, "min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "outliers": [], "n": 0}
                )
                continue
            q1, med, q3 = float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))
            iqr = q3 - q1
            lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inside = arr[(arr >= lo_fence) & (arr <= hi_fence)]
            wlo = float(inside.min()) if inside.size else float(arr.min())
            whi = float(inside.max()) if inside.size else float(arr.max())
            outliers = arr[(arr < lo_fence) | (arr > hi_fence)].tolist()
            bp[t].append(
                {
                    "month": m,
                    "min": round(wlo, 1),
                    "q1": round(q1, 1),
                    "median": round(med, 1),
                    "q3": round(q3, 1),
                    "max": round(whi, 1),
                    "outliers": [round(o, 1) for o in outliers],
                    "n": int(arr.size),
                }
            )

    return views, count, avg, bp


def load_sub_history():
    if OUTPUT.exists():
        try:
            return json.loads(OUTPUT.read_text("utf-8")).get("subscriber_history", [])
        except Exception:
            return []
    return []


def load_fixed_fields():
    """Carica campi fissi (non derivabili da API) dal JSON esistente."""
    fixed = {}
    if OUTPUT.exists():
        try:
            data = json.loads(OUTPUT.read_text("utf-8"))
            for key in ("audience_age_distribution",):
                if key in data:
                    fixed[key] = data[key]
        except Exception:
            pass
    return fixed


def load_historical_csv():
    """Carica yt/Totali.csv → lista di (date, daily_gain)."""
    if not HISTORICAL_CSV.exists():
        return []
    rows = []
    with open(HISTORICAL_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Date"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                rows.append((parts[0], int(parts[1])))
    return rows


def historical_monthly_snapshots(csv_rows, anchor_count, anchor_cumsum=None):
    """Dai gain giornalieri del CSV, ricostruisce snapshot mensili cumulativi.

    anchor_count è il conteggio iscritti al punto di ancoraggio.
    anchor_cumsum è il cumsum CSV al punto di ancoraggio (default: cumsum totale).
    offset = anchor_count − anchor_cumsum.
    Per ogni mese prende l'ultimo giorno disponibile e riporta offset + cumsum.
    """
    if not csv_rows:
        return []
    cumsum = 0
    cumulative = []
    for date_str, gain in csv_rows:
        cumsum += gain
        cumulative.append((date_str, cumsum))

    if anchor_cumsum is None:
        anchor_cumsum = cumsum
    offset = anchor_count - anchor_cumsum

    monthly = {}
    for date_str, cum in cumulative:
        month = date_str[:7]
        monthly[month] = {"date": date_str, "count": max(0, offset + cum)}

    return [monthly[m] for m in sorted(monthly)]


def build_full_sub_history(csv_rows, existing_history, current_count):
    """Combina storico CSV (mensile) + snapshot API giornalieri.

    L'offset CSV è ancorato alla PRIMA entry API disponibile (continuità tra
    fine CSV e inizio API). Senza questa correzione, il `current_count` cresce
    nel tempo ma le entry API storiche restano scolpite nei loro valori
    originali, creando un gradino artificiale (e gain mensili negativi falsi).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if csv_rows:
        last_csv_date = csv_rows[-1][0]
        # Mantieni solo le entry API successive all'ultimo giorno del CSV
        api_entries = [e for e in existing_history if e["date"] > last_csv_date]
        # Anchor: prima entry API disponibile (fallback al current count se assente)
        if api_entries:
            anchor_count = api_entries[0]["count"]
        else:
            anchor_count = current_count
        anchor_cumsum = sum(g for _, g in csv_rows)
        hist_monthly = historical_monthly_snapshots(
            csv_rows, anchor_count, anchor_cumsum=anchor_cumsum
        )
    else:
        hist_monthly = []
        api_entries = list(existing_history)

    # Aggiorna / aggiungi il dato di oggi
    if api_entries and api_entries[-1]["date"] == today:
        api_entries[-1]["count"] = current_count
    else:
        api_entries.append({"date": today, "count": current_count})

    return hist_monthly + api_entries


def monthly_sub_gains(hist):
    by_month = {}
    for e in hist:
        by_month[e["date"][:7]] = e["count"]
    ms = sorted(by_month)
    return [
        {"month": ms[i], "gain": by_month[ms[i]] - by_month[ms[i - 1]]}
        for i in range(1, len(ms))
    ]


def main():
    yt = yt_client()

    print("Fetching channel stats...")
    ch = channel_stats(yt)
    print(f"  {ch['name']} — {ch['subscribers']:,} iscritti, {ch['totalVideos']:,} video")

    print("Fetching video IDs...")
    ids = all_video_ids(yt, ch["playlistId"])
    print(f"  {len(ids)} video trovati")

    print("Fetching video details...")
    vids = video_details(yt, ids)
    ns = sum(1 for v in vids if v["isShort"])
    print(f"  Shorts: {ns}, Longs: {len(vids) - ns}")

    print("Computing monthly aggregations...")
    m_views, m_count, m_avg, bp = monthly_aggregations(vids)

    print("Building subscriber history (CSV + API)...")
    csv_rows = load_historical_csv()
    existing_hist = load_sub_history()
    sub_hist = build_full_sub_history(csv_rows, existing_hist, ch["subscribers"])
    print(f"  {len(sub_hist)} data points ({len(csv_rows)} giorni da CSV storico)")
    sub_gains = monthly_sub_gains(sub_hist)
    fixed = load_fixed_fields()

    pwd = os.environ.get("YT_DASHBOARD_PWD", "economiaitalia")
    pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()

    payload = {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pwd_hash": pwd_hash,
        "channel": {
            "name": ch["name"],
            "subscribers": ch["subscribers"],
            "totalViews": ch["totalViews"],
            "totalVideos": ch["totalVideos"],
        },
        "subscriber_history": sub_hist,
        "subscriber_monthly_gains": sub_gains,
        "monthly_views": m_views,
        "monthly_videos_published": m_count,
        "monthly_avg_views": m_avg,
        "boxplot_data": bp,
        **fixed,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    print(f"Output scritto in {OUTPUT}")


if __name__ == "__main__":
    main()
