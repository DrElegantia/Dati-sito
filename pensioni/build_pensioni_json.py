"""
Genera i file JSON per la dashboard pensioni e indicizzazione.

Fonti:
- INPS, Casellario dei Pensionati (distribuzione per classi di reddito)
- Sentenza 167/2025 Corte Costituzionale
- Analisi Reforming.it sulla de-indicizzazione

Due livelli di analisi:
1. Convergenza individuale: tempo per cui la pensione sotto-indicizzata
   converge in termini reali al livello target PG = PA * (1 - sconto)
2. Risparmio aggregato: risparmio totale per lo Stato su 5 anni
"""

import json
import math
import openpyxl
import os

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
TRATTAMENTO_MINIMO_MENSILE = 598.61
TRATTAMENTO_MINIMO_ANNUO = 7183.32

TASSI_INFLAZIONE = [0.02, 0.03]
VALORI_K = [0.75, 0.50, 0.25, 0.00]
SCONTI = [0.10, 0.20, 0.33]

SOGLIA_ESENZIONE = 4  # fasce fino a 4x il minimo hanno k=1
ORIZZONTE_RISPARMIO = 5

# Soglia alternativa: 60% mediana redditi (fonte FT, mediana 30.755 EUR/anno)
MEDIANA_REDDITI_ANNUA = 30755
SOGLIA_MEDIANA_FRAZ = 0.60
SOGLIA_MEDIANA_ANNUA = MEDIANA_REDDITI_ANNUA * SOGLIA_MEDIANA_FRAZ  # 18.453 EUR/anno
SOGLIA_MEDIANA_MENSILE = SOGLIA_MEDIANA_ANNUA / 12  # 1.537,75 EUR/mese
SOGLIA_MEDIANA_MULTIPLO = SOGLIA_MEDIANA_ANNUA / TRATTAMENTO_MINIMO_ANNUO  # ~2.57x minimo

# ---------------------------------------------------------------------------
# Lettura dati
# ---------------------------------------------------------------------------
def leggi_dati(path):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    fasce = []
    for row in rows:
        classe, range_mensile, n_pensionati, reddito_totale, reddito_medio = row
        fasce.append({
            "classe": classe,
            "range_mensile": range_mensile,
            "n_pensionati": int(n_pensionati),
            "reddito_totale_annuo": float(reddito_totale),
            "reddito_medio_annuo": float(reddito_medio),
        })
    return fasce


def moltiplicatore_fascia(classe):
    """Restituisce il limite inferiore del moltiplicatore della fascia."""
    if classe.startswith("Fino a"):
        return 0
    if classe.startswith("Oltre"):
        return 50
    # "Da X a Y volte il minimo"
    parts = classe.replace("Da ", "").split(" a ")
    return int(parts[0].strip())


# ---------------------------------------------------------------------------
# Livello 1: Convergenza individuale
# ---------------------------------------------------------------------------
def calcola_convergenza(PA, sconto, i, k):
    PG = PA * (1 - sconto)
    if k >= 1.0 or PA <= PG:
        return None  # nessuna convergenza necessaria o possibile

    rapporto = (1 + i) / (1 + k * i)
    if rapporto <= 1:
        return None

    t_star = math.log(PA / PG) / math.log(rapporto)
    anni = math.ceil(t_star)

    # Traiettoria anno per anno
    traiettoria = []
    cumulata = 0.0
    for t in range(anni + 1):
        # Pensione nominale con indicizzazione parziale k
        nominale_k = PA * ((1 + k * i) ** t)
        # Pensione nominale con indicizzazione piena
        nominale_100 = PA * ((1 + i) ** t)
        # Valori reali (deflazionati)
        deflatore = (1 + i) ** t
        reale_k = nominale_k / deflatore
        reale_100 = nominale_100 / deflatore  # = PA sempre

        diff_annua = reale_100 - reale_k
        cumulata += diff_annua

        traiettoria.append({
            "anno": t,
            "pensione_reale_k": round(reale_k, 2),
            "pensione_reale_100": round(reale_100, 2),
            "diff_reale_annua": round(diff_annua, 2),
            "diff_reale_cumulata": round(cumulata, 2),
        })

    return {
        "inflazione": i,
        "k": k,
        "sconto": sconto,
        "PA": round(PA, 2),
        "PG": round(PG, 2),
        "anni_convergenza": anni,
        "perdita_reale_cumulata": round(cumulata, 2),
        "traiettoria": traiettoria,
    }


def genera_convergenza(fasce, soglia_multiplo, soglia_label, soglia_annua=None):
    fasce_output = []
    for fascia in fasce:
        mult = moltiplicatore_fascia(fascia["classe"])
        if soglia_annua is not None:
            # Soglia basata su importo annuo (60% mediana)
            if fascia["reddito_medio_annuo"] < soglia_annua:
                continue
        else:
            if mult < soglia_multiplo:
                continue

        PA = fascia["reddito_medio_annuo"]
        scenari = []
        for i in TASSI_INFLAZIONE:
            for k in VALORI_K:
                for sconto in SCONTI:
                    risultato = calcola_convergenza(PA, sconto, i, k)
                    if risultato:
                        scenari.append(risultato)

        fasce_output.append({
            "classe": fascia["classe"],
            "range_mensile": fascia["range_mensile"],
            "n_pensionati": fascia["n_pensionati"],
            "reddito_medio_annuo": fascia["reddito_medio_annuo"],
            "scenari": scenari,
        })

    return {
        "metadata": {
            "fonte_dati": "INPS - Casellario dei Pensionati",
            "anno_riferimento": 2024,
            "trattamento_minimo_mensile": TRATTAMENTO_MINIMO_MENSILE,
            "trattamento_minimo_annuo": TRATTAMENTO_MINIMO_ANNUO,
            "soglia": soglia_label,
            "nota": f"Simulazione della convergenza tramite sotto-indicizzazione. Soglia: {soglia_label}"
        },
        "parametri": {
            "tassi_inflazione": TASSI_INFLAZIONE,
            "valori_k": VALORI_K,
            "sconti": SCONTI,
        },
        "fasce": fasce_output,
    }


# ---------------------------------------------------------------------------
# Livello 2: Risparmio aggregato (5 anni)
# ---------------------------------------------------------------------------
def genera_risparmio(fasce, soglia_multiplo, soglia_label, soglia_annua=None):
    scenari_output = []

    for i in TASSI_INFLAZIONE:
        for k in VALORI_K:
            per_fascia = []
            totale_risparmio_reale = 0.0
            totale_risparmio_nominale = 0.0
            n_coinvolti = 0
            n_esclusi = 0

            for fascia in fasce:
                mult = moltiplicatore_fascia(fascia["classe"])
                reddito_totale = fascia["reddito_totale_annuo"]
                n = fascia["n_pensionati"]

                if soglia_annua is not None:
                    esente = fascia["reddito_medio_annuo"] < soglia_annua
                else:
                    esente = mult < soglia_multiplo

                if esente:
                    k_eff = 1.0
                    n_esclusi += n
                else:
                    k_eff = k
                    n_coinvolti += n

                risparmio_nom_5y = 0.0
                risparmio_reale_5y = 0.0

                for t in range(1, ORIZZONTE_RISPARMIO + 1):
                    costo_pieno = reddito_totale * ((1 + i) ** t)
                    costo_parziale = reddito_totale * ((1 + k_eff * i) ** t)
                    risp_nom = costo_pieno - costo_parziale
                    risp_reale = risp_nom / ((1 + i) ** t)
                    risparmio_nom_5y += risp_nom
                    risparmio_reale_5y += risp_reale

                totale_risparmio_reale += risparmio_reale_5y
                totale_risparmio_nominale += risparmio_nom_5y

                per_fascia.append({
                    "classe": fascia["classe"],
                    "n_pensionati": n,
                    "reddito_totale_annuo": round(reddito_totale, 2),
                    "risparmio_reale_5y": round(risparmio_reale_5y, 2),
                    "risparmio_nominale_5y": round(risparmio_nom_5y, 2),
                    "risparmio_medio_per_pensionato": round(
                        risparmio_reale_5y / n, 2
                    ) if n > 0 else 0,
                })

            medio_coinvolto = (
                round(totale_risparmio_reale / n_coinvolti, 2)
                if n_coinvolti > 0 else 0
            )

            scenari_output.append({
                "inflazione": i,
                "k": k,
                "soglia_esenzione": soglia_label,
                "totale": {
                    "risparmio_reale_5y": round(totale_risparmio_reale, 2),
                    "risparmio_nominale_5y": round(totale_risparmio_nominale, 2),
                    "n_pensionati_coinvolti": n_coinvolti,
                    "n_pensionati_esclusi": n_esclusi,
                    "risparmio_medio_per_pensionato_coinvolto": medio_coinvolto,
                },
                "per_fascia": per_fascia,
            })

    return {
        "metadata": {
            "fonte_dati": "INPS - Casellario dei Pensionati",
            "anno_riferimento": 2024,
            "trattamento_minimo_mensile": TRATTAMENTO_MINIMO_MENSILE,
            "orizzonte_anni": ORIZZONTE_RISPARMIO,
            "nota": f"Risparmio aggregato da sotto-indicizzazione. Soglia: {soglia_label}.",
        },
        "scenari": scenari_output,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    xlsx_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "pensioni_classi_reddito.xlsx"
    )
    fasce = leggi_dati(xlsx_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Definizione soglie
    soglie = [
        {
            "id": "4x_minimo",
            "label": f"{SOGLIA_ESENZIONE}x minimo ({SOGLIA_ESENZIONE * TRATTAMENTO_MINIMO_MENSILE:.2f} EUR/mese)",
            "multiplo": SOGLIA_ESENZIONE,
            "annua": None,
        },
        {
            "id": "60pct_mediana",
            "label": f"60% mediana redditi ({SOGLIA_MEDIANA_MENSILE:.2f} EUR/mese, {SOGLIA_MEDIANA_ANNUA:.0f} EUR/anno)",
            "multiplo": SOGLIA_MEDIANA_MULTIPLO,
            "annua": SOGLIA_MEDIANA_ANNUA,
        },
    ]

    # Output unico con entrambe le soglie
    all_convergenza = {}
    all_risparmio = {}

    for soglia in soglie:
        sid = soglia["id"]
        label = soglia["label"]
        print(f"\n=== Soglia: {label} ===")

        conv = genera_convergenza(fasce, soglia["multiplo"], label, soglia["annua"])
        risp = genera_risparmio(fasce, soglia["multiplo"], label, soglia["annua"])

        all_convergenza[sid] = conv
        all_risparmio[sid] = risp

        # Sanity check
        for s in risp["scenari"]:
            tot = s["totale"]
            print(
                f"  i={s['inflazione']}, k={s['k']}: "
                f"risparmio reale 5y = {tot['risparmio_reale_5y']/1e9:.2f} mld EUR, "
                f"coinvolti = {tot['n_pensionati_coinvolti']:,}, "
                f"esclusi = {tot['n_pensionati_esclusi']:,}"
            )

    # Scrivi JSON con entrambe le soglie
    out_conv = os.path.join(base_dir, "pensioni_convergenza.json")
    with open(out_conv, "w", encoding="utf-8") as f:
        json.dump(all_convergenza, f, ensure_ascii=False, indent=2)
    print(f"\nScritto {out_conv}")

    out_risp = os.path.join(base_dir, "pensioni_risparmio.json")
    with open(out_risp, "w", encoding="utf-8") as f:
        json.dump(all_risparmio, f, ensure_ascii=False, indent=2)
    print(f"Scritto {out_risp}")


if __name__ == "__main__":
    main()
