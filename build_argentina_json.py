"""
Genera docs/argentina_dashboard.json — Analisi macroeconomica dell'Argentina
di Milei (dic 2023 – 2026).

Dati raccolti da: Osservatorio CPI (Unicatt), IMF, OECD, INDEC, BCRA,
Trading Economics, World Bank, Allianz Trade, BBVA Research.
"""

from pathlib import Path
import json

# ── PIL trimestrale (var. % t/t destagionalizzata) ──────────────────────────
PIL_TRIMESTRALE = [
    {"trimestre": "2024-Q1", "var_tt": -2.2},
    {"trimestre": "2024-Q2", "var_tt": -2.4},
    {"trimestre": "2024-Q3", "var_tt": 3.6},
    {"trimestre": "2024-Q4", "var_tt": 2.0},
    {"trimestre": "2025-Q1", "var_tt": 1.1},
    {"trimestre": "2025-Q2", "var_tt": -0.1},
    {"trimestre": "2025-Q3", "var_tt": 0.3},
]

# ── PIL annuale (var. % a/a reale) ──────────────────────────────────────────
PIL_ANNUALE = [
    {"anno": 2019, "var_aa": -2.0, "fonte": "INDEC"},
    {"anno": 2020, "var_aa": -9.9, "fonte": "INDEC"},
    {"anno": 2021, "var_aa": 10.7, "fonte": "INDEC"},
    {"anno": 2022, "var_aa": 5.0, "fonte": "INDEC"},
    {"anno": 2023, "var_aa": -1.6, "fonte": "INDEC"},
    {"anno": 2024, "var_aa": -1.3, "fonte": "INDEC"},
    {"anno": 2025, "var_aa": 5.2, "fonte": "OECD stima"},
    {"anno": 2026, "var_aa": 4.3, "fonte": "OECD stima"},
]

# ── Stime PIL 2025-2026 a confronto ────────────────────────────────────────
STIME_PIL = [
    {"istituzione": "IMF", "anno_2025": 4.5, "anno_2026": 3.5},
    {"istituzione": "OECD", "anno_2025": 5.2, "anno_2026": 4.3},
    {"istituzione": "World Bank", "anno_2025": 4.6, "anno_2026": 4.0},
    {"istituzione": "Allianz Trade", "anno_2025": 4.4, "anno_2026": 3.5},
    {"istituzione": "BBVA Research", "anno_2025": 5.5, "anno_2026": None},
]

# ── Inflazione annua (%, dic/dic) ───────────────────────────────────────────
INFLAZIONE_ANNUALE = [
    {"anno": 2019, "inflazione": 53.8},
    {"anno": 2020, "inflazione": 36.1},
    {"anno": 2021, "inflazione": 50.9},
    {"anno": 2022, "inflazione": 94.8},
    {"anno": 2023, "inflazione": 211.4},
    {"anno": 2024, "inflazione": 117.8},
]

# ── Inflazione mensile (% m/m, selezione mesi chiave) ──────────────────────
INFLAZIONE_MENSILE = [
    {"mese": "2023-11", "var_mm": 12.8},
    {"mese": "2023-12", "var_mm": 25.5},
    {"mese": "2024-01", "var_mm": 20.6},
    {"mese": "2024-02", "var_mm": 13.2},
    {"mese": "2024-03", "var_mm": 11.0},
    {"mese": "2024-04", "var_mm": 8.8},
    {"mese": "2024-05", "var_mm": 4.2},
    {"mese": "2024-06", "var_mm": 4.6},
    {"mese": "2024-07", "var_mm": 4.0},
    {"mese": "2024-08", "var_mm": 4.2},
    {"mese": "2024-09", "var_mm": 3.5},
    {"mese": "2024-10", "var_mm": 2.7},
    {"mese": "2024-11", "var_mm": 2.4},
    {"mese": "2024-12", "var_mm": 2.7},
    {"mese": "2025-01", "var_mm": 2.2},
    {"mese": "2025-02", "var_mm": 2.4},
    {"mese": "2025-03", "var_mm": 3.7},
    {"mese": "2025-04", "var_mm": 2.8},
    {"mese": "2025-05", "var_mm": 2.3},
    {"mese": "2025-06", "var_mm": 1.5},
    {"mese": "2025-07", "var_mm": 1.8},
    {"mese": "2025-08", "var_mm": 1.9},
    {"mese": "2025-09", "var_mm": 2.1},
]

# ── Inflazione annua a settembre 2025 (INDEC confermato) ────────────────────
INFLAZIONE_ANNUA_SET_2025 = {
    "valore_pct": 31.8,
    "nota": (
        "Ancora alta per standard internazionali e superiore ai livelli "
        "pre-pandemia, ma in forte calo dal 211.4% di fine 2023. Il "
        "'collasso' dell'inflazione è relativo: dal 25% mensile (dic 2023) "
        "a ~2% mensile (2025), ma l'inflazione annua del 31.8% resta elevata."
    ),
}

# ── Tasso di policy BCRA ────────────────────────────────────────────────────
TASSO_POLICY_BCRA = [
    {"data": "2023-12", "tasso_pct": 117.0, "nota": "picco post-elezioni"},
    {"data": "2024-06", "tasso_pct": 50.0},
    {"data": "2024-12", "tasso_pct": 32.0},
    {"data": "2025-02", "tasso_pct": 29.0, "nota": "tassi reali ancora positivi per sostenere disinflazione"},
]

# ── Spesa primaria (IMF, aprile 2025) ──────────────────────────────────────
SPESA_PRIMARIA = {
    "taglio_aa_pct": -30,
    "fonte": "IMF Country Report, aprile 2025",
    "dettaglio": (
        "Tagli profondi a sussidi, pensioni e trasferimenti provinciali. "
        "L'assenza di finanziamento monetario e il regime di cambi semi-fisso "
        "con crawl rate hanno sostenuto la disinflazione."
    ),
    "surplus_cassa_2024_pil_pct": 0.3,
}

# ── Stime inflazione 2025-2027 ─────────────────────────────────────────────
STIME_INFLAZIONE = [
    {"istituzione": "OECD", "media_2025": 37.0, "media_2026": 15.0},
    {"istituzione": "IMF", "fine_2025": 23.0, "fine_2026": None},
    {"istituzione": "Allianz Trade", "fine_2025": 20.0, "fine_2026": 18.0},
    {"istituzione": "BCRA survey (mag 2025)", "fine_2025": 28.6, "fine_2026": 16.0},
]

# ── Bilancio fiscale (% del PIL) ───────────────────────────────────────────
BILANCIO_FISCALE = [
    {"anno": 2019, "saldo_primario": -0.4, "saldo_complessivo": -3.8},
    {"anno": 2020, "saldo_primario": -6.4, "saldo_complessivo": -8.3},
    {"anno": 2021, "saldo_primario": -3.0, "saldo_complessivo": -4.4},
    {"anno": 2022, "saldo_primario": -1.8, "saldo_complessivo": -4.1},
    {"anno": 2023, "saldo_primario": -2.7, "saldo_complessivo": -4.4},
    {"anno": 2024, "saldo_primario": 1.8, "saldo_complessivo": 0.3},
    {"anno": 2025, "saldo_primario": 1.6, "saldo_complessivo": None, "nota": "stima IMF/OECD"},
    {"anno": 2026, "saldo_primario": 1.7, "saldo_complessivo": None, "nota": "stima IMF"},
]

# ── Debito pubblico (% del PIL) ────────────────────────────────────────────
DEBITO_PIL = [
    {"anno": 2019, "debito_pil": 89.4},
    {"anno": 2020, "debito_pil": 102.8},
    {"anno": 2021, "debito_pil": 80.7},
    {"anno": 2022, "debito_pil": 84.7},
    {"anno": 2023, "debito_pil": 155.4},
    {"anno": 2024, "debito_pil": 91.5},
    {"anno": 2025, "debito_pil": 78.0, "nota": "stima IMF"},
    {"anno": 2026, "debito_pil": 68.0, "nota": "stima IMF"},
]

# ── Povertà: dati INDEC semestrali (% popolazione, ufficiali) ──────────────
# NOTA METODOLOGICA: l'INDEC pubblica dati semestrali. La UCA-ODSA pubblica
# stime trimestrali basate sulla stessa EPH ma con metodologia diversa.
# La UCA contesta la Canasta Básica Total INDEC (basata su ENGHo 2004/05)
# e stima che la povertà reale sia 4-5 pp più alta.
POVERTA_INDEC = [
    {"periodo": "2018-H2", "povertà": 32.0, "indigenza": None},
    {"periodo": "2019-H1", "povertà": 35.4, "indigenza": 7.7},
    {"periodo": "2019-H2", "povertà": 35.5, "indigenza": 8.0},
    {"periodo": "2020-H1", "povertà": 40.9, "indigenza": 10.5},
    {"periodo": "2020-H2", "povertà": 42.0, "indigenza": 10.5},
    {"periodo": "2021-H1", "povertà": 40.6, "indigenza": 10.7},
    {"periodo": "2021-H2", "povertà": 37.3, "indigenza": 8.2},
    {"periodo": "2022-H1", "povertà": 36.5, "indigenza": 8.8},
    {"periodo": "2022-H2", "povertà": 39.2, "indigenza": 8.1},
    {"periodo": "2023-H1", "povertà": 40.1, "indigenza": 9.3},
    {"periodo": "2023-H2", "povertà": 41.7, "indigenza": 11.9},
    {"periodo": "2024-H1", "povertà": 52.9, "indigenza": 18.1,
     "nota": "picco massimo dal 2003, 24.9 mln di persone"},
    {"periodo": "2024-H2", "povertà": 38.1, "indigenza": 8.2,
     "nota": "calo di 14.8 pp, il più forte in 20 anni"},
    {"periodo": "2025-H1", "povertà": 31.6, "indigenza": 6.9,
     "nota": "minimo dal H1-2018 (27.3%). UCA contesta: stima 35-37%"},
]

# ── Povertà: stime trimestrali UCA-ODSA (% popolazione) ───────────────────
# Fonte: Observatorio de la Deuda Social Argentina, Univ. Católica Argentina
# Basate su microdati EPH-INDEC ma con canasta aggiornata
POVERTA_UCA_TRIMESTRALE = [
    {"trimestre": "2023-Q4", "povertà": 45.2, "indigenza": 14.6},
    {"trimestre": "2024-Q1", "povertà": 54.9, "indigenza": 20.3,
     "nota": "tassi più alti dal 2004, shock svalutazione + inflazione"},
    {"trimestre": "2024-Q2", "povertà": 51.0, "indigenza": 15.8,
     "nota": "inizio decelerazione prezzi"},
    {"trimestre": "2024-Q3", "povertà": 49.9, "indigenza": 12.9,
     "nota": "23 mln di persone ancora in povertà"},
    {"trimestre": "2024-Q4", "povertà": None, "indigenza": None,
     "nota": "non ancora pubblicato al momento della raccolta dati"},
    {"trimestre": "2024-media", "povertà": 41.6, "indigenza": None,
     "nota": "media annua UCA per il 2024"},
]

# ── Povertà infantile (< 18 anni) ─────────────────────────────────────────
POVERTA_INFANTILE = {
    "fonte_primaria": "UCA-ODSA / UNICEF / INDEC",
    "nota_metodologica": (
        "I dati UCA includono dimensioni non monetarie. I dati INDEC "
        "sono puramente monetari (sotto la linea di povertà)."
    ),
    "serie": [
        {"periodo": "2023", "povertà_pct": 56.6, "indigenza_pct": None,
         "fonte": "UCA-ODSA"},
        {"periodo": "2024-H1", "povertà_pct": 67.3, "indigenza_pct": 30.6,
         "fonte": "UCA-ODSA",
         "nota": "record dal 2001, oltre 8 mln di minori"},
        {"periodo": "2024-H2", "povertà_pct": 52.7, "indigenza_pct": None,
         "fonte": "INDEC/UNICEF",
         "nota": "calo di 14 pp, 1.7 mln di minori usciti dalla povertà"},
        {"periodo": "2024-Q3", "povertà_pct": 65.5, "indigenza_pct": 19.2,
         "fonte": "UCA-ODSA"},
        {"periodo": "2025-H1", "povertà_pct": 45.4, "indigenza_pct": None,
         "fonte": "INDEC",
         "nota": "< 14 anni"},
    ],
    "disuguaglianze_H2_2024": {
        "capofamiglia_senza_primaria": 80.9,
        "capofamiglia_con_secondaria": 10.6,
        "lavoratori_informali": 68.4,
        "insediamenti_informali": 72.3,
        "madre_sola": 60.0,
        "buenos_aires_citta": 27.1,
        "concordia_massimo": 75.0,
    },
    "effetto_trasferimenti": {
        "nota": (
            "Senza AUH e Tarjeta Alimentar, l'indigenza infantile sarebbe "
            "stata 10 pp più alta nel H2-2024 (~1 mln di minori in più)"
        ),
        "auh_aumento_gen_2024": "100%",
        "auh_aumento_mar_2024": "27%",
        "auh_aumento_giu_2024": "41%",
    },
}

# ── Dibattito metodologico INDEC vs UCA ───────────────────────────────────
DIBATTITO_METODOLOGICO = {
    "questione": (
        "La Canasta Básica Total INDEC usa la struttura di spesa ENGHo "
        "2004/05. La UCA propone di usare ENGHo 2017/18, il che alza "
        "la CBT da ~$1.276.649 a ~$1.942.000 (pesos, giu 2025)."
    ),
    "differenza_stimata_pp": "4-5",
    "indec_H1_2025": 31.6,
    "uca_stima_H1_2025": "35-37",
    "cedlas_stima_ott24_mar25": 34.4,
    "utdt_stima_Q1_2025": 32.3,
    "nota": (
        "Come evidenziato nel dibattito italiano (ISPI/Seminerio/Economia "
        "Italia): i dati semestrali INDEC mostrano un aumento netto H2-2023 "
        "→ H1-2024, ma i dati trimestrali UCA mostrano che il picco è stato "
        "nel Q1-2024 con successiva discesa già dal Q2-2024. Entrambe le "
        "letture sono corrette ma raccontano storie diverse a seconda della "
        "granularità temporale scelta."
    ),
}

# ── Disoccupazione (%, INDEC) ──────────────────────────────────────────────
DISOCCUPAZIONE = [
    {"anno": 2019, "tasso": 9.8},
    {"anno": 2020, "tasso": 11.6},
    {"anno": 2021, "tasso": 8.7},
    {"anno": 2022, "tasso": 6.3},
    {"anno": 2023, "tasso": 6.1},
    {"anno": 2024, "tasso": 8.2},
    {"anno": 2025, "tasso": 7.0, "nota": "stima / Osservatorio CPI ~7%"},
]

# ── Mercato del lavoro: qualità e informalità ──────────────────────────────
# Fonte: INDEC (EPH Q2-2025), CIPPEC, CEPA, UCA-ODSA
MERCATO_LAVORO = {
    "disoccupazione_Q2_2025_pct": 7.6,
    "disoccupazione_Q1_2025_pct": 7.9,
    "tasso_occupazione_Q2_2025_pct": 44.5,
    "informalita": {
        "occupati_in_nero_pct": 43.2,
        "salariati_senza_contributi_pct": 38.0,
        "nota": (
            "L'informalità ha aiutato a contenere la disoccupazione ma "
            "a scapito della qualità del lavoro."
        ),
    },
    "cippec": {
        "occupazione_formale_privata": (
            "Stagnante da oltre un decennio: il numero di salariati registrati "
            "è rimasto pressoché costante, facendo scendere la loro quota "
            "nell'occupazione totale da quasi il 60% a meno del 50%."
        ),
        "monotributistas_aumento_dal_2010_pct": 64,
        "raccomandazione": (
            "Riforma del diritto del lavoro per ridurre costi contributivi "
            "e incertezza legale, favorendo la formalizzazione."
        ),
    },
    "cepa": {
        "nota": (
            "Conferma la riduzione della povertà ma ricorda che l'obsolescenza "
            "del paniere di consumo può far sottostimare la povertà effettiva."
        ),
    },
    "sintesi_esperto": (
        "I costi sociali NON sono aumentati in termini di povertà ufficiale. "
        "Tuttavia l'austerità non ha risolto i problemi strutturali: la qualità "
        "dell'occupazione è bassa, l'informalità resta molto elevata e il sistema "
        "di misurazione ufficiale potrebbe attenuare l'entità della povertà."
    ),
}

# ── Riserve internazionali BCRA (mld USD, lorde) ──────────────────────────
RISERVE = [
    {"anno": 2019, "riserve_lorde": 44.8},
    {"anno": 2020, "riserve_lorde": 39.4},
    {"anno": 2021, "riserve_lorde": 39.7},
    {"anno": 2022, "riserve_lorde": 44.6},
    {"anno": 2023, "riserve_lorde": 23.1},
    {"anno": 2024, "riserve_lorde": 29.6},
    {"anno": 2025, "riserve_lorde": 32.3, "nota": "dic 2025"},
]

# ── Riserve nette e crisi settembre 2025 ─────────────────────────────────
RISERVE_NETTE_E_CRISI = {
    "riserve_lorde_set_2025_mld": 40.374,
    "nota_set_2025": "Appena 387 mln in più rispetto ad agosto; struttura fragile",
    "riserve_liquide_jp_morgan_mld": 18.5,
    "riserve_nette_mld": -2.4,
    "nota_riserve_nette": (
        "Escludendo riserve obbligatorie in valuta e swap, le riserve "
        "nette BCRA risultano negative per ~2.4 mld USD (JP Morgan)."
    ),
    "crisi_settembre_2025": {
        "vendita_record_19_set": 678,
        "unita": "milioni USD",
        "nota": "più grande vendita giornaliera BCRA in quasi 6 anni",
        "vendita_3_sedute_mld": 1.0,
        "trigger": (
            "Sconfitta LLA alle elezioni provinciali Buenos Aires (7 set), "
            "sell-off bond (rendimenti 10y >17%), peso -7% in un giorno"
        ),
    },
    "obiettivo_acquisti_2026_mld": 10.0,
    "nota_obiettivo": "BCRA si impegna ad acquistare $10 mld di riserve entro fine 2026",
}

# ── Swap USA-Argentina (Tesoro USA / ESF) ────────────────────────────────
SWAP_USA = {
    "importo_mld_usd": 20.0,
    "meccanismo": "Exchange Stabilization Fund (ESF) del Tesoro USA",
    "annuncio": "22 settembre 2025 (Bessent su X)",
    "firma": "20 ottobre 2025",
    "cancellazione_operazioni": "dicembre 2025",
    "funzionamento": (
        "Il Tesoro USA acquista fino a $20 mld di pesos dal BCRA in cambio "
        "di dollari. L'Argentina dovrà restituire i dollari con interessi."
    ),
    "utilizzo_a_fine_ott_2025_mld": 2.5,
    "impatto_riserve": (
        "Aumenta le riserve lorde ma NON le riserve nette, essendo un "
        "passivo. Senza swap e obblighi, le riserve nette restano negative."
    ),
    "fondi_aggiuntivi": {
        "ipotesi_iniziale_mld": 20.0,
        "ridotto_a_mld": 5.0,
        "tipo": "repo agreement con JPMorgan, Citigroup, Bank of America",
        "nota": "Ridimensionato da $20 a ~$5 mld; banche attendevano garanzie collaterali",
    },
    "contesto_politico": {
        "obiettivo_pre_elezioni": (
            "Contenere il dollaro prima delle legislative del 26 ottobre 2025"
        ),
        "opposizione_argentina_pct": "55-60%",
        "nota": "Sondaggi: 55-60% degli argentini contrari all'assistenza USA",
        "opposizione_usa": (
            "8 senatori Dem (incl. Warren) hanno proposto legge per bloccare il bailout. "
            "Critiche per impatto su agricoltori USA (export soia argentina a Cina)."
        ),
    },
    "dimensione_geopolitica": {
        "swap_cina_preesistente_mld": 18.0,
        "nota": (
            "L'admin USA vedeva rischio di rafforzamento legami Argentina-Cina. "
            "Milei ha negato che USA abbiano chiesto di eliminare lo swap cinese."
        ),
    },
    "confronto_storico": (
        "Più grande bailout bilaterale USA dal salvataggio del Messico "
        "di Clinton nel 1995."
    ),
    "rischio_krugman": (
        "Krugman e altri: effetti perlopiù temporanei, rinviano il momento "
        "in cui, esaurite le riserve, l'Argentina dovrà svalutare."
    ),
    "uso_per_debito_2026": (
        "Milei ha dichiarato che se il rischio paese è troppo alto per "
        "emettere sui mercati, userà la linea di swap per i pagamenti 2026 "
        "— 'prendere debito per pagare debito'."
    ),
}

# ── Regime di cambio e tasso reale multilaterale (ITCRM) ─────────────────
REGIME_CAMBIO = {
    "fasi": [
        {
            "periodo": "dic 2023",
            "tipo": "svalutazione iniziale",
            "dettaglio": "da 366.5 a 800 ARS/USD (-54%)",
        },
        {
            "periodo": "gen 2024 – gen 2025",
            "tipo": "crawling peg 2% mensile",
            "dettaglio": (
                "Svalutazione nominale 2%/mese, ma inflazione molto superiore "
                "→ forte apprezzamento reale del peso"
            ),
        },
        {
            "periodo": "feb 2025 – apr 2025",
            "tipo": "crawling peg 1% mensile",
            "dettaglio": "Rallentamento del ritmo di svalutazione, apprezzamento reale accelera",
        },
        {
            "periodo": "apr 2025",
            "tipo": "banda gestita FMI",
            "dettaglio": "Fluttuazione libera tra 1000 e 1400 ARS/USD, banda ampia 40%",
        },
        {
            "periodo": "gen 2026",
            "tipo": "banda indicizzata all'inflazione",
            "dettaglio": (
                "Banda si espande mensilmente al tasso di inflazione INDEC (t-2). "
                "Gen 2026: +2.5% (inflaz. nov 2025). Feb 2026: +2.8% (inflaz. dic 2025)."
            ),
        },
    ],
    "itcrm": {
        "descrizione": (
            "Indice del Tasso di Cambio Reale Multilaterale BCRA. "
            "Misura il prezzo relativo dei beni argentini vs 12 partner commerciali. "
            "Ponderazioni: Brasile 32%, Cina 16%, Eurozona 19%, USA 12%."
        ),
        "reer_dic_2025": 58.6,
        "reer_nov_2025": 59.4,
        "reer_massimo_storico": {"valore": 123.1, "data": "giugno 2002"},
        "reer_minimo_storico": {"valore": 41.1, "data": "ottobre 2001"},
        "interpretazione": (
            "Calo dell'indice = apprezzamento reale del peso = export più cari, "
            "import più economici. Valore dic 2025 (58.6) indica un peso "
            "significativamente apprezzato in termini reali, vicino a livelli "
            "storicamente associati a crisi di cambio."
        ),
    },
    "paradosso": (
        "Peso forte → bene per inflazione (import economici, ancoraggio aspettative). "
        "Ma male per competitività, export, riserve. Ogni ondata di tensione "
        "riapre il fronte valutario e frena l'aggiustamento ordinato."
    ),
}

# ── Bilancia commerciale (mld USD) ────────────────────────────────────────
BILANCIA_COMMERCIALE = [
    {"anno": 2019, "export": 65.1, "import": 49.1, "saldo": 16.0},
    {"anno": 2020, "export": 54.9, "import": 42.4, "saldo": 12.5},
    {"anno": 2021, "export": 77.9, "import": 63.2, "saldo": 14.7},
    {"anno": 2022, "export": 88.4, "import": 81.5, "saldo": 6.9},
    {"anno": 2023, "export": 66.8, "import": 73.7, "saldo": -6.9},
    {"anno": 2024, "export": 79.7, "import": 60.8, "saldo": 18.9},
    {"anno": 2025, "export": None, "import": None, "saldo": 11.3, "nota": "gen-dic 2025"},
]

# ── Conto corrente (mld USD) ──────────────────────────────────────────────
CONTO_CORRENTE = [
    {"trimestre": "2024-Q1", "saldo": 0.18},
    {"trimestre": "2024-Q2", "saldo": 3.73},
    {"trimestre": "2024-Q3", "saldo": 0.89},
    {"trimestre": "2025-Q1", "saldo": -5.19},
    {"trimestre": "2025-Q2", "saldo": -3.02},
    {"trimestre": "2025-Q3", "saldo": -1.58},
]

# ── Produzione industriale (var. % a/a) ───────────────────────────────────
PRODUZIONE_INDUSTRIALE = [
    {"anno": 2022, "var_aa": 4.1},
    {"anno": 2023, "var_aa": -2.3},
    {"anno": 2024, "var_aa": -9.4},
    {"anno": 2025, "var_aa": 1.6},
]

# ── Salari reali (indice, ott 2023 = 100, settore privato formale) ────────
SALARI_REALI = [
    {"mese": "2023-10", "indice": 100.0},
    {"mese": "2023-12", "indice": 85.0, "nota": "shock svalutazione dic 2023"},
    {"mese": "2024-06", "indice": 89.0},
    {"mese": "2024-12", "indice": 101.5, "nota": "salari nominali +145.5% vs inflaz. 117.8%"},
    {"mese": "2025-06", "indice": 106.0, "nota": "sopra livelli pre-crisi"},
]

# ── Country risk EMBI+ (punti base, JP Morgan) ───────────────────────────
EMBI = [
    {"data": "2024-01", "spread_bp": 1907},
    {"data": "2024-06", "spread_bp": 1400},
    {"data": "2024-10", "spread_bp": 1044},
    {"data": "2025-03", "spread_bp": 780},
    {"data": "2025-04", "spread_bp": 1000, "nota": "pre-accordo IMF apr 2025"},
    {"data": "2025-09", "spread_bp": 1500, "nota": "dopo elezioni provinciali Buenos Aires"},
    {"data": "2025-10", "spread_bp": 1050},
    {"data": "2025-12", "spread_bp": 550, "nota": "post-vittoria legislative"},
    {"data": "2026-01", "spread_bp": 650},
    {"data": "2026-03", "spread_bp": 978, "nota": "volatilità globale"},
]

# ── Tasso di cambio USD/ARS (fine periodo, ufficiale) ────────────────────
CAMBIO = [
    {"data": "2023-11", "usd_ars": 366},
    {"data": "2023-12", "usd_ars": 808, "nota": "svalutazione 118%"},
    {"data": "2024-06", "usd_ars": 912},
    {"data": "2024-12", "usd_ars": 1032},
    {"data": "2025-04", "usd_ars": 1190, "nota": "passaggio a banda gestita"},
    {"data": "2025-12", "usd_ars": 1065},
]

# ── Rating sovrano ────────────────────────────────────────────────────────
RATING = [
    {
        "agenzia": "S&P Global",
        "rating_lc": "CCC+/C",
        "data": "dic 2025",
        "nota": "upgrade da SD/SD",
    },
]

# ── Programma IMF ─────────────────────────────────────────────────────────
IMF_PROGRAMMA = {
    "tipo": "Extended Fund Facility (EFF)",
    "durata_anni": 4,
    "importo_totale_mld_usd": 20.0,
    "disbursamento_iniziale_mld_usd": 12.0,
    "data_approvazione": "aprile 2025",
    "obiettivo_surplus_primario_2025": 1.3,
}

# ── Scadenze debito estero 2026 ──────────────────────────────────────────
DEBITO_ESTERO_2026 = {
    "pagamenti_totali_mld_usd": 15.0,
    "di_cui_imf_mld_usd": 4.1,
    "pil_pct": 2.3,
    "nota": "principale rischio per sostenibilità riserve",
}

# ── Elezioni legislative ott 2025 ────────────────────────────────────────
ELEZIONI_2025 = {
    "data": "ottobre 2025",
    "la_libertad_avanza_pct": 41.0,
    "nota": "raddoppio rappresentanza parlamentare",
}

# ── KPI di sintesi ────────────────────────────────────────────────────────
KPI = {
    "pil_2024": -1.3,
    "pil_2025_stima_oecd": 5.2,
    "inflazione_2023_dic": 211.4,
    "inflazione_2024_dic": 117.8,
    "inflazione_2025_giu_mm": 1.5,
    "inflazione_2025_set_mm": 2.1,
    "inflazione_annua_set_2025": 31.8,
    "tasso_policy_bcra_feb_2025": 29.0,
    "spesa_primaria_taglio_aa_pct": -30,
    "surplus_primario_2024_pil": 1.8,
    "surplus_complessivo_2024_pil": 0.3,
    "debito_pil_2024": 91.5,
    "debito_pil_2023": 155.4,
    "poverta_indec_picco_2024h1": 52.9,
    "indigenza_indec_picco_2024h1": 18.1,
    "poverta_indec_2024h2": 38.1,
    "poverta_indec_2025h1": 31.6,
    "poverta_uca_picco_Q1_2024": 54.9,
    "indigenza_uca_picco_Q1_2024": 20.3,
    "poverta_uca_Q3_2024": 49.9,
    "poverta_infantile_picco_2024h1": 67.3,
    "poverta_infantile_2024h2": 52.7,
    "disoccupazione_2024": 8.2,
    "disoccupazione_Q2_2025": 7.6,
    "tasso_occupazione_Q2_2025": 44.5,
    "informalita_pct": 43.2,
    "riserve_lorde_dic2025_mld": 32.3,
    "riserve_nette_mld": -2.4,
    "swap_usa_mld": 20.0,
    "swap_cina_preesistente_mld": 18.0,
    "vendita_record_bcra_19set_mln": 678,
    "reer_dic_2025": 58.6,
    "trade_surplus_2024_mld": 18.9,
    "trade_surplus_2025_mld": 11.3,
    "embi_dic2025_bp": 550,
    "imf_eff_mld": 20.0,
    "bond_bonar2029_mld": 1.0,
    "fdi_2025_mld": -1.52,
}


def main():
    out_path = Path("docs/argentina_dashboard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "titolo": "L'Argentina di Milei: analisi macroeconomica",
            "descrizione": (
                "Dati ispirati all'articolo dell'Osservatorio CPI (Unicatt) "
                "'L'Argentina di Milei, due anni dopo', arricchiti con fonti "
                "IMF, OECD, World Bank, INDEC, BCRA, Trading Economics, "
                "UCA-ODSA, CIPPEC, CEPA, lavoce.info. "
                "Verificati da revisione esperta indipendente (mar 2026)."
            ),
            "fonti": [
                "Osservatorio CPI – Università Cattolica",
                "IMF – World Economic Outlook / Country Report",
                "OECD – Economic Surveys: Argentina 2025",
                "INDEC – Instituto Nacional de Estadística y Censos",
                "BCRA – Banco Central de la República Argentina",
                "Trading Economics",
                "World Bank",
                "Allianz Trade – Country Risk Report",
                "BBVA Research – Argentina Economic Outlook",
                "JP Morgan – EMBI+",
                "UCA-ODSA – Observatorio de la Deuda Social Argentina",
                "UNICEF Argentina",
                "ISPI – Istituto per gli Studi di Politica Internazionale",
                "UTDT – Universidad Torcuato Di Tella",
                "CEDLAS – Centro de Estudios Distributivos, Laborales y Sociales",
                "US Treasury – Exchange Stabilization Fund",
                "CNBC / Fortune / Axios – Reporting swap USA-Argentina",
                "CFR – Council on Foreign Relations",
                "PIIE – Peterson Institute for International Economics",
                "Osservatorio CPI – Unicatt (articolo originale)",
            ],
            "ultimo_aggiornamento": "2026-03-08",
        },
        "kpi": KPI,
        "pil": {
            "trimestrale": PIL_TRIMESTRALE,
            "annuale": PIL_ANNUALE,
            "stime_confronto": STIME_PIL,
        },
        "inflazione": {
            "annuale": INFLAZIONE_ANNUALE,
            "mensile": INFLAZIONE_MENSILE,
            "stime": STIME_INFLAZIONE,
        },
        "tasso_policy_bcra": TASSO_POLICY_BCRA,
        "spesa_primaria_imf": SPESA_PRIMARIA,
        "inflazione_annua_set_2025": INFLAZIONE_ANNUA_SET_2025,
        "bilancio_fiscale": BILANCIO_FISCALE,
        "debito_pubblico": DEBITO_PIL,
        "poverta": {
            "indec_semestrale": POVERTA_INDEC,
            "uca_trimestrale": POVERTA_UCA_TRIMESTRALE,
            "infantile": POVERTA_INFANTILE,
            "dibattito_metodologico": DIBATTITO_METODOLOGICO,
        },
        "disoccupazione": DISOCCUPAZIONE,
        "mercato_lavoro": MERCATO_LAVORO,
        "riserve_internazionali": {
            "serie_lorde": RISERVE,
            "nette_e_crisi": RISERVE_NETTE_E_CRISI,
        },
        "swap_usa": SWAP_USA,
        "regime_cambio_e_itcrm": REGIME_CAMBIO,
        "bilancia_commerciale": BILANCIA_COMMERCIALE,
        "conto_corrente": CONTO_CORRENTE,
        "produzione_industriale": PRODUZIONE_INDUSTRIALE,
        "salari_reali": SALARI_REALI,
        "country_risk_embi": EMBI,
        "cambio_usd_ars": CAMBIO,
        "rating_sovrano": RATING,
        "imf_programma": IMF_PROGRAMMA,
        "debito_estero_scadenze_2026": DEBITO_ESTERO_2026,
        "elezioni_2025": ELEZIONI_2025,
        "valutazione_complessiva": {
            "disinflazione": (
                "Rapida e reale: dal 25% mensile (dic 2023) a ~2% (2025). "
                "Deriva da severo riequilibrio fiscale (spesa primaria -30%, IMF) "
                "e tassi reali positivi. Tuttavia l'inflazione annua (31.8% a "
                "set 2025) resta alta per standard internazionali."
            ),
            "costi_sociali": (
                "CORREZIONE: i dati CONTRADDICONO l'idea di 'costi sociali in crescita'. "
                "La povertà è SCESA dal 52.9% (H1-2024) al 31.6% (H1-2025). "
                "L'indigenza è calata dal 18.1% al 6.9%. I trasferimenti mirati "
                "(AUH +100%, Tarjeta Alimentar) hanno attenuato l'impatto dell'austerità. "
                "Le ombre riguardano la QUALITÀ del lavoro e la fragilità macro, "
                "NON l'aumento della povertà."
            ),
            "fragilita_strutturali": (
                "Bassa produttività, elevata informalità (43.2% in nero), "
                "occupazione formale stagnante da un decennio, scarsa credibilità "
                "del regime di cambio, dipendenza da sostegni esterni (swap USA $20 mld). "
                "Senza ricostruire le riserve e adottare un regime di cambio credibile, "
                "il rischio di nuove tensioni resta elevato."
            ),
            "conclusione": (
                "L'intervento di Milei ha ottenuto un risultato significativo sul fronte "
                "dell'inflazione a costo di una contrazione brusca della spesa pubblica. "
                "Ciò NON ha provocato un aumento della povertà, ma NON ha neppure risolto "
                "le debolezze strutturali dell'economia."
            ),
        },
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ Scritto {out_path.resolve()}")


if __name__ == "__main__":
    main()
