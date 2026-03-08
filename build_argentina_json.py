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
]

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

# ── Tasso di povertà (% popolazione, INDEC) ────────────────────────────────
POVERTA = [
    {"periodo": "2018-H2", "tasso": 32.0},
    {"periodo": "2019-H1", "tasso": 35.4},
    {"periodo": "2019-H2", "tasso": 35.5},
    {"periodo": "2020-H1", "tasso": 40.9},
    {"periodo": "2020-H2", "tasso": 42.0},
    {"periodo": "2021-H1", "tasso": 40.6},
    {"periodo": "2021-H2", "tasso": 37.3},
    {"periodo": "2022-H1", "tasso": 36.5},
    {"periodo": "2022-H2", "tasso": 39.2},
    {"periodo": "2023-H1", "tasso": 40.1},
    {"periodo": "2023-H2", "tasso": 41.7},
    {"periodo": "2024-H1", "tasso": 52.9},
    {"periodo": "2024-H2", "tasso": 38.1},
    {"periodo": "2025-H1", "tasso": 32.0},
    {"periodo": "2025-Q3", "tasso": 27.0, "nota": "Osservatorio CPI"},
]

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
    "surplus_primario_2024_pil": 1.8,
    "surplus_complessivo_2024_pil": 0.3,
    "debito_pil_2024": 91.5,
    "debito_pil_2023": 155.4,
    "poverta_picco_2024h1": 52.9,
    "poverta_2025q3": 27.0,
    "disoccupazione_2024": 8.2,
    "riserve_lorde_dic2025_mld": 32.3,
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
                "IMF, OECD, World Bank, INDEC, BCRA, Trading Economics."
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
        "bilancio_fiscale": BILANCIO_FISCALE,
        "debito_pubblico": DEBITO_PIL,
        "poverta": POVERTA,
        "disoccupazione": DISOCCUPAZIONE,
        "riserve_internazionali": RISERVE,
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
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ Scritto {out_path.resolve()}")


if __name__ == "__main__":
    main()
