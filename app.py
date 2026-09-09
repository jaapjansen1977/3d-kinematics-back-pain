from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis
from web_pipeline import run_analysis


st.set_page_config(
    page_title="Bukbewegingsanalyse",
    page_icon="↘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8f7; }
      [data-testid="stHeader"] { background: rgba(246,248,247,.92); }
      .hero { padding: 1.2rem 1.35rem; border-radius: 18px; color: white;
              background: linear-gradient(120deg,#153f3a,#23796d); margin-bottom: 1rem; }
      .hero h1 { margin: 0 0 .25rem 0; font-size: 2rem; }
      .hero p { margin: 0; color: #e7f4f1; }
      .notice { padding: .75rem 1rem; border-left: 4px solid #d69e2e;
                background: #fff8e6; border-radius: 8px; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #dde5e2;
                padding: .8rem 1rem; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


LABELS = {
    "repetition": "Herhaling",
    "cycle_duration_s": "Cyclustijd (s)",
    "flexion_duration_s": "Buigduur (s)",
    "extension_duration_s": "Strekduur (s)",
    "bottom_dwell_s": "Verblijftijd onderin (s)",
    "knee_flex_deg_L_robust_rom_deg": "Knie-ROM links (°)",
    "knee_flex_deg_R_robust_rom_deg": "Knie-ROM rechts (°)",
    "hip_flex_deg_L_robust_rom_deg": "Heup-ROM links (°)",
    "hip_flex_deg_R_robust_rom_deg": "Heup-ROM rechts (°)",
    "pelvis_flex_deg_robust_rom_deg": "Bekken-ROM in ruimte (°)",
    "lumbar_flex_rel_deg_robust_rom_deg": "Rug-ROM t.o.v. bekken (°)",
    "hip_rom_asymmetry_L_minus_R_deg": "Heupasymmetrie L−R (°)",
    "knee_rom_asymmetry_L_minus_R_deg": "Knieasymmetrie L−R (°)",
    "pelvis_onset_delay_vs_lumbar_s": "Startvertraging bekken t.o.v. rug (s)",
    "lumbar_contribution_flexion_pct": "Bijdrage rug aan flexie (%)",
    "pelvis_contribution_flexion_pct": "Bijdrage bekken aan flexie (%)",
    "trunk_lat_rel_deg_robust_rom_deg": "Lateroflexie romp t.o.v. bekken (°)",
    "trunk_rot_deg_robust_rom_deg": "Axiale romprotatie (°)",
    "pelvis_abd_deg_robust_rom_deg": "Laterale bekkenbeweging (°)",
    "signal": "Signaal",
    "mean_pointwise_sd_deg": "Gemiddelde puntgewijze SD (°)",
    "max_pointwise_sd_deg": "Maximale puntgewijze SD (°)",
    "mean_rmse_to_mean_deg": "Gemiddelde RMSE tot gemiddelde (°)",
    "median_consecutive_rmse_deg": "Mediaan RMSE opeenvolgende cycli (°)",
    "mean_correlation_with_mean_waveform": "Gemiddelde golfvormcorrelatie",
    "mean_vector_coding_variability_deg": "Vector-codingvariabiliteit (°)",
    "vector_coding_valid_points_pct": "Geldige vector-codingpunten (%)",
    "mean_absolute_crp_deg": "Gemiddelde absolute CRP (°)",
    "mean_crp_variability_deg": "CRP-variabiliteit (°)",
    "metric": "Uitkomstmaat",
    "value": "Waarde",
    "unit": "Eenheid",
    "status": "Status",
    "parameters": "Parameters",
}


EXACT_HELP = {
    "repetition": "Volgnummer van de automatisch gesegmenteerde bukbeweging.",
    "cycle_duration_s": "Tijd van het begin van één bukcyclus tot het einde ervan. Lager betekent sneller uitgevoerd.",
    "flexion_duration_s": "Tijd van het begin van de cyclus tot het diepste buigpunt.",
    "extension_duration_s": "Tijd van het diepste buigpunt tot terugkeer naar de uitgangspositie.",
    "bottom_dwell_s": "Tijd binnen de bovenste 5% van de heupflexie. Dit benadert hoe lang iemand onderin blijft.",
    "knee_flex_deg_L_robust_rom_deg": "Robuuste knie-range of motion links: 95e minus 5e percentiel binnen één herhaling.",
    "knee_flex_deg_R_robust_rom_deg": "Robuuste knie-range of motion rechts: 95e minus 5e percentiel binnen één herhaling.",
    "hip_flex_deg_L_robust_rom_deg": "Robuuste heup-range of motion links per herhaling.",
    "hip_flex_deg_R_robust_rom_deg": "Robuuste heup-range of motion rechts per herhaling.",
    "pelvis_flex_deg_robust_rom_deg": "Robuuste sagittale bekkenexcursie ten opzichte van de globale ruimte.",
    "lumbar_flex_rel_deg_robust_rom_deg": "Robuuste rugexcursie ten opzichte van het bekken. Hier planair benaderd als rompflexie minus bekkenflexie.",
    "hip_rom_asymmetry_L_minus_R_deg": "Heup-ROM links minus rechts. Het teken geeft de richting aan; de absolute waarde de grootte.",
    "knee_rom_asymmetry_L_minus_R_deg": "Knie-ROM links minus rechts. Het teken geeft de richting aan; de absolute waarde de grootte.",
    "pelvis_onset_delay_vs_lumbar_s": "Verschil in bewegingsstart. Positief betekent dat het bekken later start dan de rug.",
    "lumbar_contribution_flexion_pct": "Aandeel van de relatieve rugexcursie in de som van rug- en bekkenexcursie tijdens de buigfase.",
    "pelvis_contribution_flexion_pct": "Aandeel van de bekkenexcursie in de som van rug- en bekkenexcursie tijdens de buigfase.",
    "trunk_lat_rel_deg_robust_rom_deg": "Zijwaartse rompbeweging ten opzichte van het bekken; een beweging buiten het primaire sagittale vlak.",
    "trunk_rot_deg_robust_rom_deg": "Axiale rotatie van de romp in de globale ruimte. Relatieve rotatie is niet beschikbaar zonder bekkenrotatie.",
    "pelvis_abd_deg_robust_rom_deg": "Laterale bekkenkanteling/-beweging in de globale ruimte.",
    "mean_pointwise_sd_deg": "Gemiddelde SD over 101 tijdgenormaliseerde punten. Lager betekent meer overeenkomst tussen herhalingen.",
    "max_pointwise_sd_deg": "Grootste SD op enig moment in de tijdgenormaliseerde cyclus.",
    "mean_rmse_to_mean_deg": "Gemiddelde afstand van iedere golfvorm tot de gemiddelde golfvorm. Lager betekent stabieler.",
    "median_consecutive_rmse_deg": "Mediaan verschil tussen telkens twee opeenvolgende herhalingen. Lager betekent meer herhaalbaarheid.",
    "mean_correlation_with_mean_waveform": "Vormovereenkomst met de gemiddelde golfvorm. Dichter bij 1 betekent sterker gelijkende vorm, ongeacht een deel van het amplitudeverschil.",
    "mean_vector_coding_variability_deg": "Circulaire variabiliteit van de heup-rug coupling angle. Lager betekent consistentere interjoint-coupling.",
    "vector_coding_valid_points_pct": "Aandeel punten waar de coupling angle betrouwbaar genoeg bepaald kon worden; zeer lage gezamenlijke snelheid wordt gemaskeerd.",
    "mean_absolute_crp_deg": "Gemiddelde absolute continuous relative phase tussen heup en rug. Interpretatie is gevoelig voor normalisatie en keerpunten.",
    "mean_crp_variability_deg": "Circulaire spreiding van CRP over herhalingen. Lager betekent stabielere fasekoppeling.",
    "sample_entropy_velocity": "Sample entropy van de hoeksnelheid. Hoger duidt op minder regelmaat, maar de waarde hangt sterk af van parameters en datalengte.",
    "SPARC_angular_speed_mean": "Spectral Arc Length van de absolute hoeksnelheid. Minder negatief/hoger wijst op een vloeiender profiel bij gelijke meet- en filterinstellingen.",
    "spectral_high_fraction": "Aandeel spectraal vermogen in 2–10 Hz ten opzichte van 0,1–10 Hz. Hoger kan wijzen op meer snelle correcties of meetruis.",
    "DFA_alpha": "Detrended fluctuation analysis van fluctuaties over opeenvolgende cycli. Wordt bewust niet berekend bij minder dan 600 cycli.",
    "approximate_entropy": "Niet berekend: Sample Entropy is gekozen omdat Approximate Entropy sterker door self-matches wordt beïnvloed.",
    "value": "Berekende waarde van de betreffende uitkomstmaat.",
    "unit": "Eenheid waarin de uitkomst is weergegeven.",
    "status": "Methodologische duiding, bijvoorbeeld exploratief of niet berekend.",
    "parameters": "Instellingen die nodig zijn om de berekening te reproduceren.",
}


def help_for(column: str) -> str:
    if column in EXACT_HELP:
        return EXACT_HELP[column]
    if column.endswith("_robust_rom_deg"):
        return "Robuuste range of motion: verschil tussen het 95e en 5e percentiel binnen de herhaling."
    if column.endswith("_rom_deg"):
        return "Volledige range of motion: maximum minus minimum binnen de herhaling."
    if column.endswith("_rms_from_start_deg"):
        return "RMS van de afwijking ten opzichte van de starthoek binnen de herhaling."
    if column.endswith("_max_abs_from_start_deg"):
        return "Grootste absolute afwijking ten opzichte van de starthoek."
    if column.endswith("_peak_flex_velocity_deg_s"):
        return "Hoogste hoeksnelheid tijdens de buigfase, in graden per seconde."
    if column.endswith("_peak_extension_velocity_deg_s"):
        return "Hoogste strek-/terugkeersnelheid, met teken volgens de hoekconventie."
    if column.startswith("vector_coding_") and column.endswith("_pct"):
        return "Percentage geldige punten dat in deze vector-codingcoördinatieklasse valt."
    return "Berekende kinematische uitkomst. Zie de methode-tab voor definitie, eenheid en beperkingen."


def display_name(column: str) -> str:
    return LABELS.get(column, column.replace("_", " ").replace("deg", "°").replace("pct", "%").capitalize())


def dataframe_config(df: pd.DataFrame) -> dict:
    config = {}
    for col in df.columns:
        label = display_name(col)
        help_text = help_for(col)
        if pd.api.types.is_numeric_dtype(df[col]):
            fmt = "%d" if col in {"repetition", "start_frame", "peak_frame", "end_frame"} else "%.3f"
            config[col] = st.column_config.NumberColumn(label, help=help_text, format=fmt)
        else:
            config[col] = st.column_config.TextColumn(label, help=help_text)
    return config


def metric_value(value: float, unit: str = "", decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "Niet beschikbaar"
    return f"{value:.{decimals}f}{unit}"


def cycle_plot(normalized: pd.DataFrame, signals: list[tuple[str, str, str]]) -> go.Figure:
    fig = go.Figure()
    for key, label, color in signals:
        pivot = normalized.pivot(index="cycle_pct", columns="repetition", values=key)
        mean = pivot.mean(axis=1)
        sd = pivot.std(axis=1)
        x = pivot.index.to_numpy()
        fig.add_trace(go.Scatter(x=x, y=mean + sd, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=mean - sd, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=color.replace("1)", ".14)"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=label, line=dict(color=color.replace("rgba", "rgb").replace(",1)", ")"), width=3)))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Bukcyclus (%)", yaxis_title="Hoek (°)")
    return fig


def line_by_repetition(df: pd.DataFrame, columns: list[tuple[str, str, str]], y_title: str) -> go.Figure:
    fig = go.Figure()
    for key, label, color in columns:
        fig.add_trace(go.Scatter(x=df["repetition"], y=df[key], mode="lines+markers", name=label, line=dict(color=color, width=2)))
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=20, r=20, t=25, b=20), xaxis_title="Herhaling", yaxis_title=y_title)
    return fig


st.markdown(
    """<div class="hero"><h1>Bukbewegingsanalyse</h1>
    <p>Klinische kinematica, herhaalbaarheid en interjoint-coördinatie uit een Excelbestand.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analyse-instellingen")
    uploaded = st.file_uploader(
        "Excelbestand met kinematica",
        type=["xlsx"],
        help="Het bestand moet minimaal de verplichte kolommen uit het voorbeeld bevatten. De upload wordt niet blijvend opgeslagen.",
    )
    sheet = st.text_input("Werkblad", value="angles", help="Naam van het werkblad met de kinematische tijdreeks.")
    expected_repetitions = st.number_input("Verwacht aantal herhalingen", 3, 1000, 20, help="De analyse stopt wanneer het gedetecteerde aantal hiervan afwijkt, zodat segmentatiefouten zichtbaar blijven.")
    lowpass_hz = st.slider("Low-passfilter (Hz)", 2.0, 12.0, 6.0, 0.5, help="Afkapfrequentie van het nul-fase Butterworthfilter. Houd deze instelling gelijk bij vergelijkingen.")
    analyze_clicked = st.button("Analyse uitvoeren", type="primary", use_container_width=True, disabled=uploaded is None)
    st.caption("Publieke demo: gebruik uitsluitend fictieve of volledig geanonimiseerde data.")

if uploaded is None:
    st.info("Upload links een `.xlsx`-bestand en start de analyse. Het meegeleverde voorbeeldbestand staat in de GitHub-repository.")
    st.markdown("#### Benodigde kolommen")
    st.code(", ".join(analysis.REQUIRED_COLUMNS), language=None)
    st.stop()

if analyze_clicked:
    try:
        with st.spinner("Herhalingen segmenteren en uitkomstmaten berekenen…"):
            st.session_state["analysis_result"] = run_analysis(
                uploaded.getvalue(), uploaded.name, sheet, int(expected_repetitions), float(lowpass_hz)
            )
            st.session_state["analysis_file"] = uploaded.name
    except Exception as exc:
        st.session_state.pop("analysis_result", None)
        st.error(f"Analyse niet uitgevoerd: {exc}")

if "analysis_result" not in st.session_state:
    st.warning("Klik op ‘Analyse uitvoeren’ om dit bestand te verwerken.")
    st.stop()

result = st.session_state["analysis_result"]
qc = result["qc"]
tables = result["tables"]
reps = tables["repetition_metrics"]
wave = tables["waveform_metrics"].set_index("signal")
coupling = tables["coupling_summary"].iloc[0]

if st.session_state.get("analysis_file") != uploaded.name:
    st.warning("Er staat nog een resultaat van een ander bestand. Klik opnieuw op ‘Analyse uitvoeren’.")

st.success(f"Analyse voltooid: {qc['detected_repetitions']} herhalingen gevonden.")
if qc["warnings"]:
    for warning in qc["warnings"]:
        st.warning(warning)

top_primary = st.columns(3)
top_primary[0].metric("Herhalingen", f"{qc['detected_repetitions']}", help="Aantal automatisch gedetecteerde en geanalyseerde bukcycli.", border=True)
top_primary[1].metric("Cyclustijd", metric_value(reps["cycle_duration_s"].mean(), " s"), help=EXACT_HELP["cycle_duration_s"], border=True)
top_primary[2].metric("Heup-ROM", metric_value(0.5 * (reps["hip_flex_deg_L_robust_rom_deg"].mean() + reps["hip_flex_deg_R_robust_rom_deg"].mean()), "°"), help="Gemiddelde robuuste ROM van linker- en rechterheup over alle herhalingen.", border=True)

top_secondary = st.columns(3)
top_secondary[0].metric("Rug-ROM", metric_value(reps["lumbar_flex_rel_deg_robust_rom_deg"].mean(), "°"), help=EXACT_HELP["lumbar_flex_rel_deg_robust_rom_deg"], border=True)
top_secondary[1].metric("Heupstabiliteit", metric_value(wave.loc["hip_flex_mean_deg", "mean_pointwise_sd_deg"], "°"), help=EXACT_HELP["mean_pointwise_sd_deg"], border=True)
top_secondary[2].metric("Couplingvariabiliteit", metric_value(coupling["mean_vector_coding_variability_deg"], "°"), help=EXACT_HELP["mean_vector_coding_variability_deg"], border=True)

tabs = st.tabs(["Overzicht", "ROM & tempo", "Stabiliteit & coupling", "Deviaties & regulariteit", "Methode & export"])

with tabs[0]:
    st.subheader("Gemiddelde bewegingsprofielen")
    st.plotly_chart(cycle_plot(tables["normalized_waveforms"], [
        ("hip_flex_mean_deg", "Heup", "rgba(35,121,109,1)"),
        ("pelvis_flex_deg", "Bekken", "rgba(224,124,60,1)"),
        ("lumbar_flex_rel_deg", "Rug t.o.v. bekken", "rgba(72,103,148,1)"),
    ]), use_container_width=True)
    st.caption("Lijnen: gemiddelde van alle tijdgenormaliseerde herhalingen. Schaduw: ±1 SD.")
    st.subheader("Kwaliteitscontrole")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Samplefrequentie", metric_value(qc["sample_frequency_hz"], " Hz"), help="Afgeleid uit de volledige tijdas van het bestand.")
    q2.metric("Opnameduur", metric_value(qc["duration_s"], " s"), help="Duur van de ingelezen tijdreeks inclusief één sample-interval.")
    q3.metric("Ontbrekende waarden", str(qc["missing_required_values"]), help="Aantal ontbrekende of niet-numerieke waarden in verplichte kolommen.")
    q4.metric("Segmentatie klopt", "Ja" if qc["detection_matches_expected"] else "Nee", help="Vergelijking tussen gevonden en vooraf verwacht aantal herhalingen.")

with tabs[1]:
    st.subheader("Range of motion per herhaling")
    rom_cols = ["repetition", "knee_flex_deg_L_robust_rom_deg", "knee_flex_deg_R_robust_rom_deg", "hip_flex_deg_L_robust_rom_deg", "hip_flex_deg_R_robust_rom_deg", "pelvis_flex_deg_robust_rom_deg", "lumbar_flex_rel_deg_robust_rom_deg", "hip_rom_asymmetry_L_minus_R_deg", "knee_rom_asymmetry_L_minus_R_deg"]
    st.plotly_chart(line_by_repetition(reps, [
        ("hip_flex_deg_L_robust_rom_deg", "Heup links", "#23796d"),
        ("hip_flex_deg_R_robust_rom_deg", "Heup rechts", "#57a89d"),
        ("lumbar_flex_rel_deg_robust_rom_deg", "Rug t.o.v. bekken", "#486794"),
        ("pelvis_flex_deg_robust_rom_deg", "Bekken", "#e07c3c"),
    ], "Robuuste ROM (°)"), use_container_width=True)
    st.dataframe(reps[rom_cols], column_config=dataframe_config(reps[rom_cols]), hide_index=True, use_container_width=True)
    st.subheader("Buktempo per herhaling")
    tempo_cols = ["repetition", "cycle_duration_s", "flexion_duration_s", "extension_duration_s", "bottom_dwell_s"]
    st.plotly_chart(line_by_repetition(reps, [("cycle_duration_s", "Totale cyclus", "#153f3a"), ("flexion_duration_s", "Buigen", "#23796d"), ("extension_duration_s", "Terugkomen", "#e07c3c")], "Tijd (s)"), use_container_width=True)
    st.dataframe(reps[tempo_cols], column_config=dataframe_config(reps[tempo_cols]), hide_index=True, use_container_width=True)

with tabs[2]:
    st.subheader("Herhaalbaarheid van de golfvorm")
    waveform = tables["waveform_metrics"]
    st.dataframe(waveform, column_config=dataframe_config(waveform), hide_index=True, use_container_width=True)
    st.subheader("Heup-rugkoppeling")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vector-codingvariabiliteit", metric_value(coupling["mean_vector_coding_variability_deg"], "°"), help=EXACT_HELP["mean_vector_coding_variability_deg"])
    c2.metric("Geldige punten", metric_value(coupling["vector_coding_valid_points_pct"], "%", 1), help=EXACT_HELP["vector_coding_valid_points_pct"])
    c3.metric("Absolute CRP", metric_value(coupling["mean_absolute_crp_deg"], "°"), help=EXACT_HELP["mean_absolute_crp_deg"])
    c4.metric("CRP-variabiliteit", metric_value(coupling["mean_crp_variability_deg"], "°"), help=EXACT_HELP["mean_crp_variability_deg"])
    profile = tables["coupling_profile"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=profile["cycle_pct"], y=profile["vector_coding_variability_deg"], name="Vector coding", line=dict(color="#c75450", width=3)))
    fig.add_trace(go.Scatter(x=profile["cycle_pct"], y=profile["crp_variability_deg"], name="CRP", line=dict(color="#74508f", width=3)))
    fig.update_layout(template="plotly_white", height=390, xaxis_title="Bukcyclus (%)", yaxis_title="Circulaire SD (°)", margin=dict(l=20, r=20, t=25, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='notice'>Deze maten beschrijven consistentie, niet automatisch kwaliteit. Een stabiel compensatiepatroon kan klinisch nog steeds relevant zijn.</div>", unsafe_allow_html=True)

with tabs[3]:
    st.subheader("Beweging buiten het sagittale vlak")
    dev_cols = ["repetition", "trunk_lat_rel_deg_robust_rom_deg", "trunk_rot_deg_robust_rom_deg", "pelvis_abd_deg_robust_rom_deg"]
    st.plotly_chart(line_by_repetition(reps, [("trunk_lat_rel_deg_robust_rom_deg", "Lateroflexie romp", "#74508f"), ("trunk_rot_deg_robust_rom_deg", "Romprotatie", "#a65a33"), ("pelvis_abd_deg_robust_rom_deg", "Bekken lateraal", "#6c757d")], "Robuuste ROM (°)"), use_container_width=True)
    st.dataframe(reps[dev_cols], column_config=dataframe_config(reps[dev_cols]), hide_index=True, use_container_width=True)
    st.caption("Axiale romprotatie is globaal. Relatieve romp-bekkenrotatie vereist ook een bekkenrotatiesignaal.")
    st.subheader("Regulariteit, vloeiendheid en frequentie-inhoud")
    nonlinear = tables["nonlinear_metrics"].copy()
    nonlinear["uitleg"] = nonlinear["metric"].map(lambda x: EXACT_HELP.get(x, help_for(x)))
    nonlinear_config = dataframe_config(nonlinear)
    nonlinear_config["uitleg"] = st.column_config.TextColumn("Uitleg", help="Klinisch-methodologische betekenis van de maat.", width="large")
    st.dataframe(nonlinear, column_config=nonlinear_config, hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Methode en beperkingen")
    st.markdown(
        """
        - De rughoek is een planaire benadering: globale rompflexie minus globale bekkenflexie.
        - De uitkomsten zijn beschrijvend. Zonder populatiespecifieke normwaarden, test-hertestbetrouwbaarheid en minimale detecteerbare verandering worden geen klinische afkapwaarden toegepast.
        - Sample Entropy, SPARC en spectrale verhoudingen zijn exploratief en gevoelig voor datalengte en preprocessing.
        - DFA wordt bij deze korte test bewust niet berekend; daarvoor is een veel langere reeks nodig.
        """
    )
    methods = tables["methods"]
    st.dataframe(methods, column_config={
        "topic": st.column_config.TextColumn("Onderwerp", help="Onderdeel van de analyse waarop de bron betrekking heeft."),
        "reference": st.column_config.TextColumn("Wetenschappelijke bron", help="Publicatie waarop de methode of beperking is gebaseerd.", width="large"),
        "doi_or_url": st.column_config.LinkColumn("DOI / PubMed", help="Link naar de betreffende publicatie.", display_text="Open bron"),
        "use_in_script": st.column_config.TextColumn("Toepassing", help="Hoe de bron in het analysescript is gebruikt.", width="large"),
    }, hide_index=True, use_container_width=True)
    st.download_button("Download alle resultaten (.zip)", data=result["zip"], file_name="bukbewegingsanalyse_resultaten.zip", mime="application/zip", use_container_width=True, help="Bevat alle tabellen als CSV, kwaliteitscontrole als JSON en de drie figuren als PNG.")
    st.caption("Deze applicatie ondersteunt klinische beoordeling, maar stelt geen diagnose.")
