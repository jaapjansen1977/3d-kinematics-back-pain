#!/usr/bin/env python3
"""Klinische en bewegingswetenschappelijke analyse van herhaald bukken.

Het script leest een Excelbestand met de Sports2D-achtige kolomnamen uit het
meegeleverde voorbeeld. Het segmenteert herhalingen, berekent klinische
kinematische maten, golfvormvariabiliteit, lumbopelviene coördinatie,
deviaties buiten het sagittale vlak en exploratieve niet-lineaire maten.

Belangrijke definities
----------------------
* Positieve waarden worden als flexie geïnterpreteerd.
* ``lumbar_flex_rel_deg`` wordt bij gebrek aan oriëntatiematrices benaderd als
  globale rompflexie minus globale bekkenflexie. Dit is alleen een geldige
  benadering wanneer beide Eulerhoeken hetzelfde globale assenstelsel en
  dezelfde rotatieconventie gebruiken en de beweging grotendeels sagittaal is.
* Relatieve axiale romp-bekkenrotatie kan niet worden berekend omdat de
  brondata geen globale bekkenrotatie bevat.
* De uitkomsten zijn beschrijvend. Klinische afkapwaarden vereisen normdata,
  test-hertestbetrouwbaarheid en een minimale detecteerbare verandering.

Wetenschappelijke basis (zie ook methods.csv in de uitvoer)
-----------------------------------------------------------
Lumbopelviene ROM, timing en snelheid:
  Laird et al. 2019. doi:10.1186/s12891-018-2387-x; PMID 30658610.
Vector coding en circulaire statistiek:
  Needham et al. 2014. PMID 24485511.
  Needham et al. 2015. PMID 26303167.
  Needham et al. 2020. PMID 32629370.
Continuous relative phase:
  Lamb & Stöckl 2014. PMID 24726779.
  Zhou et al. 2016. doi:10.1016/j.clinbiomech.2015.10.012.
Sample entropy:
  Richman & Moorman 2000. doi:10.1152/ajpheart.2000.278.6.H2039.
  Thiry et al. 2022. doi:10.3390/e24040437.
Movement smoothness (SPARC):
  Balasubramanian et al. 2012. PMID 22180502.
DFA en minimale serielengte:
  Damouras et al. 2010. doi:10.1016/j.gaitpost.2009.12.002.
  Marmelat & Meidinger 2019. doi:10.1016/j.gaitpost.2019.02.023.

Voorbeeld
---------
python buk_kinematica_analyse.py input.xlsx --output-dir analyse_resultaten
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, detrend, filtfilt, find_peaks, welch
from scipy.spatial import cKDTree
from scipy.stats import circmean, circstd, linregress
from sklearn.decomposition import PCA


REQUIRED_COLUMNS = [
    "frame", "time_s", "knee_flex_deg_L", "knee_flex_deg_R",
    "hip_flex_deg_L", "hip_flex_deg_R", "pelvis_flex_deg",
    "trunk_flex_deg", "pelvis_abd_deg", "trunk_abd_deg",
    "trunk_rot_deg",
]

OPTIONAL_COLUMNS = [
    "hip_abd_deg_L", "hip_abd_deg_R", "hip_rot_deg_L", "hip_rot_deg_R",
]


@dataclass(frozen=True)
class Config:
    sheet_name: str = "angles"
    lowpass_hz: float = 6.0
    segmentation_lowpass_hz: float = 2.0
    expected_repetitions: int = 20
    min_cycle_duration_s: float = 1.10
    peak_prominence_fraction: float = 0.25
    onset_fraction: float = 0.05
    robust_low_percentile: float = 5.0
    robust_high_percentile: float = 95.0
    normalized_points: int = 101
    sampen_m: int = 2
    sampen_r_fraction_sd: float = 0.20
    spectral_low_band_hz: tuple[float, float] = (0.10, 2.0)
    spectral_high_band_hz: tuple[float, float] = (2.0, 10.0)
    dfa_min_cycles: int = 600


REFERENCES = [
    {
        "topic": "Klinische lumbopelviene kinematica",
        "reference": "Laird RA et al. Does movement matter in people with back pain? BMC Musculoskelet Disord. 2019;20:28.",
        "doi_or_url": "https://doi.org/10.1186/s12891-018-2387-x",
        "use_in_script": "ROM, bewegingsduur en onset-delay tussen lumbaal en bekken.",
    },
    {
        "topic": "Lumbopelvien ritme",
        "reference": "Zhou J et al. Differences in lumbopelvic rhythm between trunk flexion and extension. Clin Biomech. 2016;32:274-279.",
        "doi_or_url": "https://doi.org/10.1016/j.clinbiomech.2015.10.012",
        "use_in_script": "Flexie en extensie afzonderlijk beoordelen; CRP en fasevariabiliteit.",
    },
    {
        "topic": "Vector coding",
        "reference": "Needham R et al. Quantifying lumbar-pelvis coordination during gait using vector coding and circular statistics. Clin Biomech. 2014.",
        "doi_or_url": "https://pubmed.ncbi.nlm.nih.gov/24485511/",
        "use_in_script": "Coupling angle en circulaire coördinatievariabiliteit.",
    },
    {
        "topic": "Coördinatieclassificatie",
        "reference": "Needham RA et al. A new coordination pattern classification to assess gait kinematics. J Biomech. 2015.",
        "doi_or_url": "https://pubmed.ncbi.nlm.nih.gov/26303167/",
        "use_in_script": "Vier vector-codingpatronen: heupdominant, rugdominant, in-fase en anti-fase.",
    },
    {
        "topic": "Continuous relative phase",
        "reference": "Lamb PF, Stöckl M. On the use of continuous relative phase. Hum Mov Sci. 2014.",
        "doi_or_url": "https://pubmed.ncbi.nlm.nih.gov/24726779/",
        "use_in_script": "Normalisatie en voorzichtige interpretatie van CRP.",
    },
    {
        "topic": "Sample entropy",
        "reference": "Richman JS, Moorman JR. Physiological time-series analysis using approximate entropy and sample entropy. Am J Physiol. 2000.",
        "doi_or_url": "https://doi.org/10.1152/ajpheart.2000.278.6.H2039",
        "use_in_script": "Definitie van Sample Entropy zonder self-matches.",
    },
    {
        "topic": "Sample entropy bij herhaald bukken",
        "reference": "Thiry P et al. Sample Entropy as a Tool to Assess Lumbo-Pelvic Movements. Entropy. 2022;24:437.",
        "doi_or_url": "https://doi.org/10.3390/e24040437",
        "use_in_script": "Exploratieve SampEn op hoeksnelheid; 70 s/50 herhalingen als relevantere onderzoeksduur.",
    },
    {
        "topic": "Bewegingsvloeiendheid",
        "reference": "Balasubramanian S et al. A robust and sensitive metric for quantifying movement smoothness. IEEE Trans Biomed Eng. 2012.",
        "doi_or_url": "https://pubmed.ncbi.nlm.nih.gov/22180502/",
        "use_in_script": "Spectral Arc Length (SPARC) op absolute hoeksnelheid per herhaling.",
    },
    {
        "topic": "DFA-datalengte",
        "reference": "Damouras S et al. An empirical examination of detrended fluctuation analysis for gait data. Gait Posture. 2010.",
        "doi_or_url": "https://doi.org/10.1016/j.gaitpost.2009.12.002",
        "use_in_script": "DFA alleen bij minimaal 600 opeenvolgende cycli; boxgroottes 16 tot N/9.",
    },
    {
        "topic": "DFA-betrouwbaarheid",
        "reference": "Marmelat V, Meidinger RL. Fractal analysis of gait in Parkinson's disease: three minutes is not enough. Gait Posture. 2019.",
        "doi_or_url": "https://doi.org/10.1016/j.gaitpost.2019.02.023",
        "use_in_script": "Korte cyclische reeksen geven geen robuuste DFA-schatter.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xlsx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet", default="angles")
    parser.add_argument("--expected-repetitions", type=int, default=20)
    parser.add_argument("--lowpass-hz", type=float, default=6.0)
    return parser.parse_args()


def load_and_validate(path: Path, sheet_name: str) -> tuple[pd.DataFrame, float, list[str]]:
    df = pd.read_excel(path, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Verplichte kolommen ontbreken: {missing}")
    df = df.copy()
    for col in set(REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"Ontbrekende/niet-numerieke waarden in verplichte kolommen: {bad[bad > 0].to_dict()}")
    time = df["time_s"].to_numpy(float)
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time_s moet strikt oplopend zijn.")
    # Gebruik de helling over de volledige opname. Dit blijft nauwkeurig wanneer
    # time_s, zoals in het voorbeeldbestand, op drie decimalen is afgerond en de
    # lokale stappen daardoor afwisselend 0,016 en 0,017 s zijn.
    fs = float((len(time) - 1) / (time[-1] - time[0]))
    expected_dt = 1.0 / fs
    jitter = float(np.max(np.abs(dt - expected_dt)))
    warnings: list[str] = []
    if jitter > 0.10 * expected_dt:
        warnings.append("Tijdstappen zijn niet uniform; resampling wordt aanbevolen.")
    if abs(fs - 60.0) > 1.0:
        warnings.append(f"Afgeleide samplefrequentie is {fs:.2f} Hz en wijkt af van 60 Hz.")
    return df, fs, warnings


def lowpass(values: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Nul-fase Butterworth-filter; cutoff blijft configureerbaar en wordt gerapporteerd."""
    x = np.asarray(values, dtype=float)
    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return x.copy()
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def prepare_signals(df: pd.DataFrame, fs: float, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    source_cols = [c for c in df.columns if c.endswith("_deg_L") or c.endswith("_deg_R") or c.endswith("_deg")]
    for col in source_cols:
        out[col] = lowpass(df[col].to_numpy(float), fs, cfg.lowpass_hz)
    out["hip_flex_mean_deg"] = 0.5 * (out["hip_flex_deg_L"] + out["hip_flex_deg_R"])
    out["knee_flex_mean_deg"] = 0.5 * (out["knee_flex_deg_L"] + out["knee_flex_deg_R"])
    # Planare benadering; zie module-docstring. Bij 3D-data moeten relatieve
    # rotatiematrices/quaternions worden gebruikt vóór Euler-decompositie.
    out["lumbar_flex_rel_deg"] = out["trunk_flex_deg"] - out["pelvis_flex_deg"]
    out["trunk_lat_rel_deg"] = out["trunk_abd_deg"] - out["pelvis_abd_deg"]
    return out


def detect_cycles(primary: np.ndarray, fs: float, cfg: Config) -> tuple[list[dict], dict]:
    seg = lowpass(primary, fs, cfg.segmentation_lowpass_hz)
    prominence = max(5.0, cfg.peak_prominence_fraction * float(np.ptp(seg)))
    distance = max(1, int(cfg.min_cycle_duration_s * fs))
    peaks, properties = find_peaks(seg, prominence=prominence, distance=distance)
    if len(peaks) == 0:
        raise RuntimeError("Geen bukpieken gevonden.")

    boundaries = [int(np.argmin(seg[: peaks[0] + 1]))]
    for left, right in zip(peaks[:-1], peaks[1:]):
        boundaries.append(int(left + np.argmin(seg[left:right + 1])))
    boundaries.append(int(peaks[-1] + np.argmin(seg[peaks[-1]:])))

    cycles = []
    for i, peak in enumerate(peaks):
        start, end = boundaries[i], boundaries[i + 1]
        if not (start < peak < end):
            continue
        cycles.append({"repetition": len(cycles) + 1, "start_idx": start, "peak_idx": int(peak), "end_idx": end})

    qc = {
        "detected_repetitions": len(cycles),
        "expected_repetitions": cfg.expected_repetitions,
        "detection_matches_expected": len(cycles) == cfg.expected_repetitions,
        "segmentation_signal": "hip_flex_mean_deg",
        "segmentation_lowpass_hz": cfg.segmentation_lowpass_hz,
        "peak_prominence_deg": prominence,
        "peak_prominences_deg": [float(x) for x in properties.get("prominences", [])],
    }
    if len(cycles) != cfg.expected_repetitions:
        raise RuntimeError(
            f"Er zijn {len(cycles)} herhalingen gevonden; verwacht {cfg.expected_repetitions}. "
            "Controleer de segmentatiegrafiek of pas parameters aan."
        )
    return cycles, qc


def robust_rom(x: np.ndarray, cfg: Config) -> float:
    return float(np.percentile(x, cfg.robust_high_percentile) - np.percentile(x, cfg.robust_low_percentile))


def first_fraction_crossing(x: np.ndarray, fraction: float) -> int:
    if len(x) < 2:
        return 0
    start, end = float(x[0]), float(x[-1])
    threshold = start + fraction * (end - start)
    if end >= start:
        hits = np.flatnonzero(x >= threshold)
    else:
        hits = np.flatnonzero(x <= threshold)
    return int(hits[0]) if len(hits) else 0


def calculate_repetition_metrics(
    df: pd.DataFrame, signals: pd.DataFrame, cycles: list[dict], fs: float, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = df["time_s"].to_numpy(float)
    dt = 1.0 / fs
    sagittal = [
        "knee_flex_deg_L", "knee_flex_deg_R", "hip_flex_deg_L", "hip_flex_deg_R",
        "pelvis_flex_deg", "trunk_flex_deg", "lumbar_flex_rel_deg",
    ]
    deviations = [
        "pelvis_abd_deg", "trunk_lat_rel_deg", "trunk_rot_deg",
        *[c for c in OPTIONAL_COLUMNS if c in signals.columns],
    ]
    records: list[dict] = []
    quartile_records: list[dict] = []

    for cyc in cycles:
        rep = cyc["repetition"]
        s, p, e = cyc["start_idx"], cyc["peak_idx"], cyc["end_idx"]
        rec: dict[str, float | int] = {
            "repetition": rep,
            "start_frame": int(df["frame"].iloc[s]),
            "peak_frame": int(df["frame"].iloc[p]),
            "end_frame": int(df["frame"].iloc[e]),
            "start_time_s": float(time[s]),
            "peak_time_s": float(time[p]),
            "end_time_s": float(time[e]),
            "cycle_duration_s": float(time[e] - time[s]),
            "flexion_duration_s": float(time[p] - time[s]),
            "extension_duration_s": float(time[e] - time[p]),
            "flexion_fraction_pct": 100.0 * float(time[p] - time[s]) / max(float(time[e] - time[s]), 1e-9),
        }

        primary = signals["hip_flex_mean_deg"].iloc[s:e + 1].to_numpy()
        threshold = np.percentile(primary, 95)
        rec["bottom_dwell_s"] = float(np.count_nonzero(primary >= threshold) * dt)

        for name in sagittal:
            x = signals[name].iloc[s:e + 1].to_numpy()
            v = np.gradient(x, dt)
            rec[f"{name}_rom_deg"] = float(np.ptp(x))
            rec[f"{name}_robust_rom_deg"] = robust_rom(x, cfg)
            rec[f"{name}_min_deg"] = float(np.min(x))
            rec[f"{name}_max_deg"] = float(np.max(x))
            rec[f"{name}_start_deg"] = float(x[0])
            rec[f"{name}_end_deg"] = float(x[-1])
            rec[f"{name}_residual_end_deg"] = float(x[-1] - x[0])
            peak_local = p - s
            rec[f"{name}_peak_flex_velocity_deg_s"] = float(np.max(v[: peak_local + 1]))
            rec[f"{name}_peak_extension_velocity_deg_s"] = float(np.min(v[peak_local:]))
            rec[f"{name}_mean_abs_velocity_deg_s"] = float(np.mean(np.abs(v)))

        for name in deviations:
            x = signals[name].iloc[s:e + 1].to_numpy()
            centered = x - x[0]
            rec[f"{name}_rom_deg"] = float(np.ptp(x))
            rec[f"{name}_robust_rom_deg"] = robust_rom(x, cfg)
            rec[f"{name}_rms_from_start_deg"] = float(np.sqrt(np.mean(centered**2)))
            rec[f"{name}_max_abs_from_start_deg"] = float(np.max(np.abs(centered)))
            rec[f"{name}_signed_at_peak_deg"] = float(x[p - s] - x[0])

        rec["hip_rom_asymmetry_L_minus_R_deg"] = (
            rec["hip_flex_deg_L_robust_rom_deg"] - rec["hip_flex_deg_R_robust_rom_deg"]
        )
        rec["knee_rom_asymmetry_L_minus_R_deg"] = (
            rec["knee_flex_deg_L_robust_rom_deg"] - rec["knee_flex_deg_R_robust_rom_deg"]
        )

        lum_flex = signals["lumbar_flex_rel_deg"].iloc[s:p + 1].to_numpy()
        pel_flex = signals["pelvis_flex_deg"].iloc[s:p + 1].to_numpy()
        lum_on = first_fraction_crossing(lum_flex, cfg.onset_fraction)
        pel_on = first_fraction_crossing(pel_flex, cfg.onset_fraction)
        rec["pelvis_onset_delay_vs_lumbar_s"] = float((pel_on - lum_on) * dt)
        lum_exc = abs(float(lum_flex[-1] - lum_flex[0]))
        pel_exc = abs(float(pel_flex[-1] - pel_flex[0]))
        denom = lum_exc + pel_exc
        rec["lumbar_contribution_flexion_pct"] = 100.0 * lum_exc / denom if denom > 1e-9 else np.nan
        rec["pelvis_contribution_flexion_pct"] = 100.0 * pel_exc / denom if denom > 1e-9 else np.nan
        rec["lumbar_to_pelvis_excursion_ratio"] = lum_exc / pel_exc if pel_exc >= 2.0 else np.nan

        for phase_name, a, b in [("flexion", s, p), ("extension", p, e)]:
            lum = signals["lumbar_flex_rel_deg"].iloc[a:b + 1].to_numpy()
            pel = signals["pelvis_flex_deg"].iloc[a:b + 1].to_numpy()
            grid = np.linspace(0, len(lum) - 1, 5)
            lum_q = np.interp(grid, np.arange(len(lum)), lum)
            pel_q = np.interp(grid, np.arange(len(pel)), pel)
            for q in range(4):
                dl = abs(float(lum_q[q + 1] - lum_q[q]))
                dp = abs(float(pel_q[q + 1] - pel_q[q]))
                total = dl + dp
                quartile_records.append({
                    "repetition": rep,
                    "phase": phase_name,
                    "quartile": q + 1,
                    "lumbar_change_deg": dl,
                    "pelvis_change_deg": dp,
                    "lumbar_contribution_pct": 100.0 * dl / total if total > 1e-9 else np.nan,
                    "lumbar_to_pelvis_ratio": dl / dp if dp >= 2.0 else np.nan,
                })
        records.append(rec)
    return pd.DataFrame(records), pd.DataFrame(quartile_records)


def normalize_cycles(signals: pd.DataFrame, cycles: list[dict], cfg: Config) -> pd.DataFrame:
    names = [
        "hip_flex_mean_deg", "knee_flex_mean_deg", "pelvis_flex_deg",
        "trunk_flex_deg", "lumbar_flex_rel_deg", "trunk_lat_rel_deg", "trunk_rot_deg",
    ]
    rows = []
    target = np.linspace(0.0, 100.0, cfg.normalized_points)
    for cyc in cycles:
        s, e = cyc["start_idx"], cyc["end_idx"]
        source = np.linspace(0.0, 100.0, e - s + 1)
        block = {"repetition": np.repeat(cyc["repetition"], cfg.normalized_points), "cycle_pct": target}
        for name in names:
            block[name] = np.interp(target, source, signals[name].iloc[s:e + 1].to_numpy())
        rows.append(pd.DataFrame(block))
    return pd.concat(rows, ignore_index=True)


def calculate_waveform_metrics(normalized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [c for c in normalized.columns if c not in {"repetition", "cycle_pct"}]
    reps = sorted(normalized["repetition"].unique())
    profile_rows = []
    metric_rows = []
    for name in names:
        matrix = np.vstack([normalized.loc[normalized["repetition"] == r, name].to_numpy() for r in reps])
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        rmse = np.sqrt(np.mean((matrix - mean) ** 2, axis=1))
        consecutive = np.sqrt(np.mean(np.diff(matrix, axis=0) ** 2, axis=1))
        correlations = []
        for row in matrix:
            correlations.append(float(np.corrcoef(row, mean)[0, 1]) if np.std(row) > 0 and np.std(mean) > 0 else np.nan)
        metric_rows.append({
            "signal": name,
            "mean_pointwise_sd_deg": float(np.mean(sd)),
            "max_pointwise_sd_deg": float(np.max(sd)),
            "mean_rmse_to_mean_deg": float(np.mean(rmse)),
            "median_consecutive_rmse_deg": float(np.median(consecutive)),
            "mean_correlation_with_mean_waveform": float(np.nanmean(correlations)),
        })
        for pct, mu, sigma in zip(range(len(mean)), mean, sd):
            profile_rows.append({"signal": name, "cycle_pct": pct, "mean_deg": mu, "sd_deg": sigma})
    return pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def wrap_degrees(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def classify_vector_coding(angle_deg: np.ndarray) -> np.ndarray:
    """Vier patroonklassen volgens de 45°-sectorlogica van modified vector coding."""
    a = np.asarray(angle_deg) % 360.0
    labels = np.empty(a.shape, dtype=object)
    hip_dom = ((a < 22.5) | (a >= 337.5) | ((a >= 157.5) & (a < 202.5)))
    lumbar_dom = (((a >= 67.5) & (a < 112.5)) | ((a >= 247.5) & (a < 292.5)))
    in_phase = (((a >= 22.5) & (a < 67.5)) | ((a >= 202.5) & (a < 247.5)))
    labels[hip_dom] = "heupdominant"
    labels[lumbar_dom] = "rugdominant"
    labels[in_phase] = "in_fase"
    labels[~(hip_dom | lumbar_dom | in_phase)] = "anti_fase"
    return labels


def calculate_coupling(normalized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vector coding en CRP voor de koppeling heupflexie versus relatieve rugflexie.

    Vector coding gebruikt de richting van opeenvolgende punten in het
    angle-angle-diagram en circulaire statistiek over herhalingen.
    CRP wordt aanvullend gerapporteerd en is gevoeliger voor normalisatie en
    lage snelheden rond keerpunten (Lamb & Stöckl 2014; PMID 24726779).
    """
    reps = sorted(normalized["repetition"].unique())
    ca_matrix, crp_matrix = [], []
    all_classes = []
    valid_count = 0
    total_count = 0
    for rep in reps:
        block = normalized[normalized["repetition"] == rep]
        hip = block["hip_flex_mean_deg"].to_numpy()
        lum = block["lumbar_flex_rel_deg"].to_numpy()
        dhip = np.gradient(hip)
        dlum = np.gradient(lum)
        resultant_speed = np.hypot(dhip, dlum)
        # De coupling angle is mathematisch slecht bepaald wanneer beide
        # segmenten vrijwel stilstaan, met name tijdens het onderste keerpunt.
        # Maskering voorkomt dat kleine meetruis daar een grote circulaire SD
        # veroorzaakt. De drempel is relatief aan iedere afzonderlijke cyclus.
        valid = resultant_speed >= 0.05 * np.max(resultant_speed)
        ca = np.degrees(np.arctan2(dlum, dhip)) % 360.0
        ca[~valid] = np.nan
        ca_matrix.append(ca)
        all_classes.extend(classify_vector_coding(ca[valid]))
        valid_count += int(np.count_nonzero(valid))
        total_count += int(len(valid))

        def phase_angle(x: np.ndarray) -> np.ndarray:
            midpoint = 0.5 * (np.max(x) + np.min(x))
            amp = max(0.5 * np.ptp(x), 1e-9)
            xn = (x - midpoint) / amp
            vn = np.gradient(xn)
            vmax = max(np.max(np.abs(vn)), 1e-9)
            return np.arctan2(vn / vmax, xn)

        crp = wrap_degrees(np.degrees(phase_angle(lum) - phase_angle(hip)))
        crp_matrix.append(crp)

    ca_rad = np.radians(np.vstack(ca_matrix))
    crp_rad = np.radians(np.vstack(crp_matrix))
    ca_mean = np.full(ca_rad.shape[1], np.nan)
    ca_sd = np.full(ca_rad.shape[1], np.nan)
    for j in range(ca_rad.shape[1]):
        values = ca_rad[np.isfinite(ca_rad[:, j]), j]
        if len(values) >= 2:
            ca_mean[j] = np.degrees(circmean(values, high=2 * np.pi, low=0)) % 360.0
            ca_sd[j] = np.degrees(circstd(values, high=2 * np.pi, low=0))
    crp_mean = wrap_degrees(np.degrees(circmean(crp_rad, high=np.pi, low=-np.pi, axis=0)))
    crp_sd = np.degrees(circstd(crp_rad, high=np.pi, low=-np.pi, axis=0))
    profile = pd.DataFrame({
        "cycle_pct": np.arange(len(ca_mean)),
        "vector_coding_mean_angle_deg": ca_mean,
        "vector_coding_variability_deg": ca_sd,
        "crp_mean_deg": crp_mean,
        "crp_variability_deg": crp_sd,
        "valid_vector_coding_cycles_n": np.sum(np.isfinite(ca_rad), axis=0),
    })
    labels, counts = np.unique(np.asarray(all_classes, dtype=str), return_counts=True)
    percentages = {f"vector_coding_{label}_pct": 100.0 * count / len(all_classes) for label, count in zip(labels, counts)}
    summary = pd.DataFrame([{
        "coupling_pair": "hip_flex_mean_deg versus lumbar_flex_rel_deg",
        "mean_vector_coding_variability_deg": float(np.nanmean(ca_sd)),
        "vector_coding_valid_points_pct": 100.0 * valid_count / total_count,
        "mean_absolute_crp_deg": float(np.mean(np.abs(crp_mean))),
        "mean_crp_variability_deg": float(np.mean(crp_sd)),
        **percentages,
    }])
    return summary, profile


def sample_entropy(x: np.ndarray, m: int = 2, r_fraction_sd: float = 0.2) -> tuple[float, float, int]:
    """Sample Entropy volgens Richman & Moorman (2000), Chebyshev-afstand.

    Self-matches worden verwijderd. r wordt als fractie van de standaarddeviatie
    gerapporteerd, omdat SampEn sterk parameter- en datalengteafhankelijk is.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    tolerance = r_fraction_sd * float(np.std(x, ddof=0))
    if len(x) <= m + 2 or tolerance <= 0:
        return np.nan, tolerance, len(x)

    def match_count(dim: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, dim)
        tree = cKDTree(emb)
        counts = tree.query_ball_point(emb, tolerance, p=np.inf, return_length=True)
        return int(np.sum(counts - 1))

    b = match_count(m)
    a = match_count(m + 1)
    value = -math.log(a / b) if a > 0 and b > 0 else np.inf
    return float(value), tolerance, len(x)


def sparc(speed: np.ndarray, fs: float, max_frequency_hz: float = 10.0, amplitude_threshold: float = 0.05) -> float:
    """Spectral Arc Length, als exploratieve vloeiendheidsmaat.

    Gebaseerd op Balasubramanian et al. (2012; PMID 22180502). De absolute
    hoeksnelheid wordt gebruikt. Minder negatieve/hogere waarden betekenen een
    vloeiender snelheidsprofiel, mits taak en preprocessing gelijk blijven.
    """
    x = np.asarray(speed, dtype=float)
    x = np.abs(x)
    if len(x) < 8 or np.max(x) <= 0:
        return np.nan
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spectrum /= max(np.max(spectrum), 1e-12)
    valid = np.flatnonzero((freqs <= max_frequency_hz) & (spectrum >= amplitude_threshold))
    if len(valid) < 2:
        return np.nan
    last = int(valid[-1])
    f = freqs[: last + 1]
    s = spectrum[: last + 1]
    f_norm = (f - f[0]) / max(f[-1] - f[0], 1e-12)
    return float(-np.sum(np.sqrt(np.diff(f_norm) ** 2 + np.diff(s) ** 2)))


def bandpower_ratio(x: np.ndarray, fs: float, low_band: tuple[float, float], high_band: tuple[float, float]) -> dict:
    """Exploratieve spectrale verhouding; vaste banden worden expliciet gerapporteerd."""
    f, pxx = welch(detrend(np.asarray(x, dtype=float)), fs=fs, nperseg=min(512, len(x)))
    def power(band: tuple[float, float]) -> float:
        mask = (f >= band[0]) & (f < band[1])
        return float(trapezoid(pxx[mask], f[mask])) if np.count_nonzero(mask) >= 2 else np.nan
    low = power(low_band)
    high = power(high_band)
    total = low + high
    return {"low_band_power": low, "high_band_power": high, "high_fraction": high / total if total > 0 else np.nan}


def dfa_alpha_if_eligible(x: Iterable[float], cfg: Config) -> tuple[float, str]:
    """DFA met boxgroottes 16..N/9; bij minder dan 600 cycli bewust niet berekend."""
    arr = np.asarray(list(x), dtype=float)
    n = len(arr)
    if n < cfg.dfa_min_cycles:
        return np.nan, f"Niet berekend: {n} cycli; minimum ingesteld op {cfg.dfa_min_cycles}."
    profile = np.cumsum(arr - np.mean(arr))
    sizes = np.unique(np.logspace(np.log10(16), np.log10(n / 9), 12).astype(int))
    fluctuations, used = [], []
    for size in sizes:
        segments = n // size
        if segments < 4:
            continue
        rms = []
        for j in range(segments):
            y = profile[j * size:(j + 1) * size]
            t = np.arange(size)
            fit = np.polyval(np.polyfit(t, y, 1), t)
            rms.append(np.sqrt(np.mean((y - fit) ** 2)))
        fluctuations.append(np.mean(rms))
        used.append(size)
    if len(used) < 4:
        return np.nan, "Niet berekend: onvoldoende geldige boxgroottes."
    alpha = linregress(np.log(used), np.log(fluctuations)).slope
    return float(alpha), "Berekend op een voldoende lange cyclusreeks."


def calculate_nonlinear(
    df: pd.DataFrame, signals: pd.DataFrame, cycles: list[dict], reps: pd.DataFrame, fs: float, cfg: Config
) -> pd.DataFrame:
    s, e = cycles[0]["start_idx"], cycles[-1]["end_idx"]
    dt = 1.0 / fs
    rows = []
    for name in ["hip_flex_mean_deg", "lumbar_flex_rel_deg", "pelvis_flex_deg"]:
        angle = signals[name].iloc[s:e + 1].to_numpy()
        velocity = np.gradient(angle, dt)
        se, tolerance, n = sample_entropy(velocity, cfg.sampen_m, cfg.sampen_r_fraction_sd)
        spectral = bandpower_ratio(velocity, fs, cfg.spectral_low_band_hz, cfg.spectral_high_band_hz)
        cycle_sparc = []
        for cyc in cycles:
            a, b = cyc["start_idx"], cyc["end_idx"]
            v = np.gradient(signals[name].iloc[a:b + 1].to_numpy(), dt)
            cycle_sparc.append(sparc(v, fs))
        rows.extend([
            {
                "signal": name, "metric": "sample_entropy_velocity", "value": se,
                "unit": "dimensionless", "status": "exploratief",
                "parameters": f"m={cfg.sampen_m}; r={cfg.sampen_r_fraction_sd}*SD={tolerance:.4f}; N={n}",
            },
            {
                "signal": name, "metric": "SPARC_angular_speed_mean", "value": float(np.nanmean(cycle_sparc)),
                "unit": "dimensionless", "status": "exploratief",
                "parameters": "per cyclus; absolute hoeksnelheid; max 10 Hz; amplitudedrempel 0.05",
            },
            {
                "signal": name, "metric": "spectral_high_fraction", "value": spectral["high_fraction"],
                "unit": "proportion", "status": "exploratief; gevoelig voor filtering/meetruis",
                "parameters": f"laag={cfg.spectral_low_band_hz} Hz; hoog={cfg.spectral_high_band_hz} Hz",
            },
        ])

    for feature in ["cycle_duration_s", "lumbar_flex_rel_deg_robust_rom_deg"]:
        alpha, status = dfa_alpha_if_eligible(reps[feature], cfg)
        rows.append({
            "signal": feature, "metric": "DFA_alpha", "value": alpha,
            "unit": "dimensionless", "status": status,
            "parameters": f"boxgroottes 16..N/9; minimaal {cfg.dfa_min_cycles} cycli",
        })
    rows.append({
        "signal": "all", "metric": "approximate_entropy", "value": np.nan,
        "unit": "dimensionless", "status": "Niet berekend; redundant naast SampEn en sterker beïnvloed door self-matches.",
        "parameters": "Bewuste methodiekeuze",
    })
    return pd.DataFrame(rows)


def calculate_pca(normalized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beschrijvende multivariate PCA van tijdgenormaliseerde herhalingen."""
    features = ["hip_flex_mean_deg", "lumbar_flex_rel_deg", "pelvis_flex_deg", "knee_flex_mean_deg"]
    reps = sorted(normalized["repetition"].unique())
    matrix = np.vstack([
        np.concatenate([normalized.loc[normalized["repetition"] == rep, f].to_numpy() for f in features])
        for rep in reps
    ])
    pca = PCA().fit(matrix)
    scores = pca.transform(matrix)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k90 = int(np.searchsorted(cumulative, 0.90) + 1)
    reconstruction = scores[:, :k90] @ pca.components_[:k90] + pca.mean_
    rmse = np.sqrt(np.mean((matrix - reconstruction) ** 2, axis=1))
    score_df = pd.DataFrame({"repetition": reps, "pca_reconstruction_rmse_deg": rmse})
    for i in range(min(5, scores.shape[1])):
        score_df[f"PC{i + 1}_score"] = scores[:, i]
    summary = pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_pct": 100.0 * pca.explained_variance_ratio_,
        "cumulative_variance_pct": 100.0 * cumulative,
        "components_for_90_pct": k90,
    })
    return score_df, summary


def summarize_repetitions(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in reps.select_dtypes(include=[np.number]).columns:
        if col in {"repetition", "start_frame", "peak_frame", "end_frame", "start_time_s", "peak_time_s", "end_time_s"}:
            continue
        x = reps[col].to_numpy(float)
        valid = np.isfinite(x)
        if np.count_nonzero(valid) == 0:
            continue
        xv = x[valid]
        slope = linregress(np.arange(1, len(x) + 1)[valid], xv).slope if len(xv) >= 3 else np.nan
        first = float(np.mean(x[: min(5, len(x))]))
        last = float(np.mean(x[max(0, len(x) - 5):]))
        mean = float(np.mean(xv))
        sd = float(np.std(xv, ddof=1)) if len(xv) > 1 else np.nan
        rows.append({
            "metric": col,
            "mean": mean,
            "sd": sd,
            "cv_pct": 100.0 * sd / abs(mean) if np.isfinite(sd) and abs(mean) > 1e-9 else np.nan,
            "median": float(np.median(xv)),
            "iqr": float(np.percentile(xv, 75) - np.percentile(xv, 25)),
            "min": float(np.min(xv)),
            "max": float(np.max(xv)),
            "trend_per_repetition": float(slope),
            "last5_minus_first5": last - first,
        })
    return pd.DataFrame(rows)


def make_plots(
    df: pd.DataFrame, signals: pd.DataFrame, cycles: list[dict], normalized: pd.DataFrame,
    coupling_profile: pd.DataFrame, output_dir: Path
) -> list[Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    time = df["time_s"].to_numpy()
    peaks = [c["peak_idx"] for c in cycles]
    boundaries = [cycles[0]["start_idx"]] + [c["end_idx"] for c in cycles]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time, signals["hip_flex_mean_deg"], label="Heupflexie gemiddeld", color="#2F75B5")
    axes[0].plot(time, signals["pelvis_flex_deg"], label="Bekkenflexie globaal", color="#ED7D31")
    axes[0].plot(time, signals["lumbar_flex_rel_deg"], label="Rugflexie relatief", color="#70AD47")
    axes[0].scatter(time[peaks], signals["hip_flex_mean_deg"].iloc[peaks], s=18, color="#C00000", label="Keerpunt")
    for idx in boundaries:
        axes[0].axvline(time[idx], color="#BFBFBF", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("Hoek (°)")
    axes[0].set_title("Segmentatie en sagittale beweging")
    axes[0].legend(ncol=4, loc="upper center")
    axes[0].grid(alpha=0.2)

    axes[1].plot(time, signals["trunk_lat_rel_deg"], label="Lateroflexie romp t.o.v. bekken", color="#7030A0")
    axes[1].plot(time, signals["trunk_rot_deg"], label="Romprotatie globaal", color="#A64B00")
    axes[1].plot(time, signals["pelvis_abd_deg"], label="Bekken lateroflexie globaal", color="#7F7F7F")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_xlabel("Tijd (s)")
    axes[1].set_ylabel("Hoek (°)")
    axes[1].set_title("Deviaties buiten het sagittale vlak")
    axes[1].legend(ncol=3, loc="upper center")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    p1 = output_dir / "klinisch_overzicht.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, name, title, color in [
        (axes[0], "hip_flex_mean_deg", "Heupflexie", "#2F75B5"),
        (axes[1], "pelvis_flex_deg", "Bekkenflexie globaal", "#ED7D31"),
        (axes[2], "lumbar_flex_rel_deg", "Rugflexie relatief", "#70AD47"),
    ]:
        matrix = np.vstack([g[name].to_numpy() for _, g in normalized.groupby("repetition")])
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        pct = np.arange(matrix.shape[1])
        for row in matrix:
            ax.plot(pct, row, color=color, alpha=0.12, lw=0.8)
        ax.fill_between(pct, mean - sd, mean + sd, color=color, alpha=0.22, label="±1 SD")
        ax.plot(pct, mean, color=color, lw=2.2, label="Gemiddelde")
        ax.set_title(title)
        ax.set_xlabel("Bukcyclus (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Hoek (°)")
    axes[2].legend(loc="upper right")
    fig.suptitle("Tijdgenormaliseerde golfvormen van 20 herhalingen")
    fig.tight_layout()
    p2 = output_dir / "golfvorm_variabiliteit.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(coupling_profile["cycle_pct"], coupling_profile["vector_coding_mean_angle_deg"], color="#2F75B5")
    axes[0].set_ylabel("Coupling angle (°)")
    axes[0].set_title("Vector coding: gemiddelde coupling angle")
    axes[0].grid(alpha=0.2)
    axes[1].plot(coupling_profile["cycle_pct"], coupling_profile["vector_coding_variability_deg"], color="#C00000", label="Vector-codingvariabiliteit")
    axes[1].plot(coupling_profile["cycle_pct"], coupling_profile["crp_variability_deg"], color="#7030A0", label="CRP-variabiliteit")
    axes[1].set_xlabel("Bukcyclus (%)")
    axes[1].set_ylabel("Circulaire SD (°)")
    axes[1].set_title("Stabiliteit van de heup-rugkoppeling")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    p3 = output_dir / "interjoint_coupling.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2, p3]


def main() -> None:
    args = parse_args()
    cfg = Config(sheet_name=args.sheet, expected_repetitions=args.expected_repetitions, lowpass_hz=args.lowpass_hz)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, fs, warnings = load_and_validate(args.input_xlsx.resolve(), cfg.sheet_name)
    signals = prepare_signals(df, fs, cfg)
    cycles, detection_qc = detect_cycles(signals["hip_flex_mean_deg"].to_numpy(), fs, cfg)
    reps, quartiles = calculate_repetition_metrics(df, signals, cycles, fs, cfg)
    normalized = normalize_cycles(signals, cycles, cfg)
    waveform_metrics, waveform_profile = calculate_waveform_metrics(normalized)
    coupling_summary, coupling_profile = calculate_coupling(normalized)
    nonlinear = calculate_nonlinear(df, signals, cycles, reps, fs, cfg)
    pca_scores, pca_summary = calculate_pca(normalized)
    reps = reps.merge(pca_scores, on="repetition", how="left")
    summary = summarize_repetitions(reps)
    plot_paths = make_plots(df, signals, cycles, normalized, coupling_profile, output_dir)

    qc = {
        "input_file": str(args.input_xlsx.resolve()),
        "sheet": cfg.sheet_name,
        "rows": len(df),
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0] + 1.0 / fs),
        "sample_frequency_hz": fs,
        "missing_required_values": int(df[REQUIRED_COLUMNS].isna().sum().sum()),
        "warnings": warnings,
        "relative_lumbar_angle_method": "trunk_flex_deg - pelvis_flex_deg (planar approximation)",
        "relative_axial_rotation_available": False,
        **detection_qc,
    }

    outputs = {
        "repetition_metrics.csv": reps,
        "summary_metrics.csv": summary,
        "lumbopelvic_quartiles.csv": quartiles,
        "normalized_waveforms.csv": normalized,
        "waveform_metrics.csv": waveform_metrics,
        "waveform_profile.csv": waveform_profile,
        "coupling_summary.csv": coupling_summary,
        "coupling_profile.csv": coupling_profile,
        "nonlinear_metrics.csv": nonlinear,
        "pca_summary.csv": pca_summary,
        "methods.csv": pd.DataFrame(REFERENCES),
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.6f")
    # JSON-kopie voor verliesvrije, getypeerde overdracht naar het afzonderlijke
    # presentatiebestand. Dit verandert geen analyses of waarden.
    workbook_payload = {
        filename.removesuffix(".csv"): json.loads(table.to_json(orient="records"))
        for filename, table in outputs.items()
    }
    (output_dir / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "quality_control.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "config": asdict(cfg),
        "quality_control": qc,
        "tables": list(outputs),
        "plots": [p.name for p in plot_paths],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
