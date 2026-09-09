"""Framework-onafhankelijke pijplijn voor upload, analyse en resultaatexport."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import analysis


def run_analysis(
    uploaded_bytes: bytes,
    filename: str,
    sheet: str,
    expected_repetitions: int,
    lowpass_hz: float,
) -> dict:
    """Analyseer één upload zonder bestanden buiten de tijdelijke map te bewaren."""
    with tempfile.TemporaryDirectory(prefix="buk_analyse_") as temp_dir:
        work = Path(temp_dir)
        input_path = work / Path(filename).name
        output_dir = work / "resultaten"
        input_path.write_bytes(uploaded_bytes)
        output_dir.mkdir()

        cfg = analysis.Config(
            sheet_name=sheet,
            expected_repetitions=expected_repetitions,
            lowpass_hz=lowpass_hz,
        )
        df, fs, warnings = analysis.load_and_validate(input_path, cfg.sheet_name)
        signals = analysis.prepare_signals(df, fs, cfg)
        cycles, detection_qc = analysis.detect_cycles(
            signals["hip_flex_mean_deg"].to_numpy(), fs, cfg
        )
        reps, quartiles = analysis.calculate_repetition_metrics(
            df, signals, cycles, fs, cfg
        )
        normalized = analysis.normalize_cycles(signals, cycles, cfg)
        waveform_metrics, waveform_profile = analysis.calculate_waveform_metrics(
            normalized
        )
        coupling_summary, coupling_profile = analysis.calculate_coupling(normalized)
        nonlinear = analysis.calculate_nonlinear(
            df, signals, cycles, reps, fs, cfg
        )
        pca_scores, pca_summary = analysis.calculate_pca(normalized)
        reps = reps.merge(pca_scores, on="repetition", how="left")
        summary = analysis.summarize_repetitions(reps)
        plot_paths = analysis.make_plots(
            df, signals, cycles, normalized, coupling_profile, output_dir
        )

        quality_control = {
            "sheet": cfg.sheet_name,
            "rows": len(df),
            "duration_s": float(
                df["time_s"].iloc[-1] - df["time_s"].iloc[0] + 1.0 / fs
            ),
            "sample_frequency_hz": fs,
            "missing_required_values": int(
                df[analysis.REQUIRED_COLUMNS].isna().sum().sum()
            ),
            "warnings": warnings,
            "relative_lumbar_angle_method": (
                "trunk_flex_deg - pelvis_flex_deg (planar approximation)"
            ),
            "relative_axial_rotation_available": False,
            **detection_qc,
        }
        tables = {
            "repetition_metrics": reps,
            "summary_metrics": summary,
            "lumbopelvic_quartiles": quartiles,
            "normalized_waveforms": normalized,
            "waveform_metrics": waveform_metrics,
            "waveform_profile": waveform_profile,
            "coupling_summary": coupling_summary,
            "coupling_profile": coupling_profile,
            "nonlinear_metrics": nonlinear,
            "pca_summary": pca_summary,
            "methods": pd.DataFrame(analysis.REFERENCES),
        }
        for name, table in tables.items():
            table.to_csv(
                output_dir / f"{name}.csv",
                index=False,
                float_format="%.6f",
            )

        (output_dir / "quality_control.json").write_text(
            json.dumps(quality_control, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {
            "config": asdict(cfg),
            "quality_control": quality_control,
            "tables": list(tables),
            "plots": [path.name for path in plot_paths],
        }
        (output_dir / "analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(output_dir.iterdir()):
                archive.write(path, arcname=path.name)

        return {
            "qc": quality_control,
            "tables": tables,
            "plots": {path.name: path.read_bytes() for path in plot_paths},
            "zip": zip_buffer.getvalue(),
        }
