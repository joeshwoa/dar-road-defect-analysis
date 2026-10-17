"""Generate README.md (English) and README.ar.md (Arabic) with the numbers taken from the result files.

    python scripts/build_readme.py --results results --bench results/benchmarks/benchmark_t4.json

Every number in the READMEs comes from results/clip*/{defects.csv, road_summary.json, run_info.json, road_furniture.csv},
so the text can never drift from the delivered outputs. Re-run after every pipeline run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(res: Path, clip: str) -> dict:
    d = pd.read_csv(res / clip / "defects.csv")
    ri = json.loads((res / clip / "run_info.json").read_text())
    rs = json.loads((res / clip / "road_summary.json").read_text())
    fu = res / clip / "road_furniture.csv"
    try:
        f = pd.read_csv(fu) if fu.exists() else None
    except pd.errors.EmptyDataError:  # clip with no road furniture rows
        f = None
    return dict(d=d, ri=ri, rs=rs, f=f)


def counts_table(d: pd.DataFrame, header: tuple[str, str]) -> str:
    vc = d.groupby("defect_type").agg(n=("track_id", "count"), conf=("max_confidence", "mean")).sort_values("n", ascending=False)
    rows = [f"| {header[0]} | {header[1]} | mean best conf |", "|---|---:|---:|"]
    for k, r in vc.iterrows():
        rows.append(f"| {k} | {int(r.n)} | {r.conf:.2f} |")
    rows.append(f"| **total** | **{len(d)}** | |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    A, B = load(res, "clipA"), load(res, "clipB")
    hw = A["ri"]["hardware"]
    gpu = hw["gpus"][0]["name"] + (f" x{len(hw['gpus'])}" if len(hw["gpus"]) > 1 else "")
    tA, tB = A["ri"]["timing"], B["ri"]["timing"]
    oA, oB = A["rs"]["odometry"], B["rs"]["odometry"]
    sA, sB = A["rs"]["summary"], B["rs"]["summary"]
    fA = A["f"]
    n_light = int((fA.kind == "light").sum()) if fA is not None else 0
    n_sign = int((fA.kind == "sign").sum()) if fA is not None else 0
    n_cw = int((fA.kind == "crosswalk").sum()) if fA is not None else 0
    n_prob = int((fA.condition == "possible_problem").sum()) if fA is not None else 0
    mk = (A["ri"].get("furniture") or {}).get("marking", {})
    tpA = A["ri"].get("temporal", {})
    cA, cB = A["ri"]["config"]["detect"], B["ri"]["config"]["detect"]

    num = dict(
        nA=len(A["d"]), nB=len(B["d"]), gpu=gpu, cpu=hw["cpu"], torch=hw["torch"], cuda=hw["cuda"],
        fpsA=tA["processing_fps"], fpsA_all=tA["processing_fps_all_models"], fpsB=tB["processing_fps"],
        fpsB_all=tB["processing_fps_all_models"], lenA=oA["route_length_m_approx"], spdA=oA["mean_speed_kmh_approx"],
        lenB=oB["route_length_m_approx"], spdB=oB["mean_speed_kmh_approx"], sw=sA["sidewalk_present_pct"],
        mkvis=mk.get("markings_visible_pct", 0), n_light=n_light, n_sign=n_sign, n_cw=n_cw, n_prob=n_prob,
        rawA=A["ri"]["tracking"]["raw_tracks"], repA=A["ri"]["tracking"]["reported"],
        rawB=B["ri"]["tracking"]["raw_tracks"], repB=B["ri"]["tracking"]["reported"],
        boosted=tpA.get("boosted_detections", 0), imgsz=cA["imgsz"], conf=cA["conf"],
        tabA=counts_table(A["d"], ("clipA defect type", "unique defects")),
        tabB=counts_table(B["d"], ("clipB defect type", "unique defects")),
        water=sB.get("mean_standing_water_pct_of_roi", 0),
    )
    def jet(clip):
        rec = res / "jetson_recommended.json"
        want = json.loads(rec.read_text())["profile"] if rec.exists() else None
        d = (res / clip / want) if want and (res / clip / want).exists() else next(iter(sorted((res / clip).glob("jetson_*fps"))), None)
        if d is None or not (d / "run_info.json").exists():
            return {}
        ji = json.loads((d / "run_info.json").read_text())
        return {"rate": ji.get("analysis_rate", {}), "t": ji["timing"], "n_frames": ji.get("frames_export", {}).get("frames")}
    jA, jB = jet("clipA"), jet("clipB")
    _rec = res / "jetson_recommended.json"
    _rj = json.loads(_rec.read_text()) if _rec.exists() else {}
    _jd = _rj.get("profile") or next((d.name for d in sorted((res / "clipA").glob("jetson_*fps"))), "jetson_10fps")
    _tested = ", ".join(sorted({d.name[7:-3] for c in ("clipA", "clipB") for d in (res / c).glob("jetson_*fps")}, key=float, reverse=True))
    _met = _rj.get("rule_met", False)
    mt = res / "mode_comparison.md"
    num.update(
        jdir=_jd, tested=_tested,
        jline_en=(f"The recommended profile is `{_jd}`" if _met else f"No tested profile reaches 90 % recall and precision on both clips yet; the best recall is `{_jd}`") + f" (clipA stride {jA.get('rate', {}).get('stride', '?')} = **{jA.get('rate', {}).get('analysis_fps', '?')} FPS**, clipB stride {jB.get('rate', {}).get('stride', '?')} = **{jB.get('rate', {}).get('analysis_fps', '?')} FPS**)",
        jline_ar=(f"البروفايل الموصى بيه `{_jd}`" if _met else f"لسه مفيش بروفايل متجرّب وصل 90% recall و precision في الكليبين؛ أحسن recall هو `{_jd}`") + f" (clipA: stride {jA.get('rate', {}).get('stride', '?')} = **{jA.get('rate', {}).get('analysis_fps', '?')} إطار/ث**، clipB: stride {jB.get('rate', {}).get('stride', '?')} = **{jB.get('rate', {}).get('analysis_fps', '?')} إطار/ث**)",
        mode_table=mt.read_text().strip() if mt.exists() else "*(comparison table is generated by `scripts/compare_modes.py`)*",
        strideA=jA.get("rate", {}).get("stride", "?"), rateA=jA.get("rate", {}).get("analysis_fps", "?"),
        strideB=jB.get("rate", {}).get("stride", "?"), rateB=jB.get("rate", {}).get("analysis_fps", "?"),
        jfpsA=jA.get("t", {}).get("processing_fps", "?"), jfpsB=jB.get("t", {}).get("processing_fps", "?"),
        jallA=jA.get("t", {}).get("processing_fps_all_models", "?"), jallB=jB.get("t", {}).get("processing_fps_all_models", "?"),
        nfA=jA.get("n_frames", "?"), nfB=jB.get("n_frames", "?"),
    )
    for k, fn in (("run_note", "RUN_NOTE.md"), ("run_note_ar", "RUN_NOTE.ar.md")):
        f = res / fn
        num[k] = f.read_text().strip() if f.exists() else ""
    en = (Path(__file__).parent / "readme_template_en.md").read_text()
    ar = (Path(__file__).parent / "readme_template_ar.md").read_text()
    (out / "README.md").write_text(en.format(**num))
    (out / "README.ar.md").write_text(ar.format(**num))
    print("README.md / README.ar.md written")


if __name__ == "__main__":
    main()
