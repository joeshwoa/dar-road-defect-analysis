#!/usr/bin/env python3
"""Compare the full-rate reference run with the Jetson-mode run (same code, models see ~10 FPS).

usage: python scripts/compare_modes.py results            # reads results/clip*/{defects.csv,run_info.json} and results/clip*/jetson_*fps/
Writes results/mode_comparison.md and results/clip*/mode_comparison.json
A defect of the Jetson run "matches" a reference defect when the class is the same and the visible time intervals overlap
(+/- 0.75 s slack). Everything unmatched is listed, nothing is hidden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SLACK = 0.75


def load(d: Path):
    df = pd.read_csv(d / "defects.csv")
    ri = json.loads((d / "run_info.json").read_text())
    return df, ri


def match(ref: pd.DataFrame, jet: pd.DataFrame):
    used, pairs = set(), []
    for i, r in ref.iterrows():
        for j, q in jet.iterrows():
            if j in used or q.defect_type != r.defect_type:
                continue
            if q.first_seen_s <= r.last_seen_s + SLACK and r.first_seen_s <= q.last_seen_s + SLACK:
                used.add(j)
                pairs.append((i, j))
                break
    return pairs, sorted(set(jet.index) - used)


def one(clip: Path, jd: Path, ref, ri):
    jet, jri = load(jd)
    pairs, jet_only = match(ref, jet)
    ref_only = sorted(set(ref.index) - {i for i, _ in pairs})
    rate = jri.get("analysis_rate", {})
    t = jri["timing"]
    rec = {"profile": jd.name, "analysis_rate": rate, "reference_defects": len(ref), "jetson_defects": len(jet),
           "matched": len(pairs), "recall": round(len(pairs) / max(len(ref), 1), 3),
           "precision": round(len(pairs) / max(len(jet), 1), 3),
           "reference_only": ref.loc[ref_only, ["track_id", "defect_type", "first_seen_s"]].to_dict("records"),
           "jetson_only": jet.loc[jet_only, ["track_id", "defect_type", "first_seen_s"]].to_dict("records"),
           "by_class_reference": ref.defect_type.value_counts().to_dict(),
           "by_class_jetson": jet.defect_type.value_counts().to_dict(),
           "timing_reference": {k: ri["timing"].get(k) for k in ("processing_fps", "processing_fps_with_scene", "processing_fps_all_models")},
           "timing_jetson": {k: t.get(k) for k in ("processing_fps", "processing_fps_with_scene", "processing_fps_all_models")}}
    return rec


def main(root: str) -> None:
    root = Path(root)
    lines = ["| clip | profile (cap) | analysis rate | stride | defects (reference) | defects (Jetson mode) | matched | only reference | only Jetson | "
             "recall | precision | core FPS (T4) | all-models FPS (T4) |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    allrec = {}
    for clip in sorted(p for p in root.glob("clip*") if p.is_dir()):
        jdirs = sorted(clip.glob("jetson_*fps"), key=lambda d: -float(d.name[7:-3]))
        if not jdirs:
            continue
        ref, ri = load(clip)
        recs = [one(clip, jd, ref, ri) for jd in jdirs]
        (clip / "mode_comparison.json").write_text(json.dumps(recs, indent=2, default=float))
        allrec[clip.name] = recs
        for r in recs:
            rate, t = r["analysis_rate"], r["timing_jetson"]
            lines.append(f"| {clip.name} | {r['profile'][7:-3]} FPS | {rate.get('analysis_fps')} FPS of {rate.get('video_fps')} | {rate.get('stride')} | "
                         f"{r['reference_defects']} | {r['jetson_defects']} | {r['matched']} | {r['reference_defects'] - r['matched']} | "
                         f"{r['jetson_defects'] - r['matched']} | {r['recall']:.0%} | {r['precision']:.0%} | {t.get('processing_fps')} | {t.get('processing_fps_all_models')} |")
    # recommended profile = the cheapest cap whose worst-clip recall AND precision are both >= 90 % (else the best recall)
    caps = sorted({r["profile"] for recs in allrec.values() for r in recs}, key=lambda n: float(n[7:-3]))
    def worst(cap, key):
        return min(r[key] for recs in allrec.values() for r in recs if r["profile"] == cap)
    ok = [c for c in caps if worst(c, "recall") >= 0.90 and worst(c, "precision") >= 0.90]
    best = ok[0] if ok else max(caps, key=lambda c: (worst(c, "recall"), float(c[7:-3])), default=None)
    if best:
        (root / "jetson_recommended.json").write_text(json.dumps(
            {"profile": best, "rule_met": bool(ok), "worst_clip_recall": worst(best, "recall"), "worst_clip_precision": worst(best, "precision"),
             "rule": "cheapest frame-rate cap with recall >= 90% and precision >= 90% on both clips"}, indent=2))
        lines.append("")
        lead = "Recommended Jetson profile" if ok else "No tested profile reaches 90 % recall and precision on both clips; best recall"
        lines.append(f"**{lead}: `{best}`** (worst-clip recall {worst(best, 'recall'):.0%}, precision {worst(best, 'precision'):.0%}).")
    (root / "mode_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
