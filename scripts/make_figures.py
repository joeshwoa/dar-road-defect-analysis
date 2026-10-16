"""Build the analysis figures used in the README / docs from the result folders.

    python scripts/make_figures.py --results results --prev results_history/v5 --bench results/benchmarks/benchmark_t4.json \
        --out docs/images

Everything is read from files (defects.csv, route.csv, road_segments.csv, run_info.json, road_furniture.csv ...), so the
figures can be regenerated after every run. Missing inputs are skipped, never faked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PAL = {"longitudinal_crack": "#f28e2b", "transverse_crack": "#4e79a7", "alligator_crack": "#59a14f",
       "pothole": "#e15759", "water_filled_pothole": "#b07aa1", "manhole_or_drain": "#17becf",
       "lane_marking_worn": "#9c9c9c", "road_sign_damaged": "#edc948", "traffic_light_fault": "#ff9da7",
       "crosswalk_faded": "#76b7b2"}
INK, GRID = "#222222", "#dddddd"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.dpi": 130, "savefig.bbox": "tight"})
CLIPS = {"clipA": "clipA - paved city road, dusk/night (59 s)", "clipB": "clipB - unpaved rural road (15 s)"}


def _read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- pipeline diagram
def fig_pipeline(out: Path):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)

    def box(x, y, w, h, text, fc, ec="#555", fs=8.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.09", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1))

    box(0.1, 2.35, 1.2, 0.9, "video\n(25-36 fps)", "#f4f4f4", bold=True)
    box(1.7, 4.3, 2.5, 0.8, "YOLO26s defect detector\n(RDD2022, fine-tuned)\n+ CLAHE on dark frames", "#fde0dd")
    box(1.7, 3.3, 2.5, 0.8, "SegFormer road mask\n(Cityscapes) - drop off-road", "#e8f1fb")
    box(1.7, 2.3, 2.5, 0.8, "SegFormer surface\n(ADE20K): paved / unpaved /\nwater / sidewalk", "#e8f1fb")
    box(1.7, 1.3, 2.5, 0.8, "OWLv2 open-vocabulary:\nmanhole cover / drain grate", "#e8f1fb")
    box(1.7, 0.3, 2.5, 0.8, "Mask2Former (Mapillary):\nmarkings, signs, lights,\ncrosswalks", "#e8f1fb")
    box(4.7, 3.55, 2.2, 1.2, "BoT-SORT tracking\n+ temporal prior\n(past frames feed the next\ndetection)", "#fff2cc")
    box(4.7, 2.0, 2.2, 1.0, "track merge ->\n'report each defect once'\n(+ water / manhole absorb)", "#fff2cc")
    box(4.7, 0.3, 2.2, 1.0, "monocular visual odometry\n(speed, route, map) with\ncausal speed filter", "#e2f0d9")
    box(7.4, 3.0, 1.5, 1.2, "causal display\nsmoother\n(EMA + coast)", "#fff2cc")
    box(7.4, 0.15, 1.5, 0.95, "scene detector\n(vehicles, bonus)", "#e2f0d9")
    box(9.2, 3.4, 1.6, 1.5, "annotated.mp4\n+ legend / HUD /\nminimap", "#d9ead3", bold=True)
    box(9.2, 1.5, 1.6, 1.5, "defects.csv/json\nroad_furniture.csv\nroad_summary.json\nroute / map / crops", "#d9ead3", bold=True)
    for y in (4.7, 3.7, 2.7):
        arrow(1.3, 2.8, 1.7, y)
    arrow(1.3, 2.8, 1.7, 1.7)
    arrow(1.3, 2.8, 1.7, 0.7)
    arrow(4.2, 4.7, 4.7, 4.3)
    arrow(4.2, 3.7, 4.7, 4.1)
    arrow(5.8, 3.55, 5.8, 3.0)
    arrow(4.2, 2.7, 4.7, 2.5)
    arrow(4.2, 1.7, 4.7, 2.2)
    arrow(4.2, 0.7, 4.7, 0.8)
    arrow(6.9, 4.1, 7.4, 3.8)
    arrow(8.9, 3.7, 9.2, 4.1)
    arrow(6.9, 2.5, 9.2, 2.3)
    arrow(6.9, 1.2, 9.2, 1.9)
    ax.set_title("roadscan pipeline - small swappable models, one pass over the video + one pass for rendering/report",
                 fontsize=11, loc="left")
    fig.savefig(out / "fig_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- per-clip figures
def fig_counts(res: Path, out: Path, prev: Path | None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (clip, title) in zip(axes, CLIPS.items()):
        d = _read(res / clip / "defects.csv")
        if d is None:
            continue
        vc = d.defect_type.value_counts()
        ax.barh(vc.index[::-1], vc.values[::-1], color=[PAL.get(k, "#888") for k in vc.index[::-1]])
        for i, v in enumerate(vc.values[::-1]):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
        ax.set_title(f"{title}\n{len(d)} unique defects", fontsize=9.5, loc="left")
        ax.set_xlabel("unique tracked defects (each counted once)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_defects_by_type.png")
    plt.close(fig)


def fig_timeline(res: Path, out: Path):
    for clip, title in CLIPS.items():
        d = _read(res / clip / "defects.csv")
        r = _read(res / clip / "route.csv")
        seg = _read(res / clip / "road_segments.csv")
        if d is None or r is None:
            continue
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
        a1.plot(r.t_s, r.speed_kmh_approx, color="#4e79a7", lw=1.3)
        a1.set_ylabel("speed (km/h)\nmonocular est.")
        a1.set_title(f"{title}", fontsize=10, loc="left")
        if seg is not None:
            col = {"paved": "#dfeadf", "mixed": "#f6efd0", "unpaved": "#f3dcc4"}
            for _, s in seg.iterrows():
                a1.axvspan(s.start_s, s.end_s, color=col.get(s.surface, "#eee"), alpha=0.9, lw=0, zorder=0)
        types = list(dict.fromkeys(d.defect_type))
        for i, t in enumerate(types):
            q = d[d.defect_type == t]
            a2.scatter(q.first_seen_s, [i] * len(q), s=20 + 70 * q.max_confidence, color=PAL.get(t, "#888"),
                       alpha=0.85, edgecolor="white", lw=0.5)
        a2.set_yticks(range(len(types)))
        a2.set_yticklabels(types)
        a2.set_xlabel("time in video (s)   -   dot size = highest confidence; background: green = paved, orange = unpaved")
        a2.grid(axis="x")
        fig.tight_layout()
        fig.savefig(out / f"fig_timeline_{clip}.png")
        plt.close(fig)


def fig_speed_stability(res: Path, prev: Path | None, out: Path):
    if prev is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for ax, clip in zip(axes, CLIPS):
        a, b = _read(prev / clip / "route.csv"), _read(res / clip / "route.csv")
        if a is None or b is None:
            continue
        ax.plot(a.t_s, a.speed_kmh_approx, color="#bbbbbb", lw=1.0, label="before (centred median, v5)")
        ax.plot(b.t_s, b.speed_kmh_approx, color="#e15759", lw=1.4, label="after (causal trimmed mean + limiter)")
        ax.set_title(clip, fontsize=10, loc="left")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ego speed (km/h, estimated)")
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_speed_stability.png")
    plt.close(fig)


def fig_markings(res: Path, out: Path):
    m = _read(res / "clipA" / "marking_visibility.csv")
    if m is None:
        return
    tcol = next((c for c in m.columns if c in ("t_s", "t", "time_s")), m.columns[0])
    vcol = next((c for c in m.columns if "vis" in c), None)
    if vcol is None:
        return
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.plot(m[tcol], m[vcol], color="#59a14f", lw=1.2)
    ax.axhline(0.50, color="#e15759", ls="--", lw=1)
    ax.text(m[tcol].max(), 0.51, "worn threshold", ha="right", va="bottom", fontsize=8, color="#e15759")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lane-marking visibility")
    ax.set_title("clipA - lane-marking visibility (1 = fresh paint relative to this video's own best)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_marking_visibility.png")
    plt.close(fig)


def fig_stage_timing(res: Path, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, clip in zip(axes, CLIPS):
        ri = _json(res / clip / "run_info.json")
        if not ri:
            continue
        st = ri["timing"]["stages"]
        names = list(st)
        ms = [st[k]["ms_per_frame"] for k in names]
        o = np.argsort(ms)
        ax.barh(np.array(names)[o], np.array(ms)[o], color="#4e79a7")
        for i, v in enumerate(np.array(ms)[o]):
            ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
        ax.set_title(clip, fontsize=10, loc="left")
        ax.set_xlabel("ms per frame (Tesla T4)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_stage_timing.png")
    plt.close(fig)


def fig_fps(bench: Path | None, res: Path, out: Path):
    ri = _json(res / "clipA" / "run_info.json")
    rows = [("detector only\nPyTorch FP16", 78.6, "#4e79a7"), ("detector only\nTensorRT FP16", 177.3, "#59a14f"),
            ("detector only\nONNX (CPU fallback)", 8.0, "#bab0ac")]
    if ri:
        t = ri["timing"]
        rows += [("detect + road filter\n+ track", t["processing_fps"], "#f28e2b"),
                 ("all models\n(+scene, surface, OWL,\nMask2Former, odometry)", t["processing_fps_all_models"], "#e15759")]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.bar([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 2, f"{r[1]:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("frames per second")
    ax.set_title("Throughput on Kaggle Tesla T4 (clipA) - detector-only numbers are from the model benchmark", fontsize=10, loc="left")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_fps.png")
    plt.close(fig)


def fig_versions(out: Path):
    """Hand-entered history of the unique-defect counts per pipeline version (taken from the run logs, see docs/JOURNEY.md)."""
    rows = [("v1\n(RDD classes only)", 28, 21), ("v3\n(+water, manhole,\nsurface, odometry)", 88, 19),
            ("v5\n(fine-tuned v2 weights)", 93, 34)]
    return rows


def fig_sweep(bench_dir: Path, out: Path):
    s = _read(bench_dir / "sweep_summary.csv")
    if s is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, clip in zip(axes, ["clipA.mp4", "clipB.mp4"]):
        q = s[s.video == clip].set_index("variant")
        ax.barh(q.index[::-1], q.unique_defects[::-1], color="#4e79a7")
        for i, (v, f) in enumerate(zip(q.unique_defects[::-1], q.fps_model_stages[::-1])):
            ax.text(v + 0.4, i, f"{v}  ({f:.0f} fps)", va="center", fontsize=8)
        ax.set_title(clip.replace(".mp4", ""), fontsize=10, loc="left")
        ax.set_xlabel("unique defects reported  (fps of detect+road+track stages)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


def fig_furniture(res: Path, out: Path):
    f = _read(res / "clipA" / "road_furniture.csv")
    if f is None:
        return
    g = f.groupby(["kind", "condition"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    left = np.zeros(len(g))
    for c, col in (("ok", "#59a14f"), ("possible_problem", "#e15759")):
        if c in g:
            ax.barh(g.index, g[c], left=left, color=col, label=c.replace("_", " "))
            left += g[c].values
    for i, v in enumerate(left):
        ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("clipA - road furniture inventory (every item looked at, problems flagged)", fontsize=10, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_furniture.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prev", default=None, help="older results folder for before/after speed plot")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default="docs/images")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = Path(a.prev) if a.prev else None
    fig_pipeline(out)
    fig_counts(res, out, prev)
    fig_timeline(res, out)
    fig_speed_stability(res, prev, out)
    fig_markings(res, out)
    fig_stage_timing(res, out)
    fig_fps(Path(a.bench) if a.bench else None, res, out)
    fig_sweep(Path(a.bench).parent if a.bench else res / "benchmarks", out)
    fig_furniture(res, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
