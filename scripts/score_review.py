#!/usr/bin/env python3
"""Turn a human-labelled review.csv into precision / duplicate-rate numbers for the README.

    python scripts/score_review.py outputs/<video_name>/review.csv [more review.csv ...]

`unsure` rows are excluded from precision and reported separately. These are precision-style numbers on the
*reported* defects only: recall cannot be measured without labelling the video frames themselves.
"""
from __future__ import annotations

import sys

import pandas as pd


def main() -> None:
    frames = []
    for p in sys.argv[1:]:
        d = pd.read_csv(p)
        d["source"] = p
        frames.append(d)
    if not frames:
        sys.exit(__doc__)
    df = pd.concat(frames)
    df["verdict"] = df["verdict"].fillna("").str.strip().str.lower()
    lab = df[df["verdict"].isin(["tp", "fp", "dup"])]
    print(f"labelled {len(lab)} of {len(df)} rows ({(df['verdict'] == 'unsure').sum()} unsure)\n")

    def row(g):
        n = len(g)
        return pd.Series({
            "n": n, "tp": (g.verdict == "tp").sum(), "fp": (g.verdict == "fp").sum(), "dup": (g.verdict == "dup").sum(),
            "precision_%": round(100 * (g.verdict == "tp").sum() / n, 1) if n else None,
            "duplicate_rate_%": round(100 * (g.verdict == "dup").sum() / n, 1) if n else None,
        })

    print("overall:\n", row(lab).to_string(), "\n")
    print("per class:\n", lab.groupby("defect_type").apply(row, include_groups=False).to_string())
    print("\nper video:\n", lab.groupby("source").apply(row, include_groups=False).to_string())


if __name__ == "__main__":
    main()
