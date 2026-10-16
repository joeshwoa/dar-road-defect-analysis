"""Defect class scheme (RDD2022) and drawing colours."""
from __future__ import annotations

# RDD2022 label codes -> training class id
RDD_CODE_TO_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
DEFECT_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
CRACK_IDS = [0, 1, 2]
POTHOLE_ID = 3

# Classes produced by the extra (non-trained) channels of the pipeline
WATER_POTHOLE_ID = 4      # water-filled pothole / puddle on a damaged or unpaved road (segmentation based)
MANHOLE_ID = 5            # manhole cover / drain grate (open-vocabulary detector)
MARKING_WORN_ID = 6       # lane marking that is faded / worn (paint visibility, roadscan.furniture)
SIGN_DAMAGED_ID = 7       # road sign that is leaning, bent, faded or hidden (possible problem)
SIGNAL_FAULT_ID = 8       # traffic light whose lamp stays dark or whose housing leans (possible problem)
CROSSWALK_FADED_ID = 9    # pedestrian crossing with faded paint
EXTRA_NAMES = {WATER_POTHOLE_ID: "water_filled_pothole", MANHOLE_ID: "manhole_or_drain",
               MARKING_WORN_ID: "lane_marking_worn", SIGN_DAMAGED_ID: "road_sign_damaged",
               SIGNAL_FAULT_ID: "traffic_light_fault", CROSSWALK_FADED_ID: "crosswalk_faded"}

# Free-text names found in extra datasets -> class id (lower-cased, stripped)
NAME_ALIASES = {
    "d00": 0, "longitudinal crack": 0, "longitudinal_crack": 0, "longitudinal": 0,
    "d10": 1, "transverse crack": 1, "transverse_crack": 1, "transverse": 1,
    "d20": 2, "alligator crack": 2, "alligator_crack": 2, "alligator": 2,
    "d40": 3, "pothole": 3, "potholes": 3, "pot-hole": 3, "pot_hole": 3,
}


def normalise_name_to_id(raw: str) -> int | None:
    """Map a raw annotation label to a defect class id (None = ignore)."""
    key = raw.strip().lower()
    if key in NAME_ALIASES:
        return NAME_ALIASES[key]
    if "pothole" in key or "pot hole" in key:  # "water-filled pothole", "dry pothole", ...
        return 3
    return None


# Distinct, colour-blind-friendly BGR palette (cycled for unknown class counts)
_PALETTE = [
    (0, 165, 255),   # orange
    (255, 160, 40),  # blue-ish
    (60, 200, 60),   # green
    (60, 60, 240),   # red
    (200, 80, 200),  # magenta
    (200, 200, 40),  # cyan-ish
    (140, 140, 140),
    (40, 220, 220),
]


def color_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(class_id) % len(_PALETTE)]
