"""Typed configuration with YAML loading and dotted-key overrides."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DetectCfg:
    backend: str = "ultralytics"        # "ultralytics" (YOLO26/11/8) or "rfdetr"
    weights: str = "weights/yolo26s_rdd.pt"
    imgsz: int = 640
    conf: float = 0.20                  # low on purpose: tracker + report filter decide
    iou: float = 0.60
    device: str = "auto"                # "auto", "cpu", "0", "cuda:0", "mps"
    half: bool = True                   # FP16 on CUDA
    max_det: int = 150
    classes: Optional[list[int]] = None  # restrict to these class ids (None = all)
    tile_size: int = 0                  # >0 enables tiled inference (px, in original-frame pixels)
    tile_overlap: float = 0.25
    rfdetr_variant: str = "small"       # nano|small|medium|base|large
    rfdetr_resolution: int = 640
    enhance: str = "auto"               # off | clahe | auto: also run on a CLAHE-enhanced copy (auto = only on dark frames)
    enhance_lum_thr: float = 125.0      # auto: mean luminance of the lower (road) half below this -> ensemble


@dataclass
class RoadCfg:
    enabled: bool = True
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    every_n: int = 3                    # refresh the road mask every N frames
    min_overlap: float = 0.25           # fraction of a detection box that must lie on road
    dilate_px: int = 21                 # dilate road mask to keep edge-of-road potholes
    min_road_fraction: float = 0.04     # if road < this share of the frame, assume the mask failed
    input_width: int = 1024             # segmenter input width (aspect kept)


@dataclass
class TrackCfg:
    tracker: str = "botsort"            # botsort | bytetrack
    track_high_thresh: float = 0.30
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.35
    track_buffer: int = 45
    match_thresh: float = 0.80
    gmc_method: str = "sparseOptFlow"   # camera-motion compensation (moving camera)
    with_reid: bool = False


@dataclass
class MergeCfg:
    enabled: bool = True
    max_gap_frames: int = 40            # max gap between end of A and start of B
    iou_thr: float = 0.20               # IoU between A's extrapolated box and B's first box
    dist_factor: float = 0.60           # or centre distance <= factor * max(box side)
    size_ratio: float = 2.5             # max area ratio between predicted A and first B box
    dup_iou: float = 0.50               # coexisting tracks with IoU >= this are duplicates
    dup_frac: float = 0.50              # ... on at least this share of shared frames
    # class ids that may be merged across (e.g. [[0, 1, 2]] lets crack types merge)
    compatible_classes: list[list[int]] = field(default_factory=list)


@dataclass
class ReportCfg:
    min_observations: int = 3           # a defect must be detected in >= N frames
    min_max_conf: float = 0.40          # ... and reach at least this confidence once
    class_min_conf: dict = field(default_factory=lambda: {4: 0.50, 5: 0.24, 6: 0.45, 7: 0.50, 8: 0.60, 9: 0.50})   # per-class override of min_max_conf {class_id: value}
    class_min_obs: dict = field(default_factory=lambda: {4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2})    # per-class override of min_observations


@dataclass
class SamCfg:
    enabled: bool = False
    weights: str = "sam2.1_b.pt"


@dataclass
class SceneCfg:
    """Bonus: second detector (COCO) for vehicles / pedestrians / signals, tracked with the same tracker."""
    enabled: bool = False
    weights: str = "yolo26s.pt"
    imgsz: int = 640
    conf: float = 0.30
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 9, 11])  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
    min_observations: int = 5
    min_max_conf: float = 0.45
    compatible_classes: list[list[int]] = field(default_factory=lambda: [[2, 5, 7]])  # car/bus/truck flicker


@dataclass
class SurfaceCfg:
    """Road-surface analysis: paved / unpaved / sidewalk / standing water (ADE20K SegFormer)."""
    enabled: bool = True
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    every_n: int = 2
    input_width: int = 672
    roi_top_margin: float = 0.03        # analysed region starts this far (frame heights) below the horizon
    water_detect: bool = True           # turn standing water on unpaved/damaged road into water_filled_pothole candidates
    water_min_area_frac: float = 0.0005
    water_max_area_frac: float = 0.20
    bright_puddles: bool = True         # also pick up small bright sky-reflecting puddles the segmenter misses
    sidewalk_min_frac: float = 0.008    # share of a side's road-level area that must be sidewalk
    smooth_s: float = 1.5
    causal: bool = True                 # trailing (past-only) smoothing + hysteresis/dwell instead of centred windows
    dwell_s: float = 1.5                # a surface / sidewalk state may change only after the new value lasted this long
    min_segment_s: float = 2.0
    unpaved_thr: float = 0.60
    paved_thr: float = 0.25
    tint: bool = True


@dataclass
class OdometryCfg:
    """Monocular visual odometry for the 2-D route map (approximate: assumed camera height + field of view)."""
    enabled: bool = True
    hfov_deg: float = 75.0
    cam_height_m: float = 1.4
    horizon_frac: float = 0.0           # 0 = estimate from the road segmentation, else fixed fraction of frame height
    map_inset: bool = True


@dataclass
class OpenVocabCfg:
    """Open-vocabulary detector (OWLv2) for things RDD2022 has no class for: manholes and drain grates."""
    enabled: bool = False
    model_id: str = "google/owlv2-base-patch16-ensemble"
    prompts: dict = field(default_factory=lambda: {"a manhole cover": 5, "a storm drain grate": 5})
    every_n: int = 3
    conf: float = 0.16
    max_area_frac: float = 0.04


@dataclass
class FurnitureCfg:
    """Lane markings, crosswalks, traffic signs and traffic lights + their condition (roadscan.furniture)."""
    enabled: bool = False
    model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    clip_enabled: bool = True
    clip_id: str = "openai/clip-vit-base-patch32"
    every_n: int = 4                    # run the segmenter every N frames (it is the heavy part)
    short_side: int = 576               # input short side (frames are small; upsampling helps thin paint / far signs)
    track_max_age: int = 4              # updates a lost item survives
    min_area_px: dict = field(default_factory=lambda: {"sign": 90, "light": 30, "crosswalk": 400})
    min_obs: dict = field(default_factory=lambda: {"sign": 3, "light": 4, "crosswalk": 3})
    inventory_min_obs: int = 3
    bad_on: dict = field(default_factory=lambda: {"sign": 0.55, "light": 0.70, "crosswalk": 0.60})
    bad_off: dict = field(default_factory=lambda: {"sign": 0.35, "light": 0.40, "crosswalk": 0.40})
    # markings
    marking_near_m: float = 4.0
    marking_far_m: float = 30.0
    marking_dwell_s: float = 1.5
    marking_worn_thr: float = 0.50
    marking_ok_thr: float = 0.62
    marking_abs_weber: float = 0.10     # absolute contrast floor (paint barely brighter than asphalt)
    # signs
    lean_ok_deg: float = 8.0
    clip_min_px: int = 40               # zero-shot CLIP is meaningless on tiny crops (verified: 10x24 px gives noise)
    clip_bad_floor: float = 0.50
    # lights
    light_min_h_px: int = 14
    light_dark_obs: int = 8             # updates in view with no lit lamp before "dark" counts


@dataclass
class TemporalCfg:
    """Causal (past-only) stabilisation, see roadscan.temporal."""
    enabled: bool = True
    prior_weight: float = 0.35          # blend of the previous track confidence into the current detection
    prior_max_boost: float = 0.15
    prior_iou: float = 0.30
    track_alpha: float = 0.4
    on_thr: float = 0.35
    off_thr: float = 0.20
    min_hits: int = 2
    coast_frames: int = 3               # keep drawing a confirmed box this many frames after a dropout
    speed_median_s: float = 1.6      # trailing window of the speed filter (s)
    speed_ema: float = 0.2
    max_accel_ms2: float = 2.5
    max_decel_ms2: float = 4.5
    surface_dwell_s: float = 1.5        # paved/unpaved/sidewalk state may change only after this long
    surface_smooth_s: float = 1.5       # trailing (past-only) smoothing window


@dataclass
class RenderCfg:
    enabled: bool = True
    thickness: int = 2
    font_scale: float = 0.5
    hud: bool = True
    encode_h264: bool = True            # re-encode with ffmpeg for universal playback
    crf: int = 20
    hold_frames: int = 8                # a defect box coasts this many frames after a dropout (display only, past-only)
    box_alpha: float = 0.5              # EMA weight of the newest box (calms jitter of the drawn rectangle)
    legend: bool = True                 # colour key drawn on the video
    tint_alpha: float = 0.16            # strength of the unpaved / sidewalk ground tint
    tint_water: bool = False            # standing water already has its own defect box; the extra blue tint confuses
    marking_overlay: str = "all"        # off | worn | all: lane-paint overlay (green = fine, red = worn/faded)
    missing_marking_s: float = 3.0      # paved road with no visible paint for this long (past only) -> flagged
    save_frames: str = "data"          # all (per-frame json+csv + CLEAN original frames with boxes) | data (json+csv only) | off
    frame_jpeg_quality: int = 75
    furniture_labels: str = "flagged"   # none | flagged | all: text on signs / lights (thin boxes are always drawn)


@dataclass
class Config:
    detect: DetectCfg = field(default_factory=DetectCfg)
    road: RoadCfg = field(default_factory=RoadCfg)
    track: TrackCfg = field(default_factory=TrackCfg)
    merge: MergeCfg = field(default_factory=MergeCfg)
    report: ReportCfg = field(default_factory=ReportCfg)
    sam: SamCfg = field(default_factory=SamCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    surface: SurfaceCfg = field(default_factory=SurfaceCfg)
    odometry: OdometryCfg = field(default_factory=OdometryCfg)
    openvocab: OpenVocabCfg = field(default_factory=OpenVocabCfg)
    furniture: FurnitureCfg = field(default_factory=FurnitureCfg)
    temporal: TemporalCfg = field(default_factory=TemporalCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    stride: int = 1                     # process every Nth frame (speed knob, e.g. on Jetson)
    target_fps: float = 0.0             # >0: choose the stride per video so the models see ~this rate (Jetson mode)
    max_frames: int = 0                 # 0 = whole video

    # ---- io helpers -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Config":
        cfg = cls()
        if d:
            _update(cfg, d)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def override(self, dotted: dict[str, Any]) -> "Config":
        """Return a copy with {"detect.conf": 0.3, "stride": 2}-style overrides applied."""
        new = copy.deepcopy(self)
        for key, val in dotted.items():
            obj: Any = new
            *parents, leaf = key.split(".")
            for p in parents:
                obj = getattr(obj, p)
            if not hasattr(obj, leaf):
                raise KeyError(f"unknown config key: {key}")
            setattr(obj, leaf, val)
        return new


def _update(obj: Any, d: dict[str, Any]) -> None:
    valid = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in valid:
            raise KeyError(f"unknown config key '{k}' for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        else:
            setattr(obj, k, v)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """['detect.conf=0.3', 'road.enabled=false'] -> typed dict (YAML scalar parsing)."""
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"override must look like key=value, got: {p}")
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
