"""End-to-end analysis of one video: detect -> filter -> track -> merge -> render/export."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .classes import EXTRA_NAMES, color_for
from .config import Config
from .detectors import Dets, build_detector
from .export import track_record, write_furniture, write_outputs, write_scene, write_surface
from .hardware import StageTimer, collect_hardware, cuda_sync
from .render import VideoSink, draw_box, draw_hud, draw_legend, fill_box, overlay_mask
from .scene import build_scene_detector, scene_finalize, scene_records
from .temporal import DisplaySmoother, TemporalPrior, TrackMemory
from .tracking import DefectTracker, FinalTrack, Obs, finalize_tracks
from .video_io import iter_frames, probe, timecode
from .odometry import EgoMotion
from .mapping import MiniMap, render_map_png, write_points_csv, write_route_csv
from .vehicles import colour_name, vehicle_rows


class Analyzer:
    """Holds the (expensive) models so several videos can be processed with one load."""

    def __init__(self, cfg: Config, class_names: Optional[list[str]] = None, log=print):
        self.cfg = cfg
        self.log = log
        self.detector = build_detector(cfg.detect, class_names)
        self.names: dict[int, str] = dict(getattr(self.detector, "names", {}) or {})
        if class_names and not self.names:
            self.names = {i: n for i, n in enumerate(class_names)}
        self.names.update(EXTRA_NAMES)
        self.surface = None
        if cfg.surface.enabled:
            from .surface import SurfaceAnalyzer

            self.surface = SurfaceAnalyzer(cfg.surface, cfg.detect.device)
        self.openvocab = None
        if cfg.openvocab.enabled:
            from .openvocab import OpenVocabDetector

            self.openvocab = OpenVocabDetector(cfg.openvocab, cfg.detect.device)
        self.furniture = None
        if cfg.furniture.enabled:
            from .furniture import FurnitureAnalyzer

            self.furniture = FurnitureAnalyzer(cfg.furniture, cfg.detect.device, log=log)
        self._fur_log: dict[int, tuple] = {}
        self.prior_stats: dict[str, Any] = {}
        self._ego: EgoMotion | None = None
        self._groups: dict[int, np.ndarray] = {}
        self.road = None
        if cfg.road.enabled:
            from .road_mask import RoadSegmenter

            self.road = RoadSegmenter(cfg.road, cfg.detect.device)
        self.sam = None
        if cfg.sam.enabled:
            from .sam_masks import SamMasker

            self.sam = SamMasker(cfg.sam, cfg.detect.device)
        self.scene_detector = build_scene_detector(cfg) if cfg.scene.enabled else None
        self.scene_names: dict[int, str] = dict(self.scene_detector.names) if self.scene_detector else {}
        self.hardware = collect_hardware()

    # ---------------------------------------------------------------- pass 1
    def _detect_and_track(self, video: Path, info):
        cfg = self.cfg
        tracker = DefectTracker(cfg.track)
        scene_tracker = DefectTracker(cfg.track) if self.scene_detector else None
        timer = StageTimer()
        import copy as _copy
        xw = _copy.deepcopy(cfg.track)
        xw.track_high_thresh, xw.track_low_thresh, xw.new_track_thresh, xw.track_buffer = 0.30, 0.15, 0.35, 30
        xo = _copy.deepcopy(cfg.track)
        xo.track_high_thresh, xo.track_low_thresh, xo.new_track_thresh, xo.track_buffer = 0.16, 0.08, 0.16, 30
        water_tracker = DefectTracker(xw) if self.surface is not None else None
        ov_tracker = DefectTracker(xo) if self.openvocab is not None else None
        fps_ = info.fps_avg or info.fps_nominal or 25.0
        ego = (EgoMotion(cfg.odometry, info.width, info.height, fps=fps_, tcfg=cfg.temporal)
               if cfg.odometry.enabled else None)
        tc = cfg.temporal
        mem = prior = None
        if tc.enabled:      # past-only memory of every tracked defect, fed back into the next detection step
            mem = TrackMemory(alpha=tc.track_alpha, on_thr=tc.on_thr, off_thr=tc.off_thr, min_hits=tc.min_hits,
                              coast=tc.coast_frames)
            prior = TemporalPrior(mem, tc.prior_weight, tc.prior_max_boost, tc.prior_iou,
                                  compatible=cfg.merge.compatible_classes)
        if self.furniture is not None:
            self.furniture.reset(info.width, info.height, fps_)
        self._fur_log = {}
        self._ego = ego
        self._groups = {}
        if self.surface is not None:
            self.surface.reset()
        xlogs: dict[str, dict[int, list]] = {"water": {}, "ov": {}, "fur": {}}
        raw: dict[int, list[Obs]] = {}
        frame_log: dict[int, list[tuple]] = {}   # frame_idx -> [(raw_id, xyxy, conf, cls)]
        scene_raw: dict[int, list[Obs]] = {}
        scene_log: dict[int, list[tuple]] = {}
        n = 0

        # warm-up (cuDNN autotune, lazy init) so FPS is not polluted by one-off costs
        _gen = iter_frames(video)
        first = next(_gen, None)
        _gen.close()
        if first is not None:
            for _ in range(2):
                d = self.detector(first.image)
                if self.road is not None:
                    self.road.filter(d, first.image)
            if self.road is not None:  # warm-up must not leak into stats
                self.road.frames_total = self.road.frames_failed = self.road.dets_in = self.road.dets_dropped = 0
                self.road._mask = None
                self.road._frames_since = 10 ** 9
            cuda_sync()

        t_wall = time.perf_counter()
        for fr in iter_frames(video, cfg.stride, cfg.max_frames):
            with timer.stage("detect"):
                dets = self.detector(fr.image)
            overlap = np.full((len(dets),), np.nan, np.float32)
            if self.road is not None:
                with timer.stage("road_filter"):
                    dets, overlap = self.road.filter(dets, fr.image)
            with timer.stage("track"):
                if prior is not None and len(dets):
                    c2, k2 = prior.adjust(dets.xyxy, dets.conf, dets.cls, fr.idx)
                    dets = Dets(dets.xyxy, c2, k2)
                rows = tracker.update(dets, fr.image)
            log = []
            for row in rows:
                tid, k = int(row[4]), int(row[7])
                if k >= len(dets):
                    continue
                o = Obs(fr.idx, fr.pts, dets.xyxy[k].copy(), float(dets.conf[k]), int(dets.cls[k]), float(overlap[k]))
                raw.setdefault(tid, []).append(o)
                log.append((tid, o.xyxy, o.conf, o.cls))
            if mem is not None:
                mem.update(fr.idx, [(t_, c_, b_, k_) for t_, b_, c_, k_ in log])
                # what is drawn: confirmed tracks only (no one-frame blips), coasting a few frames after a dropout,
                # label = smoothed confidence and voted class
                log = [(t_, b_, e_, c_) for t_, b_, e_, c_, _m in mem.visible_boxes()]
            frame_log[fr.idx] = log
            if self.scene_detector is not None:
                with timer.stage("scene_detect"):
                    sd = self.scene_detector(fr.image)
                with timer.stage("scene_track"):
                    srows = scene_tracker.update(sd, fr.image)
                slog = []
                for row in srows:
                    tid, k = int(row[4]), int(row[7])
                    if k >= len(sd):
                        continue
                    o = Obs(fr.idx, fr.pts, sd.xyxy[k].copy(), float(sd.conf[k]), int(sd.cls[k]))
                    scene_raw.setdefault(tid, []).append(o)
                    slog.append((tid, o.xyxy, o.conf, o.cls))
                scene_log[fr.idx] = slog
            veh_boxes = [tuple(b) for b in sd.xyxy] if (self.scene_detector is not None and len(sd)) else []
            if self.surface is not None:
                with timer.stage("surface"):
                    fresh = self.surface.update(fr.idx, fr.pts, fr.image)
                    wd = Dets.empty()
                    if fresh:
                        self._groups[fr.idx] = cv2.resize(self.surface.group, None, fx=0.5, fy=0.5,
                                                          interpolation=cv2.INTER_NEAREST)
                        if ego is not None:
                            ego.set_horizon(self.surface.horizon_y())
                        wd = self.surface.water_dets(fr.image)
                if fresh:
                    xlogs["water"][fr.idx] = self._feed_extra(water_tracker, wd, fr, 1_000_000, raw)
            if self.openvocab is not None and fr.idx % cfg.openvocab.every_n == 0:
                with timer.stage("openvocab"):
                    od = self.openvocab(fr.image, self.surface.roi_top() if self.surface is not None
                                        else int(0.45 * info.height))
                    if self.road is not None and len(od):
                        od = self._road_filter_quiet(od, fr.image)
                xlogs["ov"][fr.idx] = self._feed_extra(ov_tracker, od, fr, 2_000_000, raw)
            if self.furniture is not None and self.furniture.due(fr.idx):
                with timer.stage("furniture"):
                    hz = self.surface.horizon_y() if self.surface is not None else (
                        ego._horizon if ego is not None else 0.42 * info.height)
                    fres = self.furniture.update(fr.idx, fr.pts, fr.image, hz, ego,
                                                 ego.poses[-1].dist_m if (ego is not None and ego.poses) else None)
                if fres is not None:
                    flog = []
                    for rid, box, conf, cls_ in fres.issue_obs:
                        raw.setdefault(rid, []).append(Obs(fr.idx, fr.pts, box.copy(), float(conf), int(cls_)))
                        flog.append((rid, box, float(conf), int(cls_)))
                    xlogs["fur"][fr.idx] = flog
                    self._fur_log[fr.idx] = (fres.items, fres.marking, fres.marking_mask_half)
            if ego is not None:
                with timer.stage("odometry"):
                    ego.update(fr.idx, fr.pts, fr.image, veh_boxes)
            n += 1
            if n % 100 == 0:
                self.log(f"  [{video.name}] {n} frames processed, {len(raw)} raw tracks")
        cuda_sync()
        wall = time.perf_counter() - t_wall
        self.prior_stats = ({"boosted_detections": prior.n_boosted, "relabelled_detections": prior.n_relabelled,
                             "tracks_ever_confirmed": sum(1 for t_ in mem.tracks.values() if t_.ever_confirmed),
                             "raw_tracks_seen": len(mem.tracks)}
                            if prior is not None else {})
        return raw, frame_log, timer, n, wall, scene_raw, scene_log, xlogs

    @staticmethod
    def _feed_extra(tracker, dets: Dets, fr, offset: int, raw: dict[int, list[Obs]]) -> list[tuple]:
        """Track one of the extra channels (water / open-vocabulary) and log its boxes for rendering."""
        rows = tracker.update(dets, fr.image)
        log = []
        for row in rows:
            tid, k = int(row[4]), int(row[7])
            if k >= len(dets):
                continue
            o = Obs(fr.idx, fr.pts, dets.xyxy[k].copy(), float(dets.conf[k]), int(dets.cls[k]))
            raw.setdefault(tid + offset, []).append(o)
            log.append((tid + offset, o.xyxy, o.conf, o.cls))
        return log

    def _road_filter_quiet(self, dets: Dets, img: np.ndarray) -> Dets:
        r = self.road
        saved = (r.frames_total, r.dets_in, r.dets_dropped)
        out, _ = r.filter(dets, img)
        r.frames_total, r.dets_in, r.dets_dropped = saved
        return out

    def _legend_entries(self, final, segments) -> list:
        """Colour key: only what can appear in this video (classes reported, surface tints, furniture boxes)."""
        from .surface import TINT
        ent = []
        seen_cls = sorted({t.cls for t in final})
        for c in seen_cls:
            name = next(t.cls_name for t in final if t.cls == c)
            ent.append((color_for(c), name.replace("_", " ")))
        if segments and any(sg.get("surface") == "unpaved" for sg in segments):
            ent.append((TINT[2], "unpaved ground (tint)"))
        if self.furniture is not None:
            ent.append(((90, 190, 90), "lane paint / sign / light / crosswalk: OK"))
            ent.append(((60, 60, 235), "worn paint / possible problem (sign, light)"))
        return ent

    @staticmethod
    def _thin(r):
        import copy as _copy
        t = _copy.copy(r)
        t.thickness = 1
        t.font_scale = max(r.font_scale * 0.8, 0.35)
        return t

    # ---------------------------------------------------------------- pass 2
    def _render_and_save(self, video: Path, info, out_dir: Path, final: list[FinalTrack],
                         raw_to_final: dict[int, int], frame_log: dict[int, list[tuple]],
                         run_info: dict[str, Any], scene_final=None, scene_map=None,
                         scene_log=None, xlogs=None, segments=None, minimap=None):
        cfg = self.cfg
        img_dir, crop_dir = out_dir / "images", out_dir / "crops"
        img_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        by_id = {t.track_id: t for t in final}
        best_at: dict[int, list[FinalTrack]] = {}
        for t in final:
            best_at.setdefault(t.best.frame_idx, []).append(t)
        first_at = sorted(t.first_frame for t in final)
        frame_area = info.width * info.height
        records: dict[int, dict[str, Any]] = {}

        sink = None
        if cfg.render.enabled:
            sink = VideoSink(out_dir / "annotated.mp4", info.fps_avg or info.fps_nominal or 25.0,
                             (info.width, info.height), cfg.render)
        last_log: list[tuple] = []
        last_slog: list[tuple] = []
        last_x: dict[str, list] = {"water": [], "ov": [], "fur": []}
        last_fur: tuple = ([], None, None)
        fur_keys = sorted(self._fur_log)
        group_keys = sorted(self._groups)
        ego = self._ego
        speeds = ego.smoothed_speed_kmh() if (ego is not None and ego.poses) else np.zeros(0)
        pose_frames = np.array([p.frame for p in ego.poses]) if (ego is not None and ego.poses) else np.zeros(0)
        veh_colors: dict[int, str] = {}
        scene_by_id = {t.track_id: t for t in (scene_final or [])}
        scene_best_at: dict[int, list] = {}
        for t in (scene_final or []):
            scene_best_at.setdefault(t.best.frame_idx, []).append(t)
        scene_first = sorted(t.first_frame for t in (scene_final or []))
        proc_fps = run_info["timing"]["processing_fps"]
        gpu = (self.hardware.get("gpus") or [{}])[0].get("name") or self.hardware.get("cpu", "CPU")
        n_total = 0
        last_idx = max(frame_log) if frame_log else -1
        rc = cfg.render
        smoother = DisplaySmoother(rc.box_alpha, rc.hold_frames)
        nomark_s, missing_frames = 0.0, 0
        legend = self._legend_entries(final, segments)
        for fr in iter_frames(video):
            if fr.idx > last_idx:       # honour --max-frames: never draw stale boxes past the processed range
                break
            n_total += 1
            if fr.idx in frame_log:
                last_log = frame_log[fr.idx]
            if scene_log is not None and fr.idx in scene_log:
                last_slog = scene_log[fr.idx]
            for key in ("water", "ov", "fur"):
                if xlogs and fr.idx in xlogs[key]:
                    last_x[key] = xlogs[key][fr.idx]
            if fr.idx in self._fur_log:
                last_fur = self._fur_log[fr.idx]
            img = fr.image
            for st in scene_best_at.get(fr.idx, []):           # rough colour of vehicles (no plates, no crops saved)
                x1, y1, x2, y2 = [int(v) for v in st.best.xyxy]
                bw, bh = x2 - x1, y2 - y1
                crop = img[max(y1 + int(0.4 * bh), 0):max(y1 + int(0.9 * bh), 1),
                           max(x1 + int(0.2 * bw), 0):max(x1 + int(0.8 * bw), 1)]
                veh_colors[st.track_id] = colour_name(crop.reshape(-1, 3)) if crop.size else "unknown"
            canvas = img.copy() if sink is not None else None
            if canvas is not None and self.surface is not None and group_keys:   # colour unpaved / water / sidewalk
                gi = int(np.searchsorted(group_keys, fr.idx, side="right")) - 1
                if gi >= 0:
                    g = self._groups[group_keys[gi]]
                    self.surface.overlay(canvas, cv2.resize(g, (info.width, info.height),
                                                            interpolation=cv2.INTER_NEAREST),
                                         alpha=rc.tint_alpha, water=rc.tint_water)
            mk_now = last_fur[1] or {}
            seg_now = _segment_at(segments, fr.pts) if segments else None
            paved_now = bool(seg_now is not None and seg_now.get("surface") == "paved")
            # causal "paint missing on a paved road" timer (past frames only): needs to persist before it is flagged
            if self.furniture is not None and paved_now and not mk_now.get("present"):
                nomark_s += 1.0 / max(info.fps_avg or info.fps_nominal or 25.0, 1.0)
            else:
                nomark_s = max(nomark_s - 2.0 / max(info.fps_avg or info.fps_nominal or 25.0, 1.0), 0.0)
            mark_missing = nomark_s >= rc.missing_marking_s
            if mark_missing:
                missing_frames += 1
            if canvas is not None and last_fur[2] is not None and rc.marking_overlay != "off":
                worn = mk_now.get("state") == "worn"      # paint overlay: green = visible & fine, red = worn / faded
                if worn or rc.marking_overlay == "all":
                    m_full = cv2.resize(last_fur[2], (info.width, info.height), interpolation=cv2.INTER_NEAREST) > 0
                    col = np.array((60, 60, 235) if worn else (80, 210, 80), np.float32)
                    canvas[m_full] = (canvas[m_full] * 0.55 + col * 0.45).astype(np.uint8)
            if canvas is not None:      # context objects: thin muted boxes, drawn under the defects
                for tid, xyxy, conf, _c in last_slog:
                    sid = (scene_map or {}).get(tid)
                    if sid is None:
                        continue
                    st = scene_by_id[sid]
                    if self.furniture is not None and st.cls_name in ("traffic light", "stop sign"):
                        continue        # judged (with condition) by the road-furniture channel instead
                    draw_box(canvas, xyxy, f"{st.cls_name} V{sid}", (170, 150, 110), self._thin(cfg.render))
            if canvas is not None:      # signs / lights / crosswalks that were looked at (problems are drawn as defects)
                for tid, xyxy, kind, label, flagged, bad in last_fur[0]:
                    if flagged and raw_to_final.get(5_000_000 + tid) is not None:
                        continue
                    show_txt = rc.furniture_labels == "all" or (rc.furniture_labels == "flagged" and flagged)
                    draw_box(canvas, xyxy, (f"{label}" + (" ?" if flagged else "")) if show_txt else "",
                             (60, 60, 235) if flagged else (90, 190, 90), self._thin(cfg.render))
            all_log = list(last_log) + last_x["water"] + last_x["ov"] + last_x["fur"]
            # one box per defect, calmed by the past-only display smoother (EMA box, short coast, best-so-far conf)
            seen_now: dict[int, tuple] = {}
            for tid, xyxy, conf, _cls in all_log:
                fid = raw_to_final.get(tid)
                if fid is None:
                    continue
                prev = smoother.last_box(fid)
                better = fid not in seen_now or (
                    (prev is not None and _iou_xyxy(xyxy, prev) > _iou_xyxy(seen_now[fid][0], prev)) or
                    (prev is None and conf > seen_now[fid][1]))
                if better:
                    seen_now[fid] = (xyxy, conf)
            draw_items = smoother.update(fr.idx, seen_now)
            if canvas is not None:      # translucent fill first, boxes on top
                for fid, box, conf in draw_items:
                    c_ = by_id[fid].cls
                    fill_box(canvas, box, color_for(c_), 0.30 if c_ in (3, 4) else 0.16, ellipse=c_ in (3, 4, 5))
                for fid, box, conf in draw_items:
                    t = by_id[fid]
                    name = "pothole+water" if (t.cls_name == "pothole" and t.water_filled) else t.cls_name
                    draw_box(canvas, box, f"#{fid} {name} {conf:.2f}", color_for(t.cls), cfg.render)
            if canvas is not None and cfg.render.hud:
                seen = np.searchsorted(first_at, fr.idx, side="right")
                hud = [f"t={fr.pts:6.2f}s  unique defects so far: {seen}"]
                if scene_final:
                    hud.append(f"road users tracked so far: {int(np.searchsorted(scene_first, fr.idx, side='right'))}")
                if segments:
                    seg = _segment_at(segments, fr.pts)
                    if seg is not None:
                        hud.append(f"road: {seg['surface'].upper()}  |  sidewalk: {seg['sidewalk']}")
                alert = set()
                if mark_missing:
                    hud.append("lane markings: MISSING / not visible - possible problem")
                    alert.add(len(hud) - 1)
                elif last_fur[1] is not None:
                    mk = last_fur[1]
                    hud.append("lane markings: " + (f"{mk['state'].upper()} (vis {mk['vis']:.2f})" if mk.get("present")
                                                     else "not visible"))
                    if mk.get("present") and mk.get("state") == "worn":
                        alert.add(len(hud) - 1)
                if len(speeds) and len(pose_frames):
                    pi = min(int(np.searchsorted(pose_frames, fr.idx)), len(speeds) - 1)
                    hud.append(f"speed ~{speeds[pi]:.0f} km/h (est.)")
                hud.append(f"{proc_fps:.1f} FPS on {gpu}")
                draw_hud(canvas, hud, cfg.render, alert)
                if rc.legend:
                    draw_legend(canvas, legend, cfg.render)
                if minimap is not None:
                    minimap.draw(canvas, fr.idx)
            if sink is not None:
                sink.write(canvas)

            for t in best_at.get(fr.idx, []):
                b = t.best
                color = color_for(t.cls)
                mask = None
                if self.sam is not None:
                    mask = self.sam.mask_for_box(img, b.xyxy)
                shot = overlay_mask(img, mask, color) if mask is not None else img.copy()
                draw_box(shot, b.xyxy, f"#{t.track_id} {t.cls_name} {b.conf:.2f}", color, cfg.render)
                stem = f"defect_{t.track_id:03d}_{t.cls_name}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), shot, [cv2.IMWRITE_JPEG_QUALITY, 92])
                x1, y1, x2, y2 = b.xyxy
                pw, ph = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
                cx1, cy1 = max(int(x1 - pw), 0), max(int(y1 - ph), 0)
                cx2, cy2 = min(int(x2 + pw), info.width), min(int(y2 + ph), info.height)
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size:
                    cv2.imwrite(str(crop_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                records[t.track_id] = track_record(
                    t, f"images/{stem}.jpg", f"crops/{stem}.jpg", frame_area,
                    int(mask.sum()) if mask is not None else None)
        self._missing_marking_s = missing_frames / max(info.fps_avg or info.fps_nominal or 25.0, 1.0)
        if sink is not None:
            sink.close()
        for t in final:  # safety net: never silently lose a reported defect
            if t.track_id not in records:
                records[t.track_id] = track_record(t, "", "", frame_area, None)
        return [records[k] for k in sorted(records)], veh_colors

    # ---------------------------------------------------------------- public
    def run(self, video_path: str | Path, out_root: str | Path) -> dict[str, Any]:
        video = Path(video_path)
        out_dir = Path(out_root) / video.stem.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        info = probe(video)
        self.log(f"[{video.name}] {info.width}x{info.height}, {info.frame_count} frames, "
                 f"{info.duration_s:.1f}s, nominal {info.fps_nominal:.2f} fps, avg {info.fps_avg:.2f} fps"
                 f"{' (variable frame rate)' if info.is_vfr else ''}")

        raw, frame_log, timer, n_frames, wall, scene_raw, scene_log, xlogs = self._detect_and_track(video, info)
        final, raw_to_final, mstats = finalize_tracks(raw, self.names, self.cfg)
        self.log(f"[{video.name}] raw tracks {mstats['raw_tracks']} -> merged {mstats['after_merge']} "
                 f"-> reported {mstats['reported']}")

        scene_final, scene_map, scene_stats = [], {}, None
        if self.scene_detector is not None:
            scene_final, scene_map, scene_stats = scene_finalize(scene_raw, self.scene_names, self.cfg)
            self.log(f"[{video.name}] scene: {scene_stats['reported']} road users / signals tracked")

        ego = self._ego
        # ---- road surface segments + summary
        segments, ssum = [], None
        if self.surface is not None and self.surface.records:
            segments, ssum = self.surface.finalize(info.fps_avg or 25.0)
            if ego is not None and ego.poses:
                pt = np.array([p.t for p in ego.poses])
                pd_ = np.array([p.dist_m for p in ego.poses])
                for sg in segments:
                    sg["distance_start_m_approx"] = round(float(np.interp(sg["start_s"], pt, pd_)), 1)
                    sg["distance_end_m_approx"] = round(float(np.interp(sg["end_s"], pt, pd_)), 1)

        # ---- where was each defect? (flat-ground model + visual odometry)
        world: dict[int, dict[str, Any]] = {}
        markers = []
        if ego is not None and ego.poses:
            for t in final:
                if t.cls in (7, 8):        # sign / traffic-light boxes float above the ground: no ground position
                    continue
                w = ego.world_of_box(t.best.frame_idx, t.best.xyxy)
                if w is None:
                    continue
                pose = ego.pose_at(t.best.frame_idx)
                world[t.track_id] = {"x_m": round(w[0], 1), "y_m": round(w[1], 1), "range_ahead_m": round(w[2], 1),
                                     "route_distance_m": round(pose.dist_m, 1)}
                bgr = color_for(t.cls)
                markers.append({"frame": t.first_frame, "x_m": w[0], "y_m": w[1], "color": bgr})
        minimap = MiniMap(ego, markers) if (ego is not None and ego.poses and self.cfg.odometry.map_inset
                                            and len(ego.poses) > 2) else None

        model_s = sum(timer.totals.get(k, 0.0) for k in ("detect", "road_filter", "track"))
        scene_s = sum(timer.totals.get(k, 0.0) for k in ("scene_detect", "scene_track"))
        extra_s = sum(timer.totals.get(k, 0.0) for k in ("surface", "openvocab", "odometry", "furniture"))
        run_info: dict[str, Any] = {
            "video": video.name,
            "video_info": info.__dict__,
            "frames_processed": n_frames,
            "stride": self.cfg.stride,
            "timing": {
                "processing_fps": round(n_frames / model_s, 2) if model_s > 0 else None,
                "end_to_end_fps_pass1": round(n_frames / wall, 2) if wall > 0 else None,
                "processing_fps_with_scene": round(n_frames / (model_s + scene_s), 2) if scene_s > 0 else None,
                "processing_fps_all_models": round(n_frames / (model_s + scene_s + extra_s), 2)
                if (scene_s + extra_s) > 0 else None,
                "pass1_wall_s": round(wall, 3),
                "stages": timer.summary(n_frames),
                "note": ("processing_fps = defect detector + road filter + tracking only (decode/render/export excluded); "
                         "processing_fps_all_models adds the scene detector, surface segmentation, open-vocabulary "
                         "detector and visual odometry; end_to_end_fps_pass1 additionally includes decode/Python overhead"),
            },
            "tracking": mstats,
            "temporal": {"enabled": self.cfg.temporal.enabled, **self.prior_stats,
                         "note": "all per-frame states use past frames only (EMA / hysteresis / dwell / rate limits); "
                                 "the final merged report is computed after the run"},
            "scene_tracking": scene_stats,
            "road_filter": self.road.stats() if self.road is not None else None,
            "road_surface": ssum,
            "odometry": ego.stats() if ego is not None else None,
            "hardware": self.hardware,
            "config": self.cfg.to_dict(),
            "class_names": self.names,
        }
        t0 = time.perf_counter()
        records, veh_colors = self._render_and_save(
            video, info, out_dir, final, raw_to_final, frame_log, run_info, scene_final, scene_map,
            scene_log if self.scene_detector else None, xlogs, segments, minimap)
        run_info["timing"]["render_export_s"] = round(time.perf_counter() - t0, 3)

        # ---- enrich the defect records (location, surface, sidewalk)
        by_type_in_seg: dict[int, Counter] = {sg["segment_id"]: Counter() for sg in segments}
        for rec in records:
            w = world.get(rec["track_id"], {})
            rec.update({"map_x_m": w.get("x_m", ""), "map_y_m": w.get("y_m", ""),
                        "route_distance_m": w.get("route_distance_m", ""), "range_ahead_m": w.get("range_ahead_m", "")})
            sg = _segment_at(segments, rec["first_seen_s"])
            rec["surface_at_defect"] = sg["surface"] if sg else ""
            rec["sidewalk_at_defect"] = sg["sidewalk"] if sg else ""
            if sg:
                by_type_in_seg[sg["segment_id"]][rec["defect_type"]] += 1
        for sg in segments:
            c = by_type_in_seg[sg["segment_id"]]
            sg["defects_total"] = sum(c.values())
            sg["defects_by_type"] = ";".join(f"{k}:{v}" for k, v in sorted(c.items()))
        # ---- road furniture: markings, crosswalks, signs, traffic lights
        if self.furniture is not None:
            inv = self.furniture.inventory()
            msum = self.furniture.marking_summary()
            run_info["furniture"] = {"inventory_items": len(inv), "marking": msum}
            write_furniture(out_dir, inv, msum, self.furniture.mark_hist)
            if ssum is not None:
                ssum.setdefault("findings", [])
                if msum["updates_with_visible_markings"]:
                    ssum["findings"].append(
                        f"Lane markings visible in {msum['markings_visible_pct']}% of analysed frames; "
                        f"{msum['episodes_worn']} worn/faded stretch(es) flagged "
                        f"(worn for {msum.get('worn_time_pct', 0)}% of the time markings were visible).")
                else:
                    ssum["findings"].append("No lane markings were visible on this route.")
                miss = float(getattr(self, "_missing_marking_s", 0.0))
                if miss > 0:
                    ssum["findings"].append(
                        f"Paved road without visible lane paint for {miss:.1f} s of the video (flagged on screen as "
                        f"'MISSING - possible problem' after {cfg.render.missing_marking_s:.0f} s without paint).")
                    msum["missing_paint_s_on_paved"] = round(miss, 1)
                bad = [r for r in inv if r["condition"] != "ok"]
                kinds = {k: sum(1 for r in inv if r["kind"] == k) for k in ("sign", "light", "crosswalk")}
                ssum["findings"].append(
                    f"Road furniture seen: {kinds['sign']} sign(s), {kinds['light']} traffic light(s), "
                    f"{kinds['crosswalk']} crosswalk(s); {len(bad)} flagged as a possible problem.")
        write_outputs(out_dir, records, run_info)
        if ssum is not None:
            ssum["defects_total_on_unpaved_or_mixed"] = int(sum(1 for r in records if r["surface_at_defect"] in ("unpaved", "mixed")))
            write_surface(out_dir, segments, ssum, ego.stats() if ego is not None else None)

        # ---- road users (bonus) with vehicle log
        vrows = []
        if self.scene_detector is not None:
            vrows = vehicle_rows(scene_final, ego, info.width, info.height, veh_colors) if ego is not None else []
            base = scene_records(scene_final, info.width)
            vmap = {v["vehicle_id"]: v for v in vrows}
            for r in base:
                v = vmap.get(r["track_id"])
                if v:
                    r.update({k: val for k, val in v.items() if k not in ("vehicle_id", "type", "first_seen_s",
                                                                        "first_seen_timecode", "last_seen_s",
                                                                        "max_confidence", "n_frames_detected")
                              and not k.startswith("_")})
            write_scene(out_dir, base)

        # ---- route map
        if ego is not None and len(ego.poses) > 2:
            speeds = ego.smoothed_speed_kmh()
            write_route_csv(out_dir / "route.csv", ego, speeds,
                            lambda t_: (_segment_at(segments, t_) or {}).get("surface", ""))
            pts = []
            for rec in records:
                if rec["map_x_m"] != "":
                    pts.append({"kind": "defect", "id": rec["track_id"], "type": rec["defect_type"],
                                "time_s": rec["first_seen_s"], "x_m": rec["map_x_m"], "y_m": rec["map_y_m"],
                                "distance_along_route_m": rec["route_distance_m"], "range_ahead_m": rec["range_ahead_m"]})
            for v in vrows:
                if v["passed_by_ego"] == "yes" and v["map_x_m"] != "":
                    pts.append({"kind": "vehicle", "id": v["vehicle_id"], "type": v["type"], "time_s": v["closest_time_s"]
                                if "closest_time_s" in v else v["first_seen_s"], "x_m": v["map_x_m"], "y_m": v["map_y_m"],
                                "distance_along_route_m": "", "range_ahead_m": v["closest_distance_m_approx"]})
            write_points_csv(out_dir / "map_points.csv", pts)
            route = ego.route()
            rs = [(_segment_at(segments, p.t) or {}).get("surface", "unknown") for p in ego.poses]
            title = (f"{video.stem}: route {ego.stats()['route_length_m_approx']:.0f} m (approx.), "
                     f"{len([p for p in pts if p['kind'] == 'defect'])} defects / objects")
            render_map_png(out_dir / "route_map.png", route, rs, pts, title,
                           f"Monocular visual odometry, assumed camera height {self.cfg.odometry.cam_height_m} m and "
                           f"{self.cfg.odometry.hfov_deg:.0f} deg HFOV - schematic, distances are approximate.")
        self.log(f"[{video.name}] done: {len(records)} unique defects, "
                 f"{run_info['timing']['processing_fps']} FPS (model stages) -> {out_dir}")
        return {"out_dir": str(out_dir), "records": records, "run_info": run_info}


def _segment_at(segments: list[dict[str, Any]], t: float):
    for sg in segments:
        if sg["start_s"] <= t < sg["end_s"] + 1e-6:
            return sg
    return segments[-1] if segments and t >= segments[-1]["end_s"] else (segments[0] if segments else None)


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
