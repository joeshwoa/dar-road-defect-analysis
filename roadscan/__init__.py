"""roadscan: multi-model road-defect analysis for dashcam / low-camera videos.

Pipeline (each stage is a separate, swappable model):
    defect detector  ->  road-surface filter  ->  tracker  ->  track merge
    ->  (optional) SAM 2 masks on each defect's best frame  ->  exports
"""

__version__ = "0.1.0"
