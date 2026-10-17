| clip | profile (cap) | analysis rate | stride | defects (reference) | defects (Jetson mode) | matched | only reference | only Jetson | recall | precision | core FPS (T4) | all-models FPS (T4) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clipA | 10 FPS | 9.004 FPS of 36.015 | 4 | 53 | 11 | 11 | 42 | 0 | 21% | 100% | 17.3 | 3.04 |
| clipB | 10 FPS | 12.5 FPS of 25.0 | 2 | 102 | 55 | 54 | 48 | 1 | 53% | 98% | 17.18 | 2.66 |

**No tested profile reaches 90 % recall and precision on both clips; best recall: `jetson_10fps`** (worst-clip recall 21%, precision 98%).
