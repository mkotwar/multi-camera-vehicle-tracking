# Scalability Benchmark

## Test Conditions

- Frames per camera: `30`
- Ingestion workers: `3`
- Detection workers: `1`
- YOLO model instances: `1`
- Colour workers: `1`
- Florence model instances: `1`
- Logical camera counts: `2, 4, 8, 12`
- Source reuse mode: one independent reader handle per logical camera, same local video path reused safely

## Comparison

| Cameras | Frames | Runtime (s) | Pipeline FPS | YOLO FPS | Colour Queue Peak | Cache Misses | Bottleneck |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 60/60 | 53.299 | 1.305 | 7.179 | 30/100 (30.0%) | 0 | YOLO |
| 4 | 120/120 | 84.198 | 1.573 | 10.140 | 52/100 (52.0%) | 0 | YOLO |
| 8 | 240/240 | 145.187 | 1.741 | 12.942 | 95/100 (95.0%) | 0 | FLORENCE |
| 12 | 360/360 | 206.862 | 1.801 | 14.296 | 100/100 (100.0%) | 0 | FLORENCE |

## Findings

- Saturation starts at: `8` cameras
- Primary bottleneck: `FLORENCE`
- Secondary bottleneck: `YOLO`
- Recommended next optimization: `F. optimize Florence`
