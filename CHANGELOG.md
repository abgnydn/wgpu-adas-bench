# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [0.1.0] — 2026-05-04

First public release. **Full ADAS sensor fusion pipeline — ten stages
fused into a single GPU dispatch via `wgpu-native`, benchmarked against
the equivalent multi-kernel PyTorch implementation on the same hardware.**

### Headline numbers (Apple M2 Pro, Metal, N=30 runs, 5 warmup)

| Config                 | wgpu-native (1 dispatch) | PyTorch MPS (10 kernels) | Speedup    |
| ---------------------- | -----------------------: | --------------------------: | ---------: |
| R=256,  C=50,  T=128   |    **780 fps** (1.28 ms) |             66 fps (15.2 ms) | **11.9×** |
| R=512,  C=100, T=256   |    **763 fps** (1.31 ms) |             50 fps (20.1 ms) | **15.3×** |
| R=1024, C=200, T=512   |    **763 fps** (1.31 ms) |             60 fps (16.6 ms) | **12.7×** |

ADAS budget is 30 fps (33 ms). Fused version runs at **763 fps (1.3 ms)
— 25× the budget**, leaving 96% of the GPU free for neural-network
inference downstream.

### Pipeline

*(Corrected 2026-08-15: this entry originally listed eleven stages, several of
which — LiDAR projection, temporal alignment, Hungarian association, track NMS,
track lifecycle, persistence write-back — were never implemented in any
release. The list below matches the released code.)*

All ten stages run in a single compute shader dispatch:

1. Radar projection (polar → Cartesian → image plane)
2. Cost matrix (radar × camera pairwise distance)
3. Greedy association (atomicMin, no CPU round-trip)
4. Kalman predict (6-state constant acceleration)
5. Kalman update (radar + camera fusion)
6. Classification (RCS + box area + velocity)
7. Lane association
8. Time-to-collision
9. Risk scoring
10. Path planning (collision-aware, keeps worst-case path cost per object)

### Companion projects

- [wgpu-native-bench](https://github.com/abgnydn/wgpu-native-bench) — sister
  Rust harness for Rastrigin / N-Body workloads (vs PyTorch CUDA / MPS).
- [webgpu-kernel-fusion](https://github.com/abgnydn/webgpu-kernel-fusion) —
  the umbrella research line on single-kernel fusion.

[0.1.0]: https://github.com/abgnydn/wgpu-adas-bench/releases/tag/v0.1.0
