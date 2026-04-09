"""PyTorch sensor fusion baseline — full ADAS pipeline, same workload as wgpu-native.
Simulates the typical multi-kernel pipeline:
  1. Radar projection (polar → image plane)
  2. Cost matrix (radar × camera pairwise distances)
  3. Greedy association (argmin per camera box)
  4. Kalman predict (6-state constant acceleration)
  5. Kalman update (fused measurement)
  6. Object classification (RCS + box size → class)
  7. Lane association (lateral position → lane ID)
  8. Time-to-collision
  9. Collision risk scoring
 10. Path planning (evaluate 16 candidate trajectories per object)

Each step is a separate tensor operation = separate kernel dispatch.
Run on the same GPU as wgpu-native for apples-to-apples comparison."""

import torch
import time
import math
import sys

WARMUP = 5
RUNS = 30

# Auto-detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
    sync = torch.cuda.synchronize
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    sync = torch.mps.synchronize
    print(f"PyTorch: {torch.__version__}, Backend: MPS")
else:
    print("No GPU available")
    sys.exit(1)

print()

# Constants
FOCAL_X, FOCAL_Y = 1000.0, 1000.0
CX, CY = 960.0, 540.0
VELOCITY_WEIGHT = 2.0
GATE_THRESHOLD = 100.0
PROCESS_NOISE = 0.5
MEAS_NOISE_POS = 2.0
MEAS_NOISE_VEL = 1.0
EGO_VX = 25.0  # ~90 km/h
LANE_WIDTH = 3.7
N_LANES = 4
N_PATH_CANDIDATES = 16
PATH_HORIZON = 3.0
PATH_STEPS = 10


def bench_sensor_fusion(n_radar, n_camera, n_tracks):
    dt = 0.033

    # Generate synthetic data
    radar_range = torch.rand(n_radar, device=device) * 145.0 + 5.0
    radar_az = (torch.rand(n_radar, device=device) - 0.5) * 1.0
    radar_el = (torch.rand(n_radar, device=device) - 0.5) * 0.2
    radar_vel = (torch.rand(n_radar, device=device) - 0.5) * 60.0
    radar_rcs = (torch.rand(n_radar, device=device) - 0.25) * 40.0

    cam_cx = torch.rand(n_camera, device=device) * 1920.0
    cam_cy = torch.rand(n_camera, device=device) * 1080.0
    cam_w = torch.rand(n_camera, device=device) * 270.0 + 30.0
    cam_h = torch.rand(n_camera, device=device) * 270.0 + 30.0
    cam_vx = (torch.rand(n_camera, device=device) - 0.5) * 60.0

    # Track state: [x, y, vx, vy, ax, ay]
    state = torch.zeros(n_tracks, 6, device=device)
    state[:, 0] = torch.rand(n_tracks, device=device) * 1920.0
    state[:, 1] = torch.rand(n_tracks, device=device) * 1080.0
    state[:, 2] = (torch.rand(n_tracks, device=device) - 0.5) * 20.0
    state[:, 3] = (torch.rand(n_tracks, device=device) - 0.5) * 20.0
    P = torch.tensor([10.0, 10.0, 5.0, 5.0, 1.0, 1.0], device=device).unsqueeze(0).expand(n_tracks, -1).clone()

    def run():
        nonlocal state, P

        # ── Kernel 1: Radar projection ──
        r = radar_range
        x_world = r * torch.cos(radar_el) * torch.sin(radar_az)
        y_world = r * torch.cos(radar_el) * torch.cos(radar_az)
        z_world = r * torch.sin(radar_el)
        inv_z = 1.0 / torch.clamp(z_world + y_world, min=0.001)
        proj_x = FOCAL_X * x_world * inv_z + CX
        proj_y = FOCAL_Y * z_world * inv_z + CY

        # ── Kernel 2: Cost matrix ──
        dx = proj_x.unsqueeze(1) - cam_cx.unsqueeze(0)
        dy = proj_y.unsqueeze(1) - cam_cy.unsqueeze(0)
        spatial_dist = torch.sqrt(dx ** 2 + dy ** 2)
        half_w = cam_w * 0.5 + 20.0
        half_h = cam_h * 0.5 + 20.0
        inside = (torch.abs(dx) < half_w.unsqueeze(0)) & (torch.abs(dy) < half_h.unsqueeze(0))
        vel_cost = torch.abs(radar_vel.unsqueeze(1) - cam_vx.unsqueeze(0)) * VELOCITY_WEIGHT
        cost = spatial_dist + vel_cost + (~inside).float() * 1000.0

        # ── Kernel 3: Association ──
        min_costs, best_cam = cost.min(dim=1)
        valid = min_costs < GATE_THRESHOLD

        # ── Kernel 4: Kalman predict ──
        dt2 = 0.5 * dt * dt
        state[:, 0] += state[:, 2] * dt + state[:, 4] * dt2
        state[:, 1] += state[:, 3] * dt + state[:, 5] * dt2
        state[:, 2] += state[:, 4] * dt
        state[:, 3] += state[:, 5] * dt
        P[:, 0] += P[:, 2] * dt ** 2 + PROCESS_NOISE
        P[:, 1] += P[:, 3] * dt ** 2 + PROCESS_NOISE
        P[:, 2] += P[:, 4] * dt ** 2 + PROCESS_NOISE * 0.1
        P[:, 3] += P[:, 5] * dt ** 2 + PROCESS_NOISE * 0.1
        P[:, 4] += PROCESS_NOISE * 0.01
        P[:, 5] += PROCESS_NOISE * 0.01

        # ── Kernel 5: Kalman update ──
        track_ids = torch.arange(n_radar, device=device) % n_tracks
        if valid.any():
            m_track = track_ids[valid]
            meas_x = proj_x[valid]
            meas_y = proj_y[valid]
            meas_vx = radar_vel[valid]
            innov_x = meas_x - state[m_track, 0]
            innov_y = meas_y - state[m_track, 1]
            innov_vx = meas_vx - state[m_track, 2]
            s_x = P[m_track, 0] + MEAS_NOISE_POS
            s_y = P[m_track, 1] + MEAS_NOISE_POS
            s_vx = P[m_track, 2] + MEAS_NOISE_VEL
            k_x = P[m_track, 0] / s_x
            k_y = P[m_track, 1] / s_y
            k_vx = P[m_track, 2] / s_vx
            state[m_track, 0] += k_x * innov_x
            state[m_track, 1] += k_y * innov_y
            state[m_track, 2] += k_vx * innov_vx
            P[m_track, 0] *= (1.0 - k_x)
            P[m_track, 1] *= (1.0 - k_y)
            P[m_track, 2] *= (1.0 - k_vx)

        # ── Kernel 6: Object classification ──
        # Get matched camera box sizes
        matched_w = cam_w[best_cam]
        matched_h = cam_h[best_cam]
        area = matched_w * matched_h
        abs_vel = torch.abs(radar_vel)

        obj_class = torch.zeros(n_radar, device=device)
        obj_conf = torch.full((n_radar,), 0.3, device=device)

        # Truck
        truck_mask = (radar_rcs > 20.0) & (area > 40000.0)
        obj_class[truck_mask] = 2.0
        obj_conf[truck_mask] = 0.85 + 0.1 * torch.clamp((radar_rcs[truck_mask] - 20.0) / 10.0, 0.0, 1.0)

        # Car
        car_mask = (~truck_mask) & (radar_rcs > 5.0) & (area > 5000.0)
        obj_class[car_mask] = 1.0
        obj_conf[car_mask] = 0.8 + 0.15 * torch.clamp((radar_rcs[car_mask] - 5.0) / 15.0, 0.0, 1.0)

        # Motorcycle
        moto_mask = (~truck_mask) & (~car_mask) & (radar_rcs > 0.0) & (area < 5000.0) & (abs_vel > 5.0)
        obj_class[moto_mask] = 5.0
        obj_conf[moto_mask] = 0.6 + 0.2 * torch.clamp(abs_vel[moto_mask] / 20.0, 0.0, 1.0)

        # Bicycle
        bike_mask = (~truck_mask) & (~car_mask) & (~moto_mask) & (radar_rcs > -5.0) & (area < 3000.0) & (abs_vel < 10.0)
        obj_class[bike_mask] = 4.0
        obj_conf[bike_mask] = 0.55

        # Pedestrian
        ped_mask = (~truck_mask) & (~car_mask) & (~moto_mask) & (~bike_mask) & (radar_rcs > -10.0) & (area < 2000.0) & (abs_vel < 3.0)
        obj_class[ped_mask] = 3.0
        obj_conf[ped_mask] = 0.7

        # ── Kernel 7: Lane association ──
        shifted = x_world + N_LANES * LANE_WIDTH * 0.5
        lane_f = shifted / LANE_WIDTH
        lane_id = torch.clamp(lane_f, 0.0, N_LANES - 1.0).floor()
        lane_center = (lane_id + 0.5) * LANE_WIDTH - N_LANES * LANE_WIDTH * 0.5
        lateral_offset = x_world - lane_center

        # ── Kernel 8: Time-to-collision ──
        closing_vel = EGO_VX - radar_vel
        ttc = torch.where(closing_vel > 0.1, radar_range / closing_vel, torch.tensor(99.0, device=device))

        # ── Kernel 9: Collision risk ──
        ttc_risk = torch.exp(-ttc * 0.5)
        class_weight = torch.ones(n_radar, device=device)
        class_weight[obj_class == 3.0] = 2.0   # pedestrian
        class_weight[obj_class == 4.0] = 1.8   # bicycle
        class_weight[obj_class == 5.0] = 1.5   # motorcycle
        lane_risk = torch.exp(-torch.abs(lateral_offset) * 0.5)
        risk = torch.clamp(ttc_risk * class_weight * lane_risk * obj_conf, 0.0, 1.0)

        # ── Kernel 10: Path planning (16 candidates × 10 steps per object) ──
        dt_path = PATH_HORIZON / PATH_STEPS
        # Generate candidate lateral offsets and accelerations
        p_idx = torch.arange(N_PATH_CANDIDATES, device=device, dtype=torch.float32)
        lateral_cands = (p_idx / (N_PATH_CANDIDATES - 1) - 0.5) * 6.0  # -3m to +3m
        accel_cands = ((p_idx % 4) / 3.0 - 0.5) * 4.0 - 1.0  # varying accel

        # For each radar detection × each path candidate × each step
        # [n_radar, N_PATH_CANDIDATES]
        obj_x_exp = x_world.unsqueeze(1).expand(-1, N_PATH_CANDIDATES)
        obj_y_exp = y_world.unsqueeze(1).expand(-1, N_PATH_CANDIDATES)

        track_vx = state[track_ids, 2].unsqueeze(1).expand(-1, N_PATH_CANDIDATES)
        track_vy = state[track_ids, 3].unsqueeze(1).expand(-1, N_PATH_CANDIDATES)

        ego_x = lateral_cands.unsqueeze(0).expand(n_radar, -1)
        ego_y = torch.zeros(n_radar, N_PATH_CANDIDATES, device=device)
        ego_v = torch.full((n_radar, N_PATH_CANDIDATES), EGO_VX, device=device)

        o_x = obj_x_exp.clone()
        o_y = obj_y_exp.clone()
        o_vx = track_vx.clone()
        o_vy = track_vy.clone()

        path_cost = torch.zeros(n_radar, N_PATH_CANDIDATES, device=device)
        min_dist = torch.full((n_radar, N_PATH_CANDIDATES), 1e9, device=device)

        # Class weight for proximity penalty
        prox_weight = torch.ones(n_radar, device=device)
        prox_weight[obj_class == 3.0] = 3.0
        prox_weight[obj_class == 4.0] = 2.5
        prox_weight_exp = prox_weight.unsqueeze(1).expand(-1, N_PATH_CANDIDATES)

        accel_exp = accel_cands.unsqueeze(0).expand(n_radar, -1)

        for s in range(PATH_STEPS):
            ego_v = ego_v + accel_exp * dt_path
            ego_v = torch.clamp(ego_v, min=0.0)
            ego_y = ego_y + ego_v * dt_path

            o_x = o_x + o_vx * dt_path
            o_y = o_y + o_vy * dt_path

            d_x = ego_x - o_x
            d_y = ego_y - o_y
            dist = torch.sqrt(d_x ** 2 + d_y ** 2)
            min_dist = torch.minimum(min_dist, dist)

            close_mask = dist < 10.0
            path_cost += close_mask.float() * prox_weight_exp * (10.0 - dist) ** 2

        # Comfort penalties
        path_cost += torch.abs(lateral_cands).unsqueeze(0) * 2.0
        path_cost += torch.clamp(-accel_exp - 1.0, min=0.0) * 5.0

        # Collision penalty
        path_cost += (min_dist < 2.0).float() * 10000.0

        # Best path per radar detection
        best_path_cost, best_path_id = path_cost.min(dim=1)

        sync()
        return best_path_cost.sum().item()

    # Warmup
    for _ in range(WARMUP):
        run()

    # Benchmark
    times = []
    for _ in range(RUNS):
        sync()
        state[:, 0] = torch.rand(n_tracks, device=device) * 1920.0
        state[:, 1] = torch.rand(n_tracks, device=device) * 1080.0
        P = torch.tensor([10.0, 10.0, 5.0, 5.0, 1.0, 1.0], device=device).unsqueeze(0).expand(n_tracks, -1).clone()
        sync()
        t = time.perf_counter()
        run()
        times.append((time.perf_counter() - t) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    fps = 1000.0 / mean
    print(f"  Sensor Fusion (R={n_radar}, C={n_camera}, T={n_tracks}): "
          f"{mean:.3f} ms/frame (±{std:.3f})  {fps:.1f} fps  (N={RUNS})")


print("─── ADAS Full Pipeline (11 stages fused → 1 dispatch) ───")
bench_sensor_fusion(256, 50, 128)
bench_sensor_fusion(512, 100, 256)
bench_sensor_fusion(1024, 200, 512)

print()
print("Pipeline stages (each = separate kernel dispatch in PyTorch):")
print("  1. Radar projection    2. Cost matrix       3. Association")
print("  4. Kalman predict      5. Kalman update      6. Classification")
print("  7. Lane association    8. Time-to-collision  9. Risk scoring")
print("  10. Path planning (16 candidates × 10 steps)")
print()
