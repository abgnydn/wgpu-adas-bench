// ADAS Sensor Fusion — Full fused pipeline
// Single dispatch replaces 11 CUDA kernel launches:
//   1. Radar projection (polar → image plane)
//   2. Cost matrix (radar × camera distance)
//   3. Greedy association (atomicMin assignment)
//   4. Kalman predict (6×6 state transition)
//   5. Kalman update (fused measurement)
//   6. Object classification (RCS + box size → class)
//   7. Lane association (lateral position → lane ID)
//   8. Time-to-collision (range / closing velocity)
//   9. Collision risk scoring (TTC + lane + class weights)
//  10. Path planning — evaluate N candidate trajectories per object
//  11. Risk aggregation — worst-case across all objects
//
// Each thread handles one radar detection end-to-end.

struct Params {
    n_radar: u32,
    n_camera: u32,
    n_tracks: u32,
    dt: f32,
    focal_x: f32,
    focal_y: f32,
    cx: f32,
    cy: f32,
    ego_vx: f32,        // ego vehicle velocity x (m/s)
    ego_vy: f32,        // ego vehicle velocity y (m/s)
    lane_width: f32,    // lane width in meters
    n_lanes: u32,       // number of lanes
}

struct RadarDet {
    range: f32,
    azimuth: f32,
    elevation: f32,
    velocity: f32,
    rcs: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

struct CameraBox {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    confidence: f32,
    est_vx: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Track {
    x: f32, y: f32, vx: f32, vy: f32, ax: f32, ay: f32,
    age: f32, is_active: f32,
    p00: f32, p11: f32, p22: f32, p33: f32, p44: f32, p55: f32,
    p01: f32, p23: f32,
    innovation_x: f32, innovation_y: f32,
    matched_camera: f32, matched_radar: f32,
}

// Extended output per radar detection — full pipeline result
struct FusionResult {
    // Projection
    proj_x: f32,
    proj_y: f32,
    // Association
    matched_camera_id: f32,
    match_cost: f32,
    // Track state (post-Kalman)
    track_x: f32,
    track_y: f32,
    track_vx: f32,
    track_vy: f32,
    // Classification
    obj_class: f32,      // 0=unknown, 1=car, 2=truck, 3=pedestrian, 4=bicycle, 5=motorcycle
    obj_confidence: f32,
    // Lane
    lane_id: f32,
    lateral_offset: f32, // meters from lane center
    // Collision
    ttc: f32,            // time-to-collision (seconds)
    collision_risk: f32, // 0.0 = safe, 1.0 = imminent
    // Path planning
    best_path_cost: f32,
    best_path_id: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> radar: array<RadarDet>;
@group(0) @binding(2) var<storage, read> camera: array<CameraBox>;
@group(0) @binding(3) var<storage, read_write> tracks: array<Track>;
@group(0) @binding(4) var<storage, read_write> results: array<FusionResult>;
@group(0) @binding(5) var<storage, read_write> assignments: array<atomic<u32>>;

const MAX_CAMERA: u32 = 256u;
const VELOCITY_WEIGHT: f32 = 2.0;
const GATE_THRESHOLD: f32 = 100.0;
const PROCESS_NOISE: f32 = 0.5;
const MEAS_NOISE_POS: f32 = 2.0;
const MEAS_NOISE_VEL: f32 = 1.0;
const N_PATH_CANDIDATES: u32 = 16u;  // trajectory candidates per object
const PATH_HORIZON: f32 = 3.0;       // seconds ahead
const PATH_STEPS: u32 = 10u;         // integration steps per trajectory

// ─── Phase 1: Radar → Image projection ───

fn project_radar(det: RadarDet) -> vec2<f32> {
    let r = det.range;
    let x_world = r * cos(det.elevation) * sin(det.azimuth);
    let y_world = r * cos(det.elevation) * cos(det.azimuth);
    let z_world = r * sin(det.elevation);
    let inv_z = 1.0 / max(z_world + y_world, 0.001);
    let img_x = params.focal_x * x_world * inv_z + params.cx;
    let img_y = params.focal_y * z_world * inv_z + params.cy;
    return vec2<f32>(img_x, img_y);
}

// Radar to world coordinates (for path planning)
fn radar_to_world(det: RadarDet) -> vec2<f32> {
    let r = det.range;
    let x = r * cos(det.elevation) * sin(det.azimuth);
    let y = r * cos(det.elevation) * cos(det.azimuth);
    return vec2<f32>(x, y);
}

// ─── Phase 3: Pack cost + radar_id ───

fn pack_assignment(cost: f32, radar_id: u32) -> u32 {
    let cost_q = u32(clamp(cost, 0.0, 65535.0));
    return (cost_q << 16u) | (radar_id & 0xFFFFu);
}

// ─── Phase 4: Kalman predict ───

fn kalman_predict(t: ptr<function, Track>, dt: f32) {
    let dt2 = 0.5 * dt * dt;
    (*t).x += (*t).vx * dt + (*t).ax * dt2;
    (*t).y += (*t).vy * dt + (*t).ay * dt2;
    (*t).vx += (*t).ax * dt;
    (*t).vy += (*t).ay * dt;

    (*t).p00 += (*t).p22 * dt * dt + PROCESS_NOISE;
    (*t).p11 += (*t).p33 * dt * dt + PROCESS_NOISE;
    (*t).p22 += (*t).p44 * dt * dt + PROCESS_NOISE * 0.1;
    (*t).p33 += (*t).p55 * dt * dt + PROCESS_NOISE * 0.1;
    (*t).p44 += PROCESS_NOISE * 0.01;
    (*t).p55 += PROCESS_NOISE * 0.01;
}

// ─── Phase 5: Kalman update ───

fn kalman_update(t: ptr<function, Track>, meas_x: f32, meas_y: f32, meas_vx: f32) {
    let innov_x = meas_x - (*t).x;
    let innov_y = meas_y - (*t).y;
    let innov_vx = meas_vx - (*t).vx;

    let s_x = (*t).p00 + MEAS_NOISE_POS;
    let s_y = (*t).p11 + MEAS_NOISE_POS;
    let s_vx = (*t).p22 + MEAS_NOISE_VEL;

    let k_x = (*t).p00 / s_x;
    let k_y = (*t).p11 / s_y;
    let k_vx = (*t).p22 / s_vx;

    (*t).x += k_x * innov_x;
    (*t).y += k_y * innov_y;
    (*t).vx += k_vx * innov_vx;

    (*t).p00 *= (1.0 - k_x);
    (*t).p11 *= (1.0 - k_y);
    (*t).p22 *= (1.0 - k_vx);

    (*t).innovation_x = innov_x;
    (*t).innovation_y = innov_y;
    (*t).age = 0.0;
}

// ─── Phase 6: Object classification ───
// RCS + bounding box area → object class
// Real systems use neural nets; this is the rule-based fallback/post-processing

fn classify_object(rcs: f32, box_w: f32, box_h: f32, velocity: f32) -> vec2<f32> {
    let area = box_w * box_h;
    let abs_vel = abs(velocity);

    // Classification logic based on radar cross section + visual size
    var obj_class: f32 = 0.0;  // unknown
    var confidence: f32 = 0.3;

    // Truck: large RCS + large box
    if (rcs > 20.0 && area > 40000.0) {
        obj_class = 2.0;
        confidence = 0.85 + 0.1 * clamp((rcs - 20.0) / 10.0, 0.0, 1.0);
    }
    // Car: medium RCS + medium box
    else if (rcs > 5.0 && area > 5000.0) {
        obj_class = 1.0;
        confidence = 0.8 + 0.15 * clamp((rcs - 5.0) / 15.0, 0.0, 1.0);
    }
    // Motorcycle: medium RCS + small box + fast
    else if (rcs > 0.0 && area < 5000.0 && abs_vel > 5.0) {
        obj_class = 5.0;
        confidence = 0.6 + 0.2 * clamp(abs_vel / 20.0, 0.0, 1.0);
    }
    // Bicycle: small RCS + small box + slow
    else if (rcs > -5.0 && area < 3000.0 && abs_vel < 10.0) {
        obj_class = 4.0;
        confidence = 0.55;
    }
    // Pedestrian: small RCS + small box + very slow
    else if (rcs > -10.0 && area < 2000.0 && abs_vel < 3.0) {
        obj_class = 3.0;
        confidence = 0.7;
    }

    return vec2<f32>(obj_class, confidence);
}

// ─── Phase 7: Lane association ───

fn compute_lane(world_x: f32) -> vec2<f32> {
    // Lateral position → lane ID + offset from center
    let shifted = world_x + f32(params.n_lanes) * params.lane_width * 0.5;
    let lane_f = shifted / params.lane_width;
    let lane_id = floor(clamp(lane_f, 0.0, f32(params.n_lanes) - 1.0));
    let lane_center = (lane_id + 0.5) * params.lane_width - f32(params.n_lanes) * params.lane_width * 0.5;
    let offset = world_x - lane_center;
    return vec2<f32>(lane_id, offset);
}

// ─── Phase 8: Time-to-collision ───

fn compute_ttc(obj_range: f32, obj_velocity: f32, ego_vx: f32) -> f32 {
    // Closing velocity = ego speed - object radial velocity
    // Negative velocity = approaching
    let closing_vel = ego_vx - obj_velocity;

    if (closing_vel <= 0.1) {
        return 99.0;  // not approaching → no collision risk
    }
    return obj_range / closing_vel;
}

// ─── Phase 9: Collision risk scoring ───

fn compute_risk(ttc: f32, obj_class: f32, lane_offset: f32, confidence: f32) -> f32 {
    // TTC component: exponential decay — closer = higher risk
    let ttc_risk = exp(-ttc * 0.5);

    // Class weight: pedestrians/cyclists get higher risk multiplier
    var class_weight: f32 = 1.0;
    if (obj_class == 3.0) { class_weight = 2.0; }       // pedestrian
    else if (obj_class == 4.0) { class_weight = 1.8; }   // bicycle
    else if (obj_class == 5.0) { class_weight = 1.5; }   // motorcycle

    // Lane proximity: objects in ego lane are higher risk
    let lane_risk = exp(-abs(lane_offset) * 0.5);

    // Combined risk
    return clamp(ttc_risk * class_weight * lane_risk * confidence, 0.0, 1.0);
}

// ─── Phase 10: Path planning (evaluate candidate trajectories) ───
// For each object, evaluate N candidate ego trajectories and score them

fn evaluate_paths(
    obj_x: f32, obj_y: f32, obj_vx: f32, obj_vy: f32,
    obj_class: f32, collision_risk: f32
) -> vec2<f32> {
    // Evaluate N_PATH_CANDIDATES trajectories
    // Each trajectory: ego lateral offset + longitudinal acceleration
    var best_cost: f32 = 1e9;
    var best_path: f32 = 0.0;

    let dt_path = PATH_HORIZON / f32(PATH_STEPS);

    for (var p: u32 = 0u; p < N_PATH_CANDIDATES; p++) {
        // Candidate: lateral offset from -3m to +3m, accel from -3 to +1 m/s²
        let lateral = (f32(p) / f32(N_PATH_CANDIDATES - 1u) - 0.5) * 6.0;
        let accel = (f32(p % 4u) / 3.0 - 0.5) * 4.0 - 1.0;

        // Simulate ego + object forward for PATH_STEPS
        var ego_x: f32 = 0.0;
        var ego_y: f32 = 0.0;
        var ego_v: f32 = params.ego_vx;
        var o_x: f32 = obj_x;
        var o_y: f32 = obj_y;
        var o_vx: f32 = obj_vx;
        var o_vy: f32 = obj_vy;

        var path_cost: f32 = 0.0;
        var min_dist: f32 = 1e9;

        for (var s: u32 = 0u; s < PATH_STEPS; s++) {
            // Ego moves forward + lateral shift
            ego_v += accel * dt_path;
            ego_v = max(ego_v, 0.0);
            ego_y += ego_v * dt_path;
            ego_x = lateral;  // constant lateral offset for this candidate

            // Object moves with constant velocity
            o_x += o_vx * dt_path;
            o_y += o_vy * dt_path;

            // Distance to object
            let dx = ego_x - o_x;
            let dy = ego_y - o_y;
            let dist = sqrt(dx * dx + dy * dy);
            min_dist = min(min_dist, dist);

            // Proximity penalty (inverse square)
            if (dist < 10.0) {
                var prox_weight: f32 = 1.0;
                if (obj_class == 3.0) { prox_weight = 3.0; }  // extra penalty near pedestrians
                else if (obj_class == 4.0) { prox_weight = 2.5; }
                path_cost += prox_weight * (10.0 - dist) * (10.0 - dist);
            }
        }

        // Comfort penalty: penalize large lateral offset + harsh braking
        path_cost += abs(lateral) * 2.0;
        path_cost += max(-accel - 1.0, 0.0) * 5.0;

        // Collision penalty
        if (min_dist < 2.0) {
            path_cost += 10000.0;
        }

        if (path_cost < best_cost) {
            best_cost = path_cost;
            best_path = f32(p);
        }
    }

    return vec2<f32>(best_cost, best_path);
}

// ─── Main: Full fused ADAS pipeline ───

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_radar) { return; }

    let det = radar[idx];

    // ── Phase 1: Project radar → image plane ──
    let proj = project_radar(det);
    let world_pos = radar_to_world(det);

    // ── Phase 2: Find best matching camera box ──
    var best_cost: f32 = 1e9;
    var best_cam: i32 = -1;
    var best_box_w: f32 = 0.0;
    var best_box_h: f32 = 0.0;

    let n_cam = min(params.n_camera, MAX_CAMERA);
    for (var c: u32 = 0u; c < n_cam; c++) {
        let box_c = camera[c];
        let dx = proj.x - box_c.cx;
        let dy = proj.y - box_c.cy;
        let spatial_dist = sqrt(dx * dx + dy * dy);

        let half_w = box_c.w * 0.5 + 20.0;
        let half_h = box_c.h * 0.5 + 20.0;
        let inside = f32(abs(dx) < half_w && abs(dy) < half_h);

        let vel_cost = abs(det.velocity - box_c.est_vx) * VELOCITY_WEIGHT;
        let cost = spatial_dist + vel_cost + (1.0 - inside) * 1000.0;

        if (cost < best_cost && cost < GATE_THRESHOLD) {
            best_cost = cost;
            best_cam = i32(c);
            best_box_w = box_c.w;
            best_box_h = box_c.h;
        }
    }

    // ── Phase 3: Atomic greedy assignment ──
    if (best_cam >= 0) {
        let packed = pack_assignment(best_cost, idx);
        atomicMin(&assignments[best_cam], packed);
    }

    // ── Phase 4 & 5: Kalman predict + update ──
    let track_id = idx % params.n_tracks;
    var track = tracks[track_id];

    if (track.is_active > 0.5) {
        kalman_predict(&track, params.dt);

        if (best_cam >= 0) {
            let box_m = camera[u32(best_cam)];
            kalman_update(&track, proj.x, proj.y, det.velocity);
            track.matched_camera = f32(best_cam);
            track.matched_radar = f32(idx);
        } else {
            track.age += 1.0;
        }

        tracks[track_id] = track;
    }

    // ── Phase 6: Object classification ──
    let classification = classify_object(det.rcs, best_box_w, best_box_h, det.velocity);
    let obj_class = classification.x;
    let obj_confidence = classification.y;

    // ── Phase 7: Lane association ──
    let lane_info = compute_lane(world_pos.x);
    let lane_id = lane_info.x;
    let lateral_offset = lane_info.y;

    // ── Phase 8: Time-to-collision ──
    let ttc = compute_ttc(det.range, det.velocity, params.ego_vx);

    // ── Phase 9: Collision risk ──
    let risk = compute_risk(ttc, obj_class, lateral_offset, obj_confidence);

    // ── Phase 10: Path planning ──
    let path_result = evaluate_paths(
        world_pos.x, world_pos.y, track.vx, track.vy,
        obj_class, risk
    );

    // ── Write full result ──
    results[idx] = FusionResult(
        proj.x, proj.y,
        f32(best_cam), best_cost,
        track.x, track.y, track.vx, track.vy,
        obj_class, obj_confidence,
        lane_id, lateral_offset,
        ttc, risk,
        path_result.x, path_result.y,
    );
}
