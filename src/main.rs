use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::time::Instant;
use wgpu::util::DeviceExt;

// ─── Structs (must match WGSL layout) ───

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n_radar: u32,
    n_camera: u32,
    n_tracks: u32,
    dt: f32,
    focal_x: f32,
    focal_y: f32,
    cx: f32,
    cy: f32,
    ego_vx: f32,
    ego_vy: f32,
    lane_width: f32,
    n_lanes: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Track {
    x: f32, y: f32, vx: f32, vy: f32, ax: f32, ay: f32,
    age: f32, is_active: f32,
    p00: f32, p11: f32, p22: f32, p33: f32, p44: f32, p55: f32,
    p01: f32, p23: f32,
    innovation_x: f32, innovation_y: f32,
    matched_camera: f32, matched_radar: f32,
}

// ─── GPU setup ───

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend: String,
}

fn init_gpu() -> Gpu {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("No GPU adapter found");

        let info = adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let adapter_name = info.name.clone();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("adas-bench"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create device");

        Gpu { device, queue, adapter_name, backend }
    })
}

// ─── Benchmark ───

struct BenchResult {
    name: String,
    mean_ms: f64,
    std_ms: f64,
    fps: f64,
    runs: usize,
}

fn stats(times: &[f64]) -> (f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    (mean, var.sqrt())
}

fn run_adas(gpu: &Gpu, n_radar: u32, n_camera: u32, n_tracks: u32, warmup: usize, runs: usize) -> BenchResult {
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sensor_fusion"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sensor_fusion.wgsl").into()),
    });

    let params = Params {
        n_radar, n_camera, n_tracks,
        dt: 0.033,
        focal_x: 1000.0, focal_y: 1000.0, cx: 960.0, cy: 540.0,
        ego_vx: 25.0, ego_vy: 0.0,
        lane_width: 3.7, n_lanes: 4,
    };

    let mut rng = rand::thread_rng();

    let radar_data: Vec<RadarDet> = (0..n_radar).map(|_| RadarDet {
        range: rng.gen_range(5.0f32..150.0),
        azimuth: rng.gen_range(-0.5f32..0.5),
        elevation: rng.gen_range(-0.1f32..0.1),
        velocity: rng.gen_range(-30.0f32..30.0),
        rcs: rng.gen_range(-10.0f32..30.0),
        _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
    }).collect();

    let camera_data: Vec<CameraBox> = (0..n_camera).map(|_| CameraBox {
        cx: rng.gen_range(0.0f32..1920.0),
        cy: rng.gen_range(0.0f32..1080.0),
        w: rng.gen_range(30.0f32..300.0),
        h: rng.gen_range(30.0f32..300.0),
        confidence: rng.gen_range(0.5f32..1.0),
        est_vx: rng.gen_range(-30.0f32..30.0),
        _pad1: 0.0, _pad2: 0.0,
    }).collect();

    let track_data: Vec<Track> = (0..n_tracks).map(|_| Track {
        x: rng.gen_range(0.0f32..1920.0),
        y: rng.gen_range(0.0f32..1080.0),
        vx: rng.gen_range(-10.0f32..10.0),
        vy: rng.gen_range(-10.0f32..10.0),
        ax: 0.0, ay: 0.0, age: 0.0, is_active: 1.0,
        p00: 10.0, p11: 10.0, p22: 5.0, p33: 5.0, p44: 1.0, p55: 1.0,
        p01: 0.0, p23: 0.0,
        innovation_x: 0.0, innovation_y: 0.0,
        matched_camera: -1.0, matched_radar: -1.0,
    }).collect();

    let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let radar_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&radar_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let camera_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&camera_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let tracks_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&track_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let result_size = (n_radar as u64) * 64; // 16 floats per FusionResult
    let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let assign_data: Vec<u32> = vec![0xFFFFFFFFu32; n_camera as usize];
    let assign_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&assign_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        })),
        module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: radar_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: camera_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: tracks_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: results_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: assign_buf.as_entire_binding() },
        ],
    });

    let wg = (n_radar + 63) / 64;

    let run_one = || {
        gpu.queue.write_buffer(&assign_buf, 0, bytemuck::cast_slice(&assign_data));
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(wg, 1, 1); }
        enc.copy_buffer_to_buffer(&results_buf, 0, &readback_buf, 0, result_size);
        gpu.queue.submit(Some(enc.finish()));
        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::Maintain::Wait);
        drop(slice.get_mapped_range());
        readback_buf.unmap();
    };

    for _ in 0..warmup { run_one(); }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        run_one();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let (mean, std) = stats(&times);
    BenchResult {
        name: format!("R={n_radar}, C={n_camera}, T={n_tracks}"),
        mean_ms: mean, std_ms: std, fps: 1000.0 / mean, runs,
    }
}

// ─── Main ───

fn print_result(r: &BenchResult) {
    println!("  {:30} {:8.3} ms/frame  (±{:.3})  {:>8.1} fps  (N={})",
        r.name, r.mean_ms, r.std_ms, r.fps, r.runs);
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ADAS Sensor Fusion Benchmark                              ║");
    println!("║  Full pipeline: 11 stages fused into 1 GPU dispatch        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = init_gpu();
    println!("  GPU:     {}", gpu.adapter_name);
    println!("  Backend: {}", gpu.backend);
    println!();

    let warmup = 5;
    let runs = 30;

    println!("Pipeline stages (all fused into single dispatch):");
    println!("  1. Radar projection (polar → image)");
    println!("  2. Cost matrix (radar × camera matching)");
    println!("  3. Greedy association (atomicMin)");
    println!("  4. Kalman predict (6-state constant acceleration)");
    println!("  5. Kalman update (fused radar + camera measurement)");
    println!("  6. Object classification (RCS + box size → class)");
    println!("  7. Lane association (lateral position → lane ID)");
    println!("  8. Time-to-collision");
    println!("  9. Collision risk scoring");
    println!(" 10. Path planning (16 candidates × 10 steps)");
    println!(" 11. Risk aggregation");
    println!();

    println!("─── Results ───");
    print_result(&run_adas(&gpu, 256, 50, 128, warmup, runs));
    print_result(&run_adas(&gpu, 512, 100, 256, warmup, runs));
    print_result(&run_adas(&gpu, 1024, 200, 512, warmup, runs));

    println!();
    println!("  ADAS requires 30 fps (33 ms budget).");
    println!("  Compare with: python3 pytorch_sensor_fusion.py");
    println!();
}
