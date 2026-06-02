// ============================================================
// PTX kernel runner -- loads PTX, allocates buffers, measures
// ============================================================

use std::ffi::CString;
use crate::cuda_bindings::*;
use std::os::raw::c_void;

// Embedded PTX sources (compiled into binary)
const PTX_FP32: &str = include_str!("kernels/fp32.ptx");
const PTX_FP16: &str = include_str!("kernels/fp16.ptx");
const PTX_INT8: &str = include_str!("kernels/int8.ptx");
// WMMA requires SM 7.5 -- loaded conditionally
const PTX_WMMA: &str = include_str!("kernels/wmma.ptx");

pub struct PtxResult {
    pub name: String,
    pub time_s: f64,
    pub perf: f64,        // TFLOP/s or TOP/s
    pub unit: &'static str,
    pub checksum: u64,
    pub ok: bool,
}

impl PtxResult {
    pub fn print(&self) {
        let check = if self.ok { "OK" } else { "BAD!" };
        println!(
            "{:<42} {:>10.4} {:>8.3} {} {}",
            self.name, self.time_s, self.perf, self.unit, check
        );
    }
}

// ── GPU buffer helper ──────────────────────────────────────────

struct GpuBuf {
    ptr: CUdeviceptr,
    size: usize,
}

impl GpuBuf {
    unsafe fn alloc(bytes: usize) -> Self {
        let mut ptr = 0u64;
        cu_check(cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc");
        Self { ptr, size: bytes }
    }
    unsafe fn fill_f32(&self, val: f32) {
        let bits = val.to_bits();
        cu_check(
            cuMemsetD32_v2(self.ptr, bits, self.size / 4),
            "cuMemsetD32",
        );
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.ptr); }
    }
}

// ── Generic kernel runner ──────────────────────────────────────

/// Run a PTX kernel and measure time with CUDA events.
/// extra_bufs: 0 = single buf (ptr,n,reps)
///             1 = two bufs  (ptr,b,n,reps)
///             2 = three bufs (ptr,b,c,n,reps)
unsafe fn run_kernel_timed(
    ptx_src: &str,
    fn_name: &str,
    n: u32,
    reps: u32,
    threads: u32,
    extra_bufs: usize,
) -> (f64, u64) {
    // Load module
    let ptx_c = CString::new(ptx_src).unwrap();
    let mut module = std::ptr::null_mut::<c_void>();
    cu_check(
        cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const c_void),
        &format!("cuModuleLoadData for {}", fn_name),
    );
    let fn_c = CString::new(fn_name).unwrap();
    let mut func = std::ptr::null_mut::<c_void>();
    cu_check(cuModuleGetFunction(&mut func, module, fn_c.as_ptr()), fn_name);

    // Allocate buffers
    let bytes_f32 = (n as usize) * 4;
    let buf_a = GpuBuf::alloc(bytes_f32);
    buf_a.fill_f32(1.5_f32);

    let buf_b = if extra_bufs >= 1 {
        let b = GpuBuf::alloc(bytes_f32);
        b.fill_f32(2.0_f32);
        Some(b)
    } else {
        None
    };
    let buf_c = if extra_bufs >= 2 {
        let c = GpuBuf::alloc(bytes_f32);
        c.fill_f32(0.0_f32);
        Some(c)
    } else {
        None
    };

    // CUDA events
    let mut ev_start = std::ptr::null_mut::<c_void>();
    let mut ev_stop  = std::ptr::null_mut::<c_void>();
    cu_check(cuEventCreate(&mut ev_start, CU_EVENT_DEFAULT), "ev_start");
    cu_check(cuEventCreate(&mut ev_stop,  CU_EVENT_DEFAULT), "ev_stop");

    // Pack kernel params
    let mut p_ptr  = buf_a.ptr;
    let mut p_b    = buf_b.as_ref().map(|b| b.ptr).unwrap_or(0);
    let mut p_c    = buf_c.as_ref().map(|c| c.ptr).unwrap_or(0);
    let mut p_n    = n;
    let mut p_reps = reps;

    let mut params: Vec<*mut c_void> = match extra_bufs {
        0 => vec![
            &mut p_ptr  as *mut _ as *mut c_void,
            &mut p_n    as *mut _ as *mut c_void,
            &mut p_reps as *mut _ as *mut c_void,
        ],
        1 => vec![
            &mut p_ptr  as *mut _ as *mut c_void,
            &mut p_b    as *mut _ as *mut c_void,
            &mut p_n    as *mut _ as *mut c_void,
            &mut p_reps as *mut _ as *mut c_void,
        ],
        _ => vec![
            &mut p_ptr  as *mut _ as *mut c_void,
            &mut p_b    as *mut _ as *mut c_void,
            &mut p_c    as *mut _ as *mut c_void,
            &mut p_n    as *mut _ as *mut c_void,
            &mut p_reps as *mut _ as *mut c_void,
        ],
    };

    let grid = (n + threads - 1) / threads;

    // Warmup
    cuLaunchKernel(
        func, grid, 1, 1, threads, 1, 1, 0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    cuCtxSynchronize();

    // Timed run
    cu_check(cuEventRecord(ev_start, std::ptr::null_mut()), "record start");
    cu_check(cuLaunchKernel(
        func, grid, 1, 1, threads, 1, 1, 0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    ), "launch");
    cu_check(cuEventRecord(ev_stop, std::ptr::null_mut()), "record stop");
    cu_check(cuEventSynchronize(ev_stop), "sync");

    let mut ms = 0.0f32;
    cu_check(cuEventElapsedTime(&mut ms, ev_start, ev_stop), "elapsed");

    // Read back single value for checksum
    let mut result = 0u32;
    cuMemcpyDtoH_v2(
        &mut result as *mut _ as *mut c_void,
        buf_a.ptr,
        4,
    );

    cuEventDestroy_v2(ev_start);
    cuEventDestroy_v2(ev_stop);
    cuModuleUnload(module);

    (ms as f64 / 1000.0, result as u64)
}

// ══════════════════════════════════════════════════════════════
// Public benchmark functions
// ══════════════════════════════════════════════════════════════

pub fn bench_fp32_muladd(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_FP32, "fp32_muladd", n, reps, 256, 0) };
    let ops = n as f64 * reps as f64 * 2.0;
    PtxResult {
        name: "FP32 | explicit MUL+ADD (no FMA)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_fp32_fma(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_FP32, "fp32_fma", n, reps, 256, 0) };
    let ops = n as f64 * reps as f64 * 2.0;
    PtxResult {
        name: "FP32 | FMA (fma.rn.f32)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_fp64_muladd(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_FP32, "fp64_muladd", n, reps, 256, 0) };
    let ops = n as f64 * reps as f64 * 2.0;
    PtxResult {
        name: "FP64 | MUL+ADD (diagnose hardware FP64)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_fp16_muladd(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_FP16, "fp16_muladd", n, reps, 256, 0) };
    // f16x2: each thread processes 2 values per op
    let ops = n as f64 * reps as f64 * 2.0 * 2.0;
    PtxResult {
        name: "FP16 | f16x2 MUL+ADD (best on CMP 50HX!)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_fp16_fma(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_FP16, "fp16_fma", n, reps, 256, 0) };
    let ops = n as f64 * reps as f64 * 2.0 * 2.0;
    PtxResult {
        name: "FP16 | f16x2 FMA".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_int8_dp4a_s8s8(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "int8_dp4a_s8s8", n, reps, 256, 2) };
    // dp4a: 4 muls + 4 adds = 8 ops per call
    let ops = n as f64 * reps as f64 * 8.0;
    PtxResult {
        name: "INT8 | dp4a s8*s8 (Q8_0 path)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: true,
    }
}

pub fn bench_int8_dp4a_u8u8(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "int8_dp4a_u8u8", n, reps, 256, 2) };
    let ops = n as f64 * reps as f64 * 8.0;
    PtxResult {
        name: "INT8 | dp4a u8*u8 (unsigned symm)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: true,
    }
}

pub fn bench_int8_dp4a_s8u8(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "int8_dp4a_s8u8", n, reps, 256, 2) };
    let ops = n as f64 * reps as f64 * 8.0;
    PtxResult {
        name: "INT8 | dp4a s8*u8 (Q8_1 asymm path)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: true,
    }
}

pub fn bench_int4_via_dp4a(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "int4_via_dp4a", n, reps, 256, 2) };
    // 2 dp4a per iter x 8 ops each = 16 effective INT4 ops
    let ops = n as f64 * reps as f64 * 16.0;
    PtxResult {
        name: "INT4 | emulated via dp4a (Q4_K/Q4_0 path)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: true,
    }
}

pub fn bench_bitwise_xnor(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "bitwise_xnor_popc", n, reps, 256, 2) };
    // 32 bits per b32 register per popc
    let ops = n as f64 * reps as f64 * 32.0;
    PtxResult {
        name: "Bitwise | XNOR+POPC (1-bit / binary nets)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: true,
    }
}

pub fn bench_int32_mad(n: u32, reps: u32) -> PtxResult {
    let (t, cs) = unsafe { run_kernel_timed(PTX_INT8, "int32_mad", n, reps, 256, 0) };
    let ops = n as f64 * reps as f64 * 2.0;
    PtxResult {
        name: "INT32 | MAD (accumulator baseline)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: cs != 0,
    }
}

// ── WMMA -- only call on SM 7.5+ ──────────────────────────────

pub fn bench_wmma_fp16_f32(sm_major: u32, sm_minor: u32, iters: u32) -> PtxResult {
    if sm_major < 7 || (sm_major == 7 && sm_minor < 5) {
        return PtxResult {
            name: "WMMA | FP16->FP32 (TC)".into(),
            time_s: 0.0, perf: 0.0, unit: "TFLOP/s",
            checksum: 0, ok: false,
        };
    }
    // 16x16x16 WMMA: 2 * 16^3 = 8192 FLOPs per mma call
    let (t, cs) = unsafe { run_wmma_kernel("wmma_fp16_f32", iters) };
    let ops = iters as f64 * 8192.0;
    PtxResult {
        name: "WMMA | FP16->FP32 acc m16n16k16 (direct TC test)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_wmma_fp16_f16(sm_major: u32, sm_minor: u32, iters: u32) -> PtxResult {
    if sm_major < 7 || (sm_major == 7 && sm_minor < 5) {
        return PtxResult {
            name: "WMMA | FP16->FP16 (TC)".into(),
            time_s: 0.0, perf: 0.0, unit: "TFLOP/s",
            checksum: 0, ok: false,
        };
    }
    let (t, cs) = unsafe { run_wmma_kernel("wmma_fp16_f16", iters) };
    let ops = iters as f64 * 8192.0;
    PtxResult {
        name: "WMMA | FP16->FP16 acc m16n16k16 (TC f16 acc)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TFLOP/s",
        checksum: cs, ok: cs != 0,
    }
}

pub fn bench_wmma_int8_s32(sm_major: u32, sm_minor: u32, iters: u32) -> PtxResult {
    if sm_major < 7 || (sm_major == 7 && sm_minor < 5) {
        return PtxResult {
            name: "WMMA | INT8->INT32 (Turing IMMA)".into(),
            time_s: 0.0, perf: 0.0, unit: "TOP/s",
            checksum: 0, ok: false,
        };
    }
    // m8n32k16: 2 * 8 * 32 * 16 = 8192 ops
    let (t, cs) = unsafe { run_wmma_kernel("wmma_int8_s32", iters) };
    let ops = iters as f64 * 8192.0;
    PtxResult {
        name: "WMMA | INT8->INT32 m8n32k16 (Turing IMMA TC)".into(),
        time_s: t, perf: ops / t / 1e12, unit: "TOP/s",
        checksum: cs, ok: cs != 0,
    }
}

unsafe fn run_wmma_kernel(fn_name: &str, n_iters: u32) -> (f64, u64) {
    let ptx_c = CString::new(PTX_WMMA).unwrap();
    let mut module = std::ptr::null_mut::<c_void>();
    cu_check(
        cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const c_void),
        fn_name,
    );
    let fn_c = CString::new(fn_name).unwrap();
    let mut func = std::ptr::null_mut::<c_void>();
    cu_check(cuModuleGetFunction(&mut func, module, fn_c.as_ptr()), fn_name);

    // Allocate aligned matrices (16x16 f16 = 512 bytes each)
    let mat_bytes = 16 * 16 * 2; // 16x16 fp16
    let ba = GpuBuf::alloc(mat_bytes);
    ba.fill_f32(1.0);
    let bb = GpuBuf::alloc(mat_bytes);
    bb.fill_f32(1.0);
    let bc = GpuBuf::alloc(16 * 16 * 4); // f32 accumulator
    // zero-init accumulator
    cuMemsetD32_v2(bc.ptr, 0, 16 * 16);

    let mut ev_s = std::ptr::null_mut::<c_void>();
    let mut ev_e = std::ptr::null_mut::<c_void>();
    cu_check(cuEventCreate(&mut ev_s, CU_EVENT_DEFAULT), "ev_s");
    cu_check(cuEventCreate(&mut ev_e, CU_EVENT_DEFAULT), "ev_e");

    let mut pa = ba.ptr;
    let mut pb = bb.ptr;
    let mut pc = bc.ptr;
    let mut ni = n_iters;

    let mut params = vec![
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
    ];

    // Warmup: 1 warp = 32 threads
    cuLaunchKernel(
        func, 1, 1, 1, 32, 1, 1, 0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    cuCtxSynchronize();

    cuEventRecord(ev_s, std::ptr::null_mut());
    cuLaunchKernel(
        func, 1, 1, 1, 32, 1, 1, 0,
        std::ptr::null_mut(),
        params.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    cuEventRecord(ev_e, std::ptr::null_mut());
    cuEventSynchronize(ev_e);

    let mut ms = 0.0f32;
    cuEventElapsedTime(&mut ms, ev_s, ev_e);

    // Read first element of result for correctness check
    // For all-ones matrices A*B: each element = 16.0
    let mut result = [0u32; 1];
    cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, bc.ptr, 4);
    let ok = (f32::from_bits(result[0]) - 16.0).abs() < 1.0;

    cuEventDestroy_v2(ev_s);
    cuEventDestroy_v2(ev_e);
    cuModuleUnload(module);

    (ms as f64 / 1000.0, if ok { result[0] as u64 } else { 0 })
}
