// ============================================================
// cuBLAS GEMM probe — full algorithm sweep with correctness check
// Identical to the original nbenchmark behaviour + extended
// ============================================================

use std::os::raw::c_void;
use crate::cuda_bindings::*;

#[derive(Debug, Clone)]
pub struct GemmResult {
    pub label:     String,
    pub time_s:    f64,
    pub tflops:    f64,
    pub ok:        bool,
    pub is_ref:    bool,   // mark with * if pedantic reference
}

impl GemmResult {
    pub fn print(&self) {
        let star = if self.is_ref { " *" } else { "  " };
        let check = if self.ok { "OK" } else { "BAD!" };
        println!(
            "{:<46} {:>10.4}  {:>8.2}{} {:>6}",
            self.label, self.time_s, self.tflops, star, check
        );
    }
}

pub fn run_cublas_probe(m: i32, iters: u32) -> Vec<GemmResult> {
    let expected_val = m as f32; // all-ones matrices: each element = M

    unsafe {
        let mut handle = std::ptr::null_mut::<c_void>();
        cublas_check(cublasCreate_v2(&mut handle), "cublasCreate");

        // Allocate fp16 matrices A, B and fp32 output C
        let n_fp16 = (m * m) as usize * 2;
        let n_fp32 = (m * m) as usize * 4;

        let mut d_a16 = 0u64; cuMemAlloc_v2(&mut d_a16, n_fp16);
        let mut d_b16 = 0u64; cuMemAlloc_v2(&mut d_b16, n_fp16);
        let mut d_a32 = 0u64; cuMemAlloc_v2(&mut d_a32, n_fp32);
        let mut d_b32 = 0u64; cuMemAlloc_v2(&mut d_b32, n_fp32);
        let mut d_c   = 0u64; cuMemAlloc_v2(&mut d_c,   n_fp32);

        // Fill host buffers: all 1.0
        let ones_f16: Vec<u16> = vec![0x3C00u16; (m * m) as usize]; // 1.0 in f16
        let ones_f32: Vec<f32> = vec![1.0f32;    (m * m) as usize];
        let zeros:    Vec<f32> = vec![0.0f32;    (m * m) as usize];

        cuMemcpyHtoD_v2(d_a16, ones_f16.as_ptr() as *const c_void, n_fp16);
        cuMemcpyHtoD_v2(d_b16, ones_f16.as_ptr() as *const c_void, n_fp16);
        cuMemcpyHtoD_v2(d_a32, ones_f32.as_ptr() as *const c_void, n_fp32);
        cuMemcpyHtoD_v2(d_b32, ones_f32.as_ptr() as *const c_void, n_fp32);

        let alpha_f32 = 1.0f32;
        let beta_f32  = 0.0f32;

        let mut results = Vec::new();

        // Helper closure: time a cuBLAS gemm call
        macro_rules! time_gemm {
            ($label:expr, $is_ref:expr, $math:expr, $fn:expr) => {{
                cublasSetMathMode(handle, $math);
                // reset C
                cuMemcpyHtoD_v2(d_c, zeros.as_ptr() as *const c_void, n_fp32);
                // warmup
                $fn;
                cuCtxSynchronize();
                // timed
                let mut ev_s = std::ptr::null_mut::<c_void>();
                let mut ev_e = std::ptr::null_mut::<c_void>();
                cuEventCreate(&mut ev_s, CU_EVENT_DEFAULT);
                cuEventCreate(&mut ev_e, CU_EVENT_DEFAULT);
                cuMemcpyHtoD_v2(d_c, zeros.as_ptr() as *const c_void, n_fp32);
                cuEventRecord(ev_s, std::ptr::null_mut());
                for _ in 0..iters { $fn; }
                cuEventRecord(ev_e, std::ptr::null_mut());
                cuEventSynchronize(ev_e);
                let mut ms = 0.0f32;
                cuEventElapsedTime(&mut ms, ev_s, ev_e);
                let t = ms as f64 / 1000.0 / iters as f64;
                let flops = 2.0 * m as f64 * m as f64 * m as f64;
                let tflops = flops / t / 1e12;
                // correctness: read C[0,0], expect m
                let mut val = 0.0f32;
                cuMemcpyDtoH_v2(&mut val as *mut _ as *mut c_void, d_c, 4);
                let ok = (val - expected_val).abs() < expected_val * 0.01;
                cuEventDestroy_v2(ev_s);
                cuEventDestroy_v2(ev_e);
                GemmResult {
                    label: $label.to_string(),
                    time_s: t, tflops, ok, is_ref: $is_ref,
                }
            }};
        }

        // ─── FP16 input, various compute modes ───────────────────────────
        results.push(time_gemm!(
            "FP16 | 16F DEFAULT", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT)
        ));

        results.push(time_gemm!(
            "FP16 | 16F TENSOR_OP", false,
            CUBLAS_TF32_TENSOR_OP_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DFALT_TENSOR_OP)
        ));

        results.push(time_gemm!(
            "FP16 | 16F_PEDANTIC (no TC/FMA)", true,
            CUBLAS_PEDANTIC_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F_PEDANTIC, CUBLAS_GEMM_DEFAULT)
        ));

        results.push(time_gemm!(
            "FP16 | 16F ALGO_1 (CUDA cores)", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_ALGO1)
        ));

        results.push(time_gemm!(
            "FP16 | 16F ALGO_2", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_ALGO2)
        ));

        results.push(time_gemm!(
            "FP16 | 16F ALGO_3", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_16F, CUBLAS_GEMM_ALGO3)
        ));

        // ─── FP16 input, 32F compute (correct on Pascal!) ────────────────
        results.push(time_gemm!(
            "FP16 | 32F_PEDANTIC", true,
            CUBLAS_PEDANTIC_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT)
        ));

        results.push(time_gemm!(
            "FP16 | 32F DEFAULT (uses FMA)", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
        ));

        results.push(time_gemm!(
            "FP16 | 32F_FAST_16F", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT)
        ));

        results.push(time_gemm!(
            "FP16 | 32F_FAST_TF32", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a16 as *const c_void, CUDA_R_16F, m,
                d_b16 as *const c_void, CUDA_R_16F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT)
        ));

        // ─── FP32 input ───────────────────────────────────────────────────
        results.push(time_gemm!(
            "FP32 | 32F_PEDANTIC", true,
            CUBLAS_PEDANTIC_MATH,
            cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32,
                d_a32 as *const f32, m,
                d_b32 as *const f32, m,
                &beta_f32,
                d_c   as *mut   f32, m)
        ));

        results.push(time_gemm!(
            "FP32 | 32F DEFAULT", false,
            CUBLAS_DEFAULT_MATH,
            cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32,
                d_a32 as *const f32, m,
                d_b32 as *const f32, m,
                &beta_f32,
                d_c   as *mut   f32, m)
        ));

        results.push(time_gemm!(
            "FP32 | 32F ALGO_5", false,
            CUBLAS_DEFAULT_MATH,
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m,m,m,
                &alpha_f32 as *const _ as *const c_void,
                d_a32 as *const c_void, CUDA_R_32F, m,
                d_b32 as *const c_void, CUDA_R_32F, m,
                &beta_f32  as *const _ as *const c_void,
                d_c   as *mut   c_void, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO1_TENSOR_OP)
        ));

        // Cleanup
        cuMemFree_v2(d_a16); cuMemFree_v2(d_b16);
        cuMemFree_v2(d_a32); cuMemFree_v2(d_b32);
        cuMemFree_v2(d_c);
        cublasDestroy_v2(handle);

        results
    }
}
