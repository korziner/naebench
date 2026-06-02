// ============================================================
// Raw CUDA / cuBLAS FFI bindings
// ============================================================
// Link: -lcuda -lcublas -lcublasLt
// Required: CUDA toolkit with libcuda.so.1, libcublas.so.12

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::os::raw::{c_int, c_uint, c_void, c_char, c_float};
use std::ffi::CStr;

// ── Types ──────────────────────────────────────────────────────
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUresult = c_uint;

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = c_uint;
pub type cublasOperation_t = c_uint;
pub type cublasMath_t = c_uint;
pub type cublasGemmAlgo_t = c_int;
pub type cublasComputeType_t = c_uint;
pub type cudaDataType_t = c_uint;

// ── CUDA result codes ──────────────────────────────────────────
pub const CUDA_SUCCESS: CUresult = 0;

// ── cuBLAS status codes ────────────────────────────────────────
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// ── Operation types ────────────────────────────────────────────
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;

// ── Math modes ─────────────────────────────────────────────────
pub const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
pub const CUBLAS_PEDANTIC_MATH: cublasMath_t = 2;
pub const CUBLAS_TF32_TENSOR_OP_MATH: cublasMath_t = 3;
pub const CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION: cublasMath_t = 16;

// ── Compute types ──────────────────────────────────────────────
pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 64;
pub const CUBLAS_COMPUTE_16F_PEDANTIC: cublasComputeType_t = 65;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;
pub const CUBLAS_COMPUTE_32F_PEDANTIC: cublasComputeType_t = 69;
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 74;
pub const CUBLAS_COMPUTE_32F_FAST_TF32: cublasComputeType_t = 75;

// ── Data types ─────────────────────────────────────────────────
pub const CUDA_R_16F: cudaDataType_t = 2;
pub const CUDA_R_32F: cudaDataType_t = 0;
pub const CUDA_R_8I: cudaDataType_t = 3;
pub const CUDA_R_32I: cudaDataType_t = 10;

// ── GEMM algorithms ────────────────────────────────────────────
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_ALGO0: cublasGemmAlgo_t = 0;
pub const CUBLAS_GEMM_ALGO1: cublasGemmAlgo_t = 1;
pub const CUBLAS_GEMM_ALGO2: cublasGemmAlgo_t = 2;
pub const CUBLAS_GEMM_ALGO3: cublasGemmAlgo_t = 3;
pub const CUBLAS_GEMM_DFALT_TENSOR_OP: cublasGemmAlgo_t = 99;
pub const CUBLAS_GEMM_ALGO1_TENSOR_OP: cublasGemmAlgo_t = 100;

// ── CUevent flags ──────────────────────────────────────────────
pub const CU_EVENT_DEFAULT: c_uint = 0;
pub const CU_EVENT_BLOCKING_SYNC: c_uint = 1;

// ── Device attributes ──────────────────────────────────────────
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: c_int = 13;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: c_int = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;

extern "C" {
    // ── Driver API ─────────────────────────────────────────────
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut c_int, attrib: c_int, dev: CUdevice,
    ) -> CUresult;

    pub fn cuCtxCreate_v2(
        pctx: *mut CUcontext, flags: c_uint, dev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(
        dst: CUdeviceptr, src: *const c_void, bytecount: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        dst: *mut c_void, src: CUdeviceptr, bytecount: usize,
    ) -> CUresult;
    pub fn cuMemsetD32_v2(
        dstDevice: CUdeviceptr, ui: c_uint, n: usize,
    ) -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    pub fn cuEventCreate(phEvent: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(
        pMilliseconds: *mut c_float,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult;

    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;

    // ── cuBLAS Runtime API ─────────────────────────────────────
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetMathMode(
        handle: cublasHandle_t,
        mode: cublasMath_t,
    ) -> cublasStatus_t;
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int, n: c_int, k: c_int,
        alpha: *const c_void,
        A: *const c_void,
        Atype: cudaDataType_t,
        lda: c_int,
        B: *const c_void,
        Btype: cudaDataType_t,
        ldb: c_int,
        beta: *const c_void,
        C: *mut c_void,
        Ctype: cudaDataType_t,
        ldc: c_int,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t;
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int, n: c_int, k: c_int,
        alpha: *const c_float,
        A: *const c_float, lda: c_int,
        B: *const c_float, ldb: c_int,
        beta: *const c_float,
        C: *mut c_float, ldc: c_int,
    ) -> cublasStatus_t;
}

// ── Helper: format CUDA error ──────────────────────────────────

pub fn cu_error_text(r: CUresult) -> String {
    unsafe {
        let mut name_ptr: *const c_char = std::ptr::null();
        let mut desc_ptr: *const c_char = std::ptr::null();

        let name = if cuGetErrorName(r, &mut name_ptr) == CUDA_SUCCESS && !name_ptr.is_null() {
            CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
        } else {
            format!("CUDA_ERROR_0x{:08X}", r)
        };

        let desc = if cuGetErrorString(r, &mut desc_ptr) == CUDA_SUCCESS && !desc_ptr.is_null() {
            CStr::from_ptr(desc_ptr).to_string_lossy().into_owned()
        } else {
            "unknown CUDA error".to_string()
        };

        format!("{} (0x{:08X}): {}", name, r, desc)
    }
}

pub fn cu_check(r: CUresult, msg: &str) {
    if r != CUDA_SUCCESS {
        panic!("{} at {}", cu_error_text(r), msg);
    }
}

pub fn cublas_check(r: cublasStatus_t, msg: &str) {
    if r != CUBLAS_STATUS_SUCCESS {
        panic!("cuBLAS error {} at: {}", r, msg);
    }
}
