// ============================================================
// nbenchmark v2.0 — Full Pascal + Turing benchmark
// Reveals broken FMA/TC on NVIDIA CMP mining cards
// Maps GGUF formats to GPU instruction paths
//
// Usage:  nbenchmark <COMMAND> [OPTIONS]
// Build:  cargo build --release
// Deps:   libcuda.so.1, libcublas.so.12 (CUDA 12.x toolkit)
// ============================================================

mod cuda_bindings;
mod ptx_runner;
mod cublas_probe;
mod gpu_db;
mod report;

use std::os::raw::c_char;
use clap::{Parser, Subcommand};
use cuda_bindings::*;
use report::{BenchResults, print_full_report, print_gpu_comparison_table};

// ── CLI ──────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "nbenchmark",
    version = "2.0.0",
    about   = "Pascal+Turing GPU benchmark: PTX kernels, cuBLAS probe, GGUF advisor",
    long_about = "\
Full benchmark for NVIDIA Pascal (SM 6.x) and Turing (SM 7.5) GPUs.\n\
Reveals broken FMA / Tensor Cores on NVIDIA CMP mining cards.\n\
Maps all GGUF quantization formats to GPU instruction paths.\n\
\n\
TARGET HARDWARE:\n\
  Pascal: P106-100 (6GB), P104-100 (8GB), P102-100 (10GB), Tesla P100 (16GB)\n\
  Turing: CMP 50HX, RTX 2060 (6GB), RTX 2060 SUPER (8GB),\n\
          RTX 2060 12GB, Quadro T400, RTX 2070, RTX 2080 Ti\n\
\n\
KNOWN RESULTS:\n\
  P102-100 (Pascal):  FP32 mul+add ~5.6 TF | FP32 FMA ~10.5 TF | INT8 ~41 TOPS\n\
  CMP 50HX (Turing):  FP32 mul+add ~5.9 TF | FP32 FMA ~0.42 TF (BROKEN!) | INT8 ~1.7 TOPS\n\
  RTX 2060 (Turing):  FP32 ~6.5 TF | FP16 TC ~52 TF | INT8 TC ~104 TOPS\n\
  RTX 2060 SUPER:     FP32 ~7.2 TF | FP16 TC ~57 TF | BW 448 GB/s\n\
  RTX 2060 12GB:      FP32 ~7.3 TF | FP16 TC ~58 TF | BW 360 GB/s | VRAM 12GB\n\
  Quadro T400:        FP32 ~1.1 TF | NO TC (TU117!) | BW 80 GB/s | VRAM 4GB\n\
\n\
GGUF GPU PATHS:\n\
  dp4a MMQ:    Q8_0, Q4_0, Q4_K_M, IQ4_XS, IQ4_NL     → fastest on Pascal!\n\
  cuBLAS F32:  Q5_K_M, Q6_K, Q3_K, IQ2-3 series        → 7x slower than dp4a\n\
  FP32 GEMV:   TQ1_0, TQ2_0, IQ1_S/M                   → avoid on GPU\n\
  FORBIDDEN:   BF16, TF32, FP8 (not available SM<8.0)\n\
",
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, default_value = "67108864", global = true,
          help = "Number of elements for PTX benchmarks")]
    n: u32,

    #[arg(long, default_value = "16384", global = true,
          help = "Arithmetic repetitions per GPU thread")]
    reps: u32,

    #[arg(long, default_value = "2048", global = true,
          help = "GEMM matrix dimension M=N=K for cuBLAS probe")]
    m: i32,

    #[arg(long, default_value = "50", global = true,
          help = "GEMM iterations for stable timing")]
    gemm_iters: u32,

    #[arg(long, default_value = "0", global = true,
          help = "CUDA device index (0 = first GPU)")]
    device: i32,
}

#[derive(Subcommand)]
enum Commands {
    /// FP32 explicit mul+add (no FMA) — safe baseline
    Fp32,
    /// FP32 FMA — reveals CMP breakage if 14x slower than fp32
    Fp32Fma,
    /// FP64 mul+add — diagnose hardware FP64 (GP100 vs consumer)
    Fp64,
    /// FP16 f16x2 explicit mul+add — best path on CMP 50HX
    F16,
    /// FP16 f16x2 FMA
    F16Fma,
    /// INT8 dp4a all variants: s8s8 / u8u8 / s8u8 (Q8_0/Q8_1 paths)
    Int8,
    /// INT4 emulated via dp4a (Q4_K/Q4_0 GPU path)
    Int4,
    /// INT32 MAD — accumulator throughput baseline
    Int32,
    /// Bitwise XNOR + POPC — 1-bit / binary neural nets
    Bitwise,
    /// All PTX: fp32/fp64/fp16/int8/int4/bitwise/int32
    AllPtx,
    /// FP32 vs FMA comparison (CMP diagnostic)
    FmaCompare,
    /// KEY: cuBLAS GEMM probe — all compute modes, correctness check
    CublasProbe,
    /// WMMA direct Tensor Core test (SM 7.5+ only, bypasses cuBLAS)
    WmmaProbe,
    /// Quick CMP auto-detect: broken FMA + TC?
    CmpDetect,
    /// Print GGUF format → GPU path mapping table
    GgufTable,
    /// Print known GPU specs comparison table
    GpuTable,
    /// Everything: all PTX + FMA compare + cuBLAS + WMMA + recommendations
    FullReport,
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

fn main() {
    let cli = Cli::parse();

match &cli.command {
    Commands::GgufTable => {
        print_gguf_format_table();
        return;
    }
    Commands::GpuTable => {
        print_gpu_comparison_table();
        return;
    }
    _ => {}
}

    // ── Init CUDA ────────────────────────────────────────────
    unsafe {
        cu_check(cuInit(0), "cuInit");

        let mut dev_count = 0i32;
        cu_check(cuDeviceGetCount(&mut dev_count), "cuDeviceGetCount");
        if dev_count == 0 {
            eprintln!("No CUDA devices found!");
            std::process::exit(1);
        }

        let dev_idx = cli.device;
        if dev_idx >= dev_count {
            eprintln!("Device {} not found (only {} devices)", dev_idx, dev_count);
            std::process::exit(1);
        }

        let mut dev = 0i32;
        cu_check(cuDeviceGet(&mut dev, dev_idx), "cuDeviceGet");

        // Read GPU info
        let mut name_buf = vec![0u8; 256];
        cu_check(
            cuDeviceGetName(name_buf.as_mut_ptr() as *mut c_char, 256, dev),
            "cuDeviceGetName",
        );
        let gpu_name = String::from_utf8_lossy(
            &name_buf[..name_buf.iter().position(|&b| b==0).unwrap_or(255)]
        ).to_string();

        let mut vram_bytes = 0usize;
        cuDeviceTotalMem_v2(&mut vram_bytes, dev);
        let vram_gb = vram_bytes as f32 / 1e9;

        let mut sm_maj = 0i32; let mut sm_min = 0i32;
        cuDeviceGetAttribute(&mut sm_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&mut sm_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

        let mut mem_bus_bits = 0i32;
        cuDeviceGetAttribute(&mut mem_bus_bits, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        let mut mem_clk_khz = 0i32;
        cuDeviceGetAttribute(&mut mem_clk_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        // BW = 2 * mem_clk_hz * bus_bits / 8
        let mem_bw_gbs = 2.0 * mem_clk_khz as f32 * 1000.0 * mem_bus_bits as f32 / 8.0 / 1e9;

        println!("GPU: {} (SM {}.{})", gpu_name, sm_maj, sm_min);
        println!("VRAM: {:.1} GB   Bus: {}-bit   Bandwidth: ~{:.0} GB/s",
                 vram_gb, mem_bus_bits, mem_bw_gbs);

        // Create context
        //let mut ctx = std::ptr::null_mut::<c_void>();
        //cu_check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate");

        let mut ctx: CUcontext = std::ptr::null_mut();
        cu_check(cuDevicePrimaryCtxRetain(&mut ctx, dev), "cuDevicePrimaryCtxRetain");
        cu_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

        let n     = cli.n;
        let reps  = cli.reps;
        let m     = cli.m;
        let giters = cli.gemm_iters;
        let sm_major = sm_maj as u32;
        let sm_minor = sm_min as u32;

        match &cli.command {
            Commands::Fp32      => { ptx_runner::bench_fp32_muladd(n, reps).print(); }
            Commands::Fp32Fma   => { ptx_runner::bench_fp32_fma(n, reps).print(); }
            Commands::Fp64      => { ptx_runner::bench_fp64_muladd(n, reps).print(); }
            Commands::F16       => { ptx_runner::bench_fp16_muladd(n, reps).print(); }
            Commands::F16Fma    => { ptx_runner::bench_fp16_fma(n, reps).print(); }

            Commands::Int8 => {
                println!("\n=== INT8 dp4a variants ===");
                println!("{:<44} {:>10}  {:>10}", "Mode", "Time(s)", "TOP/s");
                println!("{}", "─".repeat(68));
                ptx_runner::bench_int8_dp4a_s8s8(n, reps).print();
                ptx_runner::bench_int8_dp4a_u8u8(n, reps).print();
                ptx_runner::bench_int8_dp4a_s8u8(n, reps).print();
            }

            Commands::Int4    => { ptx_runner::bench_int4_via_dp4a(n, reps).print(); }
            Commands::Int32   => { ptx_runner::bench_int32_mad(n, reps).print(); }
            Commands::Bitwise => { ptx_runner::bench_bitwise_xnor(n, reps).print(); }

            Commands::AllPtx => {
                run_all_ptx(n, reps, sm_major, sm_minor);
            }

            Commands::FmaCompare => {
                run_fma_compare(n, reps);
            }

            Commands::CublasProbe => {
                println!("\n========================================================================");
                println!("  cuBLAS GEMM Tensor Core Probe  ({}x{}x{}, {} iters)", m,m,m, giters);
                println!("  Timing: CUDA events (accurate GPU time)");
                println!("  Correctness: C=A*B where A=B=ones, expect C[0]=={}", m);
                println!("========================================================================");
                println!("{:<46} {:>10}  {:>8} {:>6}", "Mode", "Time(s)", "TFLOP/s", "Check");
                println!("{}", "─".repeat(76));
                let results = cublas_probe::run_cublas_probe(m, giters);
                for r in &results {
                    r.print();
                }
                println!("{}", "─".repeat(76));
                let best = results.iter().filter(|r| r.ok)
                    .max_by(|a,b| a.tflops.partial_cmp(&b.tflops).unwrap());
                if let Some(b) = best {
                    println!("Best correct: {} ({:.2} TFLOP/s)", b.label, b.tflops);
                }
            }

            Commands::WmmaProbe => {
                if sm_major < 7 || (sm_major == 7 && sm_minor < 5) {
                    println!("WMMA requires SM 7.5+ (Turing). This GPU is SM {}.{}.", sm_major, sm_minor);
                    println!("Pascal does not have Tensor Cores.");
                } else {
                    println!("\n=== WMMA Direct Tensor Core Probe (SM {}.{}) ===", sm_major, sm_minor);
                    println!("{:<44} {:>10}  {:>8} {:>6}", "Mode", "TFLOP/s", "", "Check");
                    println!("{}", "─".repeat(72));
                    ptx_runner::bench_wmma_fp16_f32(sm_major, sm_minor, 10000).print();
                    ptx_runner::bench_wmma_fp16_f16(sm_major, sm_minor, 10000).print();
                    ptx_runner::bench_wmma_int8_s32(sm_major, sm_minor, 10000).print();
                }
            }

            Commands::CmpDetect => {
                run_cmp_detect(n, reps, sm_major, sm_minor, &gpu_name);
            }

            Commands::GgufTable => {
                print_gguf_format_table();
            }

            Commands::GpuTable => {
                print_gpu_comparison_table();
            }

            Commands::FullReport => {
                println!("\nRunning full benchmark suite... (this takes a few minutes)");
                let results = run_full_suite(n, reps, m, giters, sm_major, sm_minor,
                                             &gpu_name, vram_gb, mem_bw_gbs);
                print_full_report(&results);
            }
        }

        //cuCtxDestroy_v2(ctx);
        cu_check(cuDevicePrimaryCtxRelease(dev), "cuDevicePrimaryCtxRelease");
    }
}

// ═══════════════════════════════════════════════════════════════

fn run_all_ptx(n: u32, reps: u32, sm_major: u32, sm_minor: u32) {
    println!("\n=== PTX Arithmetic Benchmark ===");
    println!("N: {}  Reps: {}", n, reps);
    println!("{:<44} {:>10}  {:>10}", "Mode", "Time(s)", "Perf");
    println!("{}", "─".repeat(68));

    ptx_runner::bench_fp64_muladd(n, reps).print();
    ptx_runner::bench_fp32_muladd(n, reps).print();
    ptx_runner::bench_fp32_fma(n, reps).print();
    ptx_runner::bench_fp16_muladd(n, reps).print();
    ptx_runner::bench_fp16_fma(n, reps).print();
    ptx_runner::bench_int32_mad(n, reps).print();
    ptx_runner::bench_int8_dp4a_s8s8(n, reps).print();
    ptx_runner::bench_int8_dp4a_u8u8(n, reps).print();
    ptx_runner::bench_int8_dp4a_s8u8(n, reps).print();
    ptx_runner::bench_int4_via_dp4a(n, reps).print();
    ptx_runner::bench_bitwise_xnor(n, reps).print();

    if sm_major >= 7 && sm_minor >= 5 {
        println!();
        println!("--- WMMA (Tensor Cores, SM 7.5) ---");
        ptx_runner::bench_wmma_fp16_f32(sm_major, sm_minor, 10000).print();
        ptx_runner::bench_wmma_fp16_f16(sm_major, sm_minor, 10000).print();
        ptx_runner::bench_wmma_int8_s32(sm_major, sm_minor, 10000).print();
    }
}

fn run_fma_compare(n: u32, reps: u32) {
    println!("\n=== FP32 FMA Comparison (CMP mining card diagnostic) ===");
    let ma  = ptx_runner::bench_fp32_muladd(n, reps);
    let fma = ptx_runner::bench_fp32_fma(n, reps);
    ma.print();
    fma.print();
    let ratio = fma.perf / ma.perf;
    println!();
    if ratio < 0.5 {
        println!("⚠️  FMA BROKEN: {:.2}x SLOWER than mul+add!", 1.0/ratio);
        println!("   This is a NVIDIA CMP mining card with disabled FMA units.");
        println!("   Use CUBLAS_PEDANTIC_MATH and fp16 f16x2 mul+add for best performance.");
    } else {
        println!("✅ FMA OK: {:.1}x speedup vs mul+add (correct hardware)", ratio);
    }

    println!("\n--- FP16 comparison ---");
    ptx_runner::bench_fp16_muladd(n, reps).print();
    ptx_runner::bench_fp16_fma(n, reps).print();
}

fn run_cmp_detect(n: u32, reps: u32, sm_major: u32, sm_minor: u32, gpu_name: &str) {
    println!("\n=== CMP / Broken Hardware Auto-Detect ===");
    println!("GPU: {} (SM {}.{})", gpu_name, sm_major, sm_minor);

    let fp32_ma  = ptx_runner::bench_fp32_muladd(n, reps);
    let fp32_fma = ptx_runner::bench_fp32_fma(n, reps);
    let fp16_ma  = ptx_runner::bench_fp16_muladd(n, reps);
    let int8     = ptx_runner::bench_int8_dp4a_s8s8(n, reps);

    fp32_ma.print();
    fp32_fma.print();
    fp16_ma.print();
    int8.print();

    let fma_ratio = fp32_fma.perf / fp32_ma.perf;
    let fma_broken = fma_ratio < 0.5;

    println!("\n--- Quick cuBLAS probe ---");
    let cublas = cublas_probe::run_cublas_probe(2048, 10);
    let tc_bad = cublas.iter().any(|r| r.label.contains("16F DEFAULT") && !r.ok);
    cublas.iter().take(3).for_each(|r| r.print()); // show first 3 modes

    println!("\n=== VERDICT ===");
    let mut flags = 0u32;

    if fma_broken {
        println!("  ⚠️  [FMA] BROKEN — {:.1}x slower than mul+add (CMP card!)", 1.0/fma_ratio);
        flags |= 1;
    } else {
        println!("  ✅  [FMA] OK ({:.1}x speedup)", fma_ratio);
    }

    if !tc_bad && (sm_major >= 7 && sm_minor >= 5) {
        println!("  ✅  [TC]  Tensor Cores: results correct");
    } else if sm_major < 7 {
        println!("  ℹ️  [TC]  Pascal — no Tensor Cores (expected)");
    } else {
        println!("  ⚠️  [TC]  BROKEN or disabled — cuBLAS 16F gives wrong results");
        flags |= 2;
    }

    if sm_major == 7 && int8.perf < 5.0 {
        println!("  ⚠️  [INT8] Suspiciously slow: {:.1} TOPS (expected >10 TOPS for SM 7.5)", int8.perf);
        flags |= 4;
    } else {
        println!("  ✅  [INT8] dp4a: {:.1} TOP/s", int8.perf);
    }

    println!();
    if flags == 0 {
        println!("  ✅ HEALTHY GPU — all paths operating normally.");
    } else {
        println!("  ⚠️  CMP / BROKEN CARD DETECTED (flags=0b{:04b})", flags);
        println!();
        println!("  RECOMMENDED SETTINGS:");
        if flags & 1 != 0 {
            println!("    CUBLAS_PEDANTIC_MATH=1  (avoids broken FMA)");
            println!("    Use f16x2 mul+add ({:.1} TFLOPS) as primary compute path", fp16_ma.perf);
        }
        if flags & 2 != 0 {
            println!("    CUBLAS_COMPUTE_32F  (not 16F — TC give wrong results)");
        }
        if flags & 4 != 0 {
            println!("    GGML_CUDA_FORCE_MMQ=0  (INT8 MMQ broken, use cuBLAS F32)");
        }
        println!();
        println!("  BEST GGUF FORMAT on this card:");
        if fp16_ma.perf > 15.0 {
            println!("    F16 (f16x2 mul+add path) for small models");
        }
        println!("    Q4_K_M (dp4a MMQ) for 7B models if dp4a works");
        println!("    Q6_K (cuBLAS F32) as safe fallback");
    }
}

fn run_full_suite(
    n: u32, reps: u32, m: i32, giters: u32,
    sm_major: u32, sm_minor: u32,
    gpu_name: &str, vram_gb: f32, mem_bw_gbs: f32
) -> BenchResults {
    println!("  [1/6] FP32 benchmarks...");
    let fp32_ma  = ptx_runner::bench_fp32_muladd(n, reps);
    let fp32_fma = ptx_runner::bench_fp32_fma(n, reps);
    let fp64_ma  = ptx_runner::bench_fp64_muladd(n, reps);

    println!("  [2/6] FP16 benchmarks...");
    let fp16_ma  = ptx_runner::bench_fp16_muladd(n, reps);
    let fp16_fma = ptx_runner::bench_fp16_fma(n, reps);

    println!("  [3/6] INT8 dp4a variants...");
    let int8_s8s8 = ptx_runner::bench_int8_dp4a_s8s8(n, reps);
    let int8_u8u8 = ptx_runner::bench_int8_dp4a_u8u8(n, reps);
    let int8_s8u8 = ptx_runner::bench_int8_dp4a_s8u8(n, reps);
    let int4      = ptx_runner::bench_int4_via_dp4a(n, reps);
    let bitwise   = ptx_runner::bench_bitwise_xnor(n, reps);
    let int32     = ptx_runner::bench_int32_mad(n, reps);

    println!("  [4/6] WMMA Tensor Core probe...");
    let wmma_f32 = ptx_runner::bench_wmma_fp16_f32(sm_major, sm_minor, 10000);
    let wmma_f16 = ptx_runner::bench_wmma_fp16_f16(sm_major, sm_minor, 10000);
    let wmma_i8  = ptx_runner::bench_wmma_int8_s32(sm_major, sm_minor, 10000);

    println!("  [5/6] cuBLAS GEMM probe ({}x{}x{}, {} iters)...", m, m, m, giters);
    let cublas = cublas_probe::run_cublas_probe(m, giters);

    println!("  [6/6] Building report...");

    BenchResults {
        gpu_name:   gpu_name.to_string(),
        sm_major, sm_minor, vram_gb, mem_bw_gbs,

        fp32_muladd: Some(fp32_ma.perf),
        fp32_fma:    Some(fp32_fma.perf),
        fp64_muladd: Some(fp64_ma.perf),
        fp16_muladd: Some(fp16_ma.perf),
        fp16_fma:    Some(fp16_fma.perf),
        int8_s8s8:   Some(int8_s8s8.perf),
        int8_u8u8:   Some(int8_u8u8.perf),
        int8_s8u8:   Some(int8_s8u8.perf),
        int4_dp4a:   Some(int4.perf),
        bitwise:     Some(bitwise.perf),
        int32_mad:   Some(int32.perf),

        wmma_fp16_f32: if sm_major >= 7 && sm_minor >= 5 {
            Some((wmma_f32.perf, wmma_f32.ok))
        } else { None },
        wmma_fp16_f16: if sm_major >= 7 && sm_minor >= 5 {
            Some((wmma_f16.perf, wmma_f16.ok))
        } else { None },
        wmma_int8: if sm_major >= 7 && sm_minor >= 5 {
            Some((wmma_i8.perf, wmma_i8.ok))
        } else { None },

        cublas,
    }
}

fn print_gguf_format_table() {
    let sep = "─".repeat(100);
    println!("\n{}", "═".repeat(100));
    println!("  GGUF Format → GPU Instruction Path Mapping");
    println!("  Pascal (SM 6.x): no TC  |  Turing RTX (SM 7.5): has TC  |  Turing CMP: broken TC");
    println!("{}", "═".repeat(100));
    println!("{:<12} {:>5} {:>7} {:>13}  {:<14}  {:<14}  {}",
             "Format", "bpw", "7B GB", "Type/Family",
             "GPU Path", "Pascal/CMP", "Notes");
    println!("{}", sep);

    let rows: &[(&str, f32, f32, &str, &str, &str, &str)] = &[
        // format, bpw, size7b, family, path, pascal/cmp, notes
        ("F32",     32.0, 28.0, "Float",    "cuBLAS F32",   "★★ ok",   "Full precision, large"),
        ("F16",     16.0, 14.0, "Float",    "cuBLAS F32*",  "★★ *32F", "Pascal: MUST use COMPUTE_32F"),
        ("BF16",    16.0, 14.0, "Float",    "❌ N/A",       "❌ N/A",  "Requires SM 8.0+"),
        ("Q8_0",     8.5,  8.0, "Legacy",   "dp4a  MMQ",    "★★★★★",  "Fastest+best on Pascal!"),
        ("Q5_1",     5.9,  5.5, "Legacy",   "cuBLAS F32",   "★★",      "Legacy, no MMQ. Use Q5_K_M"),
        ("Q5_0",     5.5,  5.1, "Legacy",   "cuBLAS F32",   "★★",      "Legacy, no MMQ"),
        ("Q4_1",     4.9,  4.6, "Legacy",   "cuBLAS F32",   "★★",      "Legacy. Use Q4_K_M instead"),
        ("Q4_0",     4.5,  4.2, "Legacy",   "dp4a  MMQ",    "★★★",     "Has MMQ path but K-quants better"),
        ("Q6_K",     6.6,  6.1, "K-Quant",  "cuBLAS F32",   "★★",      "Best quality FP32 path"),
        ("Q5_K_M",   5.7,  5.3, "K-Quant",  "cuBLAS F32",   "★★",      "Good quality/size"),
        ("Q5_K_S",   5.5,  5.1, "K-Quant",  "cuBLAS F32",   "★★",      ""),
        ("Q4_K_M",   4.8,  4.5, "K-Quant",  "dp4a  MMQ",    "★★★★★",  "← MAIN RECOMMENDATION"),
        ("Q4_K_S",   4.4,  4.1, "K-Quant",  "dp4a  MMQ",    "★★★★",   ""),
        ("Q3_K_L",   4.0,  3.8, "K-Quant",  "cuBLAS F32",   "★★",      "Mixed Q3+Q4 layers"),
        ("Q3_K_M",   3.74, 3.5, "K-Quant",  "cuBLAS F32",   "★★",      ""),
        ("Q3_K_S",   3.41, 3.2, "K-Quant",  "cuBLAS F32",   "★★",      ""),
        ("Q2_K",     2.63, 2.5, "K-Quant",  "cuBLAS F32",   "★",       "Bad quality, use IQ2_M"),
        ("Q2_K_S",   2.63, 2.5, "K-Quant",  "cuBLAS F32",   "★",       "Slightly better Q2_K"),
        ("IQ4_XS",   4.25, 4.0, "I-Quant",  "dp4a  MMQ",    "★★★★★",  "← Pareto★ (needs imatrix)"),
        ("IQ4_NL",   4.50, 4.2, "I-Quant",  "dp4a  MMQ",    "★★★★",   "Non-linear LUT, fast"),
        ("IQ3_M",    3.66, 3.4, "I-Quant",  "cuBLAS F32",   "★★",      "3-bit Pareto"),
        ("IQ3_S",    3.44, 3.2, "I-Quant",  "cuBLAS F32",   "★★",      ""),
        ("IQ3_XS",   3.30, 3.1, "I-Quant",  "cuBLAS F32",   "★★",      ""),
        ("IQ3_XXS",  3.06, 2.9, "I-Quant",  "cuBLAS F32",   "★★",      ""),
        ("IQ2_M",    2.70, 2.5, "I-Quant",  "cuBLAS F32",   "★★",      "Best 2-bit on GPU"),
        ("IQ2_S",    2.50, 2.3, "I-Quant",  "cuBLAS F32",   "★★",      ""),
        ("IQ2_XS",   2.31, 2.2, "I-Quant",  "cuBLAS F32",   "★★",      ""),
        ("IQ2_XXS",  2.06, 1.9, "I-Quant",  "cuBLAS F32",   "★",       ""),
        ("IQ1_M",    1.75, 1.6, "I-Quant",  "FP32  GEMV",   "★",       "No CUDA MMQ kernel"),
        ("IQ1_S",    1.56, 1.5, "I-Quant",  "FP32  GEMV",   "★",       "No CUDA MMQ kernel"),
        ("TQ2_0",    2.06, 1.9, "Ternary",  "FP32  GEMV",   "★",       "Ternary {-1,0,+1} GPU slow"),
        ("TQ1_0",    1.69, 1.6, "Ternary",  "FP32  GEMV",   "★",       "Ternary packed, GPU slow"),
        ("turbo4",   4.0,  3.8, "TurboQ",   "dp4a (TQ4_1S)","★★★",    "KV-cache only, 3.5x fast"),
        ("turbo3",   3.0,  2.8, "TurboQ",   "FP32  dequant","★★",      "KV-cache only"),
        ("turbo2",   2.0,  1.9, "TurboQ",   "FP32  dequant","★",       "KV-cache only, long ctx"),
    ];

    for (fmt, bpw, size7b, family, path, pascal, notes) in rows {
        println!("{:<12} {:>5.2} {:>6.1}G {:<13}  {:<14}  {:<14}  {}",
                 fmt, bpw, size7b, family, path, pascal, notes);
    }

    println!("{}", "═".repeat(100));
    println!("  * cuBLAS F32: Pascal must use CUBLAS_COMPUTE_32F (not COMPUTE_16F → BAD results!)");
    println!("  dp4a MMQ: fastest path on Pascal! 7x faster than cuBLAS for inference.");
    println!("  BF16/TF32/FP8: NOT available on Pascal (SM 6.x) or Turing (SM 7.5).");
    println!("  TurboQuant (turbo2/3/4): KV-cache quantization only, not weight quantization.");
}
