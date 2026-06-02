// ============================================================
// Report: diagnosis, GGUF recommendations, GPU comparison
// ============================================================

use crate::gpu_db::GPU_DB;
use crate::cublas_probe::GemmResult;

pub struct BenchResults {
    pub gpu_name:   String,
    pub sm_major:   u32,
    pub sm_minor:   u32,
    pub vram_gb:    f32,
    pub mem_bw_gbs: f32,

    // PTX results
    pub fp32_muladd: Option<f64>,
    pub fp32_fma:    Option<f64>,
    pub fp64_muladd: Option<f64>,
    pub fp16_muladd: Option<f64>,
    pub fp16_fma:    Option<f64>,
    pub int8_s8s8:   Option<f64>,
    pub int8_u8u8:   Option<f64>,
    pub int8_s8u8:   Option<f64>,
    pub int4_dp4a:   Option<f64>,
    pub bitwise:     Option<f64>,
    pub int32_mad:   Option<f64>,

    // WMMA (Turing only)
    pub wmma_fp16_f32: Option<(f64, bool)>,  // (tflops, ok)
    pub wmma_fp16_f16: Option<(f64, bool)>,
    pub wmma_int8:     Option<(f64, bool)>,

    // cuBLAS
    pub cublas: Vec<GemmResult>,
}

impl BenchResults {
    /// Auto-detect pathologies
    pub fn detect_fma_broken(&self) -> bool {
        match (self.fp32_muladd, self.fp32_fma) {
            (Some(ma), Some(fma)) => fma < ma * 0.5,
            _ => false,
        }
    }
    pub fn detect_fp16_tc_bad(&self) -> bool {
        self.cublas.iter().any(|r| r.label.contains("16F DEFAULT") && !r.ok)
    }
    pub fn detect_int8_suspicious(&self) -> bool {
        // On Turing we expect >10 TOPS from INT8 dp4a
        // CMP 50HX measured ~1.7 TOPS — flag it
        if self.sm_major == 7 {
            self.int8_s8s8.map(|v| v < 5.0).unwrap_or(false)
        } else {
            false
        }
    }
    pub fn best_cublas_ok(&self) -> Option<&GemmResult> {
        self.cublas.iter().filter(|r| r.ok).max_by(|a,b| a.tflops.partial_cmp(&b.tflops).unwrap())
    }
}

pub fn print_full_report(r: &BenchResults) {
    let sep = "═".repeat(76);
    let dash = "─".repeat(76);

    // ── Header ──────────────────────────────────────────────
    println!("\n{}", sep);
    println!("  nbenchmark v2.0 — FULL REPORT");
    println!("  GPU:  {} (SM {}.{})", r.gpu_name, r.sm_major, r.sm_minor);
    println!("  VRAM: {:.0} GB    Bandwidth: {:.0} GB/s", r.vram_gb, r.mem_bw_gbs);
    println!("{}", sep);

    // Look up known spec
    let known = GPU_DB.iter().find(|g| {
        g.sm_major == r.sm_major && g.sm_minor == r.sm_minor &&
        r.gpu_name.to_lowercase().contains(&g.name.to_lowercase().split(' ').next().unwrap_or(""))
    });

    // ── PTX arithmetic ──────────────────────────────────────
    println!("\n[PTX ARITHMETIC — raw instruction throughput]");
    println!("{:<44} {:>10}  {:>10}", "Mode", "TFLOP/s", "Notes");
    println!("{}", dash);

    macro_rules! ptx_row {
        ($label:expr, $val:expr, $note:expr) => {
            if let Some(v) = $val {
                println!("{:<44} {:>10.3}  {}", $label, v, $note);
            }
        }
    }

    ptx_row!("FP64 mul+add",   r.fp64_muladd, fp64_note(r));
    ptx_row!("FP32 mul+add",   r.fp32_muladd, "← safe baseline");
    ptx_row!("FP32 FMA",       r.fp32_fma,    fma_note(r));
    ptx_row!("FP16 f16x2 mul+add", r.fp16_muladd, "← best on CMP 50HX");
    ptx_row!("FP16 f16x2 FMA", r.fp16_fma,    "");
    ptx_row!("INT32 MAD",      r.int32_mad,   "accumulator baseline");
    ptx_row!("INT8 dp4a s8×s8 (Q8_0)",  r.int8_s8s8, int8_note(r));
    ptx_row!("INT8 dp4a u8×u8",         r.int8_u8u8, "");
    ptx_row!("INT8 dp4a s8×u8 (Q8_1)",  r.int8_s8u8, "asymmetric quant");
    ptx_row!("INT4 via dp4a (Q4_K/Q4_0)", r.int4_dp4a, "unpack overhead included");
    ptx_row!("Bitwise XNOR+POPC (1-bit)", r.bitwise, "binary neural nets");

    // ── WMMA ────────────────────────────────────────────────
    if r.sm_major >= 7 && r.sm_minor >= 5 {
        println!("\n[WMMA — direct PTX Tensor Core test (bypasses cuBLAS)]");
        println!("{:<44} {:>10}  {:>6}", "Mode", "TFLOP/s", "Check");
        println!("{}", dash);
        macro_rules! wmma_row {
            ($label:expr, $opt:expr, $unit:expr) => {
                if let Some((v, ok)) = $opt {
                    let chk = if *ok { "OK" } else { "BAD!" };
                    println!("{:<44} {:>10.2}  {:>6}", $label, v, chk);
                }
            }
        }
        wmma_row!("WMMA FP16→FP32 m16n16k16", &r.wmma_fp16_f32, "TFLOP/s");
        wmma_row!("WMMA FP16→FP16 m16n16k16", &r.wmma_fp16_f16, "TFLOP/s");
        wmma_row!("WMMA INT8→INT32 m8n32k16 (Turing IMMA)", &r.wmma_int8, "TOP/s");
    }

    // ── cuBLAS ──────────────────────────────────────────────
    println!("\n[cuBLAS GEMM PROBE]");
    println!("{:<46} {:>10}  {:>10} {:>6}", "Mode", "Time(s)", "TFLOP/s", "Check");
    println!("{}", dash);
    for res in &r.cublas {
        res.print();
    }
    if let Some(best) = r.best_cublas_ok() {
        println!("{}", dash);
        println!("Best correct: {} ({:.2} TFLOP/s)", best.label, best.tflops);
    }

    // ── Diagnosis ───────────────────────────────────────────
    println!("\n{}", sep);
    println!("[DIAGNOSIS]");
    println!("{}", sep);

    let fma_broken = r.detect_fma_broken();
    let tc_bad     = r.detect_fp16_tc_bad();
    let int8_sus   = r.detect_int8_suspicious();
    let has_tc     = known.map(|g| g.has_tensor_cores).unwrap_or(r.sm_major >= 7 && r.sm_minor >= 5);

    if fma_broken {
        let ratio = r.fp32_fma.unwrap_or(1.0) / r.fp32_muladd.unwrap_or(1.0);
        println!("  ⚠️  FMA BROKEN: FP32 FMA is {:.1}x SLOWER than mul+add", 1.0/ratio);
        println!("      This is a NVIDIA CMP mining card with disabled FMA units.");
        println!("      → Use CUBLAS_PEDANTIC_MATH to avoid FMA in cuBLAS.");
        println!("      → Use fp16 f16x2 mul+add path for best performance.");
    } else if let (Some(ma), Some(fma)) = (r.fp32_muladd, r.fp32_fma) {
        println!("  ✅  FMA OK: {:.1}x speedup over mul+add", fma/ma);
    }

    if !has_tc {
        println!("  ℹ️  No Tensor Cores (Pascal SM 6.x or TU117 Quadro T400/T600).");
        println!("      cuBLAS 16F compute → BAD results. Use COMPUTE_32F always.");
    } else if tc_bad {
        println!("  ⚠️  TENSOR CORES: cuBLAS 16F gives WRONG results (BAD).");
        println!("      TC physically present but disabled on this CMP card.");
    } else {
        println!("  ✅  Tensor Cores: operating correctly.");
    }

    if int8_sus {
        println!("  ⚠️  INT8 dp4a suspiciously slow ({:.1} TOP/s) for Turing.",
                 r.int8_s8s8.unwrap_or(0.0));
        println!("      Expected >10 TOPS for any SM 7.5 with working INT8 TC.");
        println!("      INT8 hardware path may be partially disabled on this CMP.");
    }

    // FP64 ratio
    if let (Some(f32p), Some(f64p)) = (r.fp32_muladd, r.fp64_muladd) {
        let ratio = f64p / f32p;
        if ratio < 0.1 {
            println!("  ℹ️  FP64 disabled: {:.1}x of FP32 ({:.3} TFLOPS)",
                     ratio, f64p);
            println!("      Consumer/mining card — FP64 blocked in firmware.");
        } else if ratio > 0.4 {
            println!("  ✅  FP64 enabled: {:.1}x of FP32 (datacenter-grade GPU!)",
                     ratio);
        }
    }

    // Quadro T400 special note
    if r.gpu_name.to_lowercase().contains("t400") || r.gpu_name.to_lowercase().contains("t 400") {
        println!("  ℹ️  Quadro T400: TU117 die — NO Tensor Cores despite SM 7.5.");
        println!("      Only 80 GB/s memory bandwidth (64-bit bus).");
        println!("      Suitable only for <3B models at minimal VRAM.");
    }

    // ── Recommendations ─────────────────────────────────────
    println!("\n{}", sep);
    println!("[RECOMMENDATIONS]");
    println!("{}", sep);

    print_pretrain_rec(r, fma_broken, tc_bad, has_tc);
    print_gguf_rec(r, fma_broken, tc_bad);
}

fn fp64_note(r: &BenchResults) -> &'static str {
    match r.fp64_muladd {
        Some(v) if v > 4.0  => "HBM GPU (GP100/V100?) — full FP64",
        Some(v) if v > 0.5  => "partial FP64",
        Some(_)             => "← FP64 hardware-disabled (consumer/mining)",
        None                => "",
    }
}
fn fma_note(r: &BenchResults) -> &'static str {
    if r.detect_fma_broken() { "← ⚠️  BROKEN ON THIS CARD! (CMP mining)" }
    else { "← uses fused multiply-add (correct)" }
}
fn int8_note(r: &BenchResults) -> &'static str {
    if r.detect_int8_suspicious() { "← ⚠️  SUSPICIOUSLY LOW for Turing!" }
    else { "← Q8_0 GGUF inference path" }
}

fn print_pretrain_rec(r: &BenchResults, fma_broken: bool, tc_bad: bool, has_tc: bool) {
    println!("\n  [Pretrain / nanochatGPT]");

    let math = if fma_broken { "CUBLAS_PEDANTIC_MATH" } else { "CUBLAS_DEFAULT_MATH" };
    let compute = if !has_tc || tc_bad { "CUBLAS_COMPUTE_32F" } else { "CUBLAS_COMPUTE_16F or 32F" };

    println!("  cuBLAS math mode : {}", math);
    println!("  cuBLAS compute   : {}", compute);
    println!("  Weight dtype     : FP16  (BF16 requires SM≥8.0 — NOT available here)");
    println!("  Master weights   : FP32  (mandatory for gradient stability)");
    println!("  Optimizer m/v    : FP16 + loss scaling (2x memory saving vs FP32 Adam)");
    println!("  Activations      : FP16");
    println!();
    println!("  // Rust / CUDA:");

    if fma_broken {
        println!("  cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH); // CRITICAL on CMP!");
    }
    if !has_tc || tc_bad {
        println!("  // cublasGemmEx: use CUBLAS_COMPUTE_32F — NOT 16F (gives BAD results!)");
    }

    if let Some(f16_ma) = r.fp16_muladd {
        if f16_ma > 15.0 && fma_broken {
            println!();
            println!("  ★ CMP 50HX tip: f16x2 mul+add ({:.1} TFLOPS) is your fastest path.", f16_ma);
            println!("    Use custom CUDA kernels with __half2 mul + add (not __hfma2).");
        }
    }
}

fn print_gguf_rec(r: &BenchResults, fma_broken: bool, _tc_bad: bool) {
    println!("\n  [GGUF Format Selection for Inference]");
    println!();

    let bw = r.mem_bw_gbs;
    let vram = r.vram_gb;
    let int8_fast = r.int8_s8s8.map(|v| v > 5.0).unwrap_or(false);

    // Estimate tok/s for 7B model
    let tok_q8  = bw * 1e9 / (7.0e9 * 8.5 / 8.0);
    let tok_q4  = bw * 1e9 / (7.0e9 * 4.8 / 8.0);
    let tok_iq4 = bw * 1e9 / (7.0e9 * 4.25 / 8.0);
    let tok_q6  = bw * 1e9 / (7.0e9 * 6.6 / 8.0);

    println!("  Memory bandwidth: {:.0} GB/s → token generation rates (7B, batch=1):", bw);
    println!("  {:<14} {:>5} bpw  {:>6} GB  {:>8.1} tok/s  GPU path",
             "Format", "", "size 7B", "");
    println!("  {}", "─".repeat(68));

    struct Row { fmt: &'static str, bpw: f32, size: f32, toks: f32, path: &'static str, note: &'static str }
    let rows = vec![
        Row { fmt:"Q8_0",    bpw:8.5,  size:8.0,  toks:tok_q8 as f32,  path:"dp4a  MMQ",   note:if vram>=10.0 {"✅ fits"} else {"⚠️ tight"} },
        Row { fmt:"Q6_K",    bpw:6.6,  size:6.1,  toks:tok_q6 as f32,  path:"cuBLAS F32",  note:if vram>=7.5 {"✅"} else {"⚠️ tight"} },
        Row { fmt:"Q5_K_M",  bpw:5.7,  size:5.3,  toks:(bw*1e9/(7.0e9*5.7/8.0)) as f32, path:"cuBLAS F32",  note:"✅" },
        Row { fmt:"IQ4_XS",  bpw:4.25, size:4.0,  toks:tok_iq4 as f32, path:"dp4a  MMQ",   note:"✅ Pareto★ (needs imatrix)" },
        Row { fmt:"Q4_K_M",  bpw:4.8,  size:4.5,  toks:tok_q4 as f32,  path:"dp4a  MMQ",   note:"✅ Main recommendation" },
        Row { fmt:"IQ4_NL",  bpw:4.5,  size:4.2,  toks:(bw*1e9/(7.0e9*4.5/8.0)) as f32,  path:"dp4a  MMQ",   note:"✅" },
        Row { fmt:"IQ3_M",   bpw:3.66, size:3.4,  toks:(bw*1e9/(7.0e9*3.66/8.0)) as f32, path:"cuBLAS F32",  note:"✅ 3-bit Pareto" },
        Row { fmt:"IQ2_M",   bpw:2.70, size:2.5,  toks:(bw*1e9/(7.0e9*2.7/8.0)) as f32,  path:"cuBLAS F32",  note:"⚠️ last resort quality" },
        Row { fmt:"TQ2_0",   bpw:2.06, size:1.9,  toks:(bw*1e9/(7.0e9*2.06/8.0)) as f32, path:"FP32  GEMV",  note:"❌ no GPU kernel, CPU faster" },
        Row { fmt:"IQ1_M",   bpw:1.75, size:1.6,  toks:(bw*1e9/(7.0e9*1.75/8.0)) as f32, path:"FP32  GEMV",  note:"❌ severe quality loss" },
    ];

    for row in &rows {
        println!("  {:<14} {:>5.2}bpw  {:>5.1}GB  {:>8.1} tok/s  {:<14}  {}",
                 row.fmt, row.bpw, row.size, row.toks, row.path, row.note);
    }

    println!();
    println!("  ❌ AVOID on SM {}.{}:", r.sm_major, r.sm_minor);
    println!("     BF16     — not supported (requires SM≥8.0 / Ampere)");
    println!("     TF32     — not supported (requires SM≥8.0)");
    println!("     FP8      — not supported (requires SM≥8.9 / Ada)");
    println!("     Q4_1/Q5_1 — legacy, no dp4a MMQ path, worse than K-quants");
    println!("     Q2_K     — use IQ2_M instead (better quality same size)");
    println!("     IQ1_S    — no CUDA kernel, CPU-only, severe quality degradation");

    if !int8_fast {
        println!("     ⚠️  INT8 dp4a is SLOW on this GPU — Q4_K/Q6_K via cuBLAS may be faster!");
    }

    println!();
    println!("  ★ Pareto frontier (speed × quality × VRAM):");
    if vram >= 10.0 {
        println!("     VRAM ≥ 10GB: Q8_0 > IQ4_XS > Q4_K_M > IQ3_M");
    } else if vram >= 8.0 {
        println!("     VRAM = 8GB:  IQ4_XS > Q4_K_M > IQ3_M (Q8_0 too large for 7B)");
    } else {
        println!("     VRAM = 6GB:  Q4_K_M > IQ4_XS (7B) | Q8_0 for ≤3B models");
    }

    println!();
    println!("  llama.cpp launch flags:");
    if fma_broken {
        println!("     GGML_CUDA_NO_FMA=1 ./llama-server -m model.Q4_K_M.gguf -ngl 99");
    } else {
        println!("     ./llama-server -m model.Q4_K_M.gguf -ngl 99 --flash-attn");
    }
    if !int8_fast {
        println!("     (skip --flash-attn KV quant if INT8 TC are broken)");
    } else {
        println!("     # KV cache quantization (saves VRAM for longer context):");
        println!("     ./llama-server -m model.gguf -ngl 99 -fa -ctk q8_0 -ctv q8_0");
    }
}

/// Print compact comparison table for all known GPUs
pub fn print_gpu_comparison_table() {
    println!("\n{}", "═".repeat(110));
    println!("  Known GPU Specs (Pascal + Turing) — nbenchmark target hardware");
    println!("{}", "═".repeat(110));
    println!("{:<22} {:<18} {:>5} {:>6} {:>7} {:>8} {:>8} {:>9} {:>4} {:>4}  {}",
             "GPU", "Arch", "SM", "VRAM", "BW", "FP32", "FP16", "INT8", "TC", "FMA", "Notes (brief)");
    println!("{}", "─".repeat(110));

    for g in GPU_DB {
        let tc  = if g.has_tensor_cores { "✅" } else { "❌" };
        let fma = match g.fma_broken {
            Some(true)  => "⚠️",
            Some(false) => "✅",
            None        => "?",
        };
        println!("{:<22} {:<18} {:>2}.{:<2} {:>5.0}G {:>6.0}GB/s {:>6.1}TF {:>6.1}TF {:>7.0}T  {}  {}  {}",
                 g.name, g.arch.split_whitespace().last().unwrap_or(g.arch),
                 g.sm_major, g.sm_minor,
                 g.vram_gb, g.mem_bw_gbs,
                 g.fp32_tflops, g.fp16_tflops, g.int8_tops,
                 tc, fma,
                 // Truncate notes
                 &g.notes[..g.notes.len().min(40)]);
    }
    println!("{}", "═".repeat(110));
}
