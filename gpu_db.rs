// ============================================================
// GPU Hardware Database
// Known specs for Pascal and Turing target cards
// ============================================================

#[derive(Debug, Clone)]
pub struct GpuSpec {
    pub name: &'static str,
    pub arch: &'static str,
    pub sm_major: u32,
    pub sm_minor: u32,
    pub cuda_cores: u32,
    pub vram_gb: f32,
    pub mem_bus_bits: u32,
    pub mem_bw_gbs: f32,       // GB/s theoretical
    pub fp32_tflops: f32,      // theoretical peak
    pub fp16_tflops: f32,      // theoretical (native f16x2 = 2x FP32)
    pub int8_tops: f32,        // theoretical dp4a peak
    pub has_tensor_cores: bool,
    pub has_rt_cores: bool,
    pub fma_broken: Option<bool>,    // None = unknown, Some(true) = confirmed broken
    pub fp16_tc_broken: Option<bool>,
    pub notes: &'static str,
}

/// Match a GPU by PCI device ID or partial name substring
pub fn lookup_gpu_by_name(name: &str) -> Option<&'static GpuSpec> {
    GPU_DB.iter().find(|g| {
        name.to_lowercase().contains(&g.name.to_lowercase())
    })
}

pub fn lookup_gpu_by_sm(sm_major: u32, sm_minor: u32, cuda_cores: u32) -> Option<&'static GpuSpec> {
    // First try exact cuda_cores match
    if let Some(g) = GPU_DB.iter().find(|g| {
        g.sm_major == sm_major && g.sm_minor == sm_minor && g.cuda_cores == cuda_cores
    }) {
        return Some(g);
    }
    // Fallback: closest by SM + approximate cores
    GPU_DB.iter().find(|g| g.sm_major == sm_major && g.sm_minor == sm_minor)
}

pub static GPU_DB: &[GpuSpec] = &[

    // ══════════════════════════════════════════════════════════════
    // PASCAL — SM 6.0 (GP100, HBM2) — Full FP64, no TC
    // ══════════════════════════════════════════════════════════════
    GpuSpec {
        name: "Tesla P100",
        arch: "Pascal GP100",
        sm_major: 6, sm_minor: 0,
        cuda_cores: 3584,
        vram_gb: 16.0,
        mem_bus_bits: 4096,
        mem_bw_gbs: 732.0,
        fp32_tflops: 9.3,
        fp16_tflops: 18.7,
        int8_tops: 37.4,      // dp4a estimated (no dedicated INT8 path on P100)
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false), // no TC, but FP32 cuBLAS works fine
        notes: "HBM2 732GB/s, FP64=4.7TF (full), FP16 f16x2 fast, no TC. Best Pascal for inference.",
    },

    // ══════════════════════════════════════════════════════════════
    // PASCAL — SM 6.1 (GP102/GP104/GP106) — Mining cards
    // ══════════════════════════════════════════════════════════════
    GpuSpec {
        name: "P102-100",
        arch: "Pascal GP102",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 3840,
        vram_gb: 10.0,
        mem_bus_bits: 320,
        mem_bw_gbs: 320.0,
        fp32_tflops: 11.0,    // theoretical; measured ~5.6 mul+add / ~10.5 FMA
        fp16_tflops: 22.0,    // f16x2, cuBLAS 16F → BAD (no TC)
        int8_tops: 44.0,      // dp4a; measured 41.6
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),   // FMA works on P102 (not a CMP!)
        fp16_tc_broken: Some(true), // No TC — cuBLAS 16F gives wrong results
        notes: "No TC. INT8 dp4a 41 TOPS is the fastest path. cuBLAS must use COMPUTE_32F.",
    },
    GpuSpec {
        name: "P104-100",
        arch: "Pascal GP104",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 2560,
        vram_gb: 8.0,
        mem_bus_bits: 256,
        mem_bw_gbs: 256.0,
        fp32_tflops: 7.3,
        fp16_tflops: 14.6,
        int8_tops: 29.0,
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "No TC. Similar to P102 but 8GB/256-bit. dp4a ~15 TOPS measured.",
    },
    GpuSpec {
        name: "P106-100",
        arch: "Pascal GP106",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 1280,
        vram_gb: 6.0,
        mem_bus_bits: 192,
        mem_bw_gbs: 192.0,
        fp32_tflops: 4.6,
        fp16_tflops: 9.2,
        int8_tops: 18.0,
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "6GB only. No TC. Smallest Pascal mining card. Q4_K_M best GGUF for 7B.",
    },

    // Consumer Pascal (for completeness / comparison)
    GpuSpec {
        name: "GTX 1080 Ti",
        arch: "Pascal GP102",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 3584,
        vram_gb: 11.0,
        mem_bus_bits: 352,
        mem_bw_gbs: 484.0,
        fp32_tflops: 11.3,
        fp16_tflops: 22.6,
        int8_tops: 45.0,
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "Full GP102. 11GB 352-bit. FMA works normally (not a mining CMP).",
    },
    GpuSpec {
        name: "GTX 1080",
        arch: "Pascal GP104",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 2560,
        vram_gb: 8.0,
        mem_bus_bits: 256,
        mem_bw_gbs: 320.0,
        fp32_tflops: 8.9,
        fp16_tflops: 17.8,
        int8_tops: 35.0,
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "GP104 full. 8GB 256-bit. Solid INT8 dp4a performance.",
    },
    GpuSpec {
        name: "GTX 1060",
        arch: "Pascal GP106",
        sm_major: 6, sm_minor: 1,
        cuda_cores: 1280,
        vram_gb: 6.0,
        mem_bus_bits: 192,
        mem_bw_gbs: 192.0,
        fp32_tflops: 4.4,
        fp16_tflops: 8.8,
        int8_tops: 17.5,
        has_tensor_cores: false,
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "Consumer version of P106. Same perf as P106-100.",
    },

    // ══════════════════════════════════════════════════════════════
    // TURING — SM 7.5
    // ══════════════════════════════════════════════════════════════

    // --- CMP Mining (broken FMA + broken TC) ---
    GpuSpec {
        name: "CMP 50HX",
        arch: "Turing TU102/TU104 (mining)",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 2304,
        vram_gb: 10.0,
        mem_bus_bits: 320,
        mem_bw_gbs: 320.0,
        fp32_tflops: 5.9,     // mul+add measured; FMA: 0.42 (BROKEN 14x slower!)
        fp16_tflops: 22.3,    // f16x2 mul+add (FMA path broken)
        int8_tops: 1.7,       // dp4a measured — suspiciously low, TC likely broken
        has_tensor_cores: true,  // physically present but disabled/broken
        has_rt_cores: false,
        fma_broken: Some(true),    // CONFIRMED: 14x slower than mul+add
        fp16_tc_broken: Some(true), // CONFIRMED: BAD in cuBLAS
        notes: "BROKEN: FMA 14x slow, TC give wrong results, INT8 only 1.7 TOPS. Use f16x2 mul+add!",
    },

    // --- RTX 2060 6GB (original) ---
    GpuSpec {
        name: "RTX 2060",
        arch: "Turing TU106",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 1920,
        vram_gb: 6.0,
        mem_bus_bits: 192,
        mem_bw_gbs: 336.0,
        fp32_tflops: 6.5,
        fp16_tflops: 52.0,    // with Tensor Cores FP16
        int8_tops: 104.0,     // INT8 TC (2x FP16 TC)
        has_tensor_cores: true,
        has_rt_cores: true,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false),
        notes: "Full Turing TC. 48 RT cores, 240 Tensor cores. FP16 TC 52 TFLOPS. INT8 TC ~104 TOPS.",
    },

    // --- RTX 2060 SUPER 8GB ---
    GpuSpec {
        name: "RTX 2060 SUPER",
        arch: "Turing TU106-410",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 2176,
        vram_gb: 8.0,
        mem_bus_bits: 256,
        mem_bw_gbs: 448.0,
        fp32_tflops: 7.2,
        fp16_tflops: 57.4,    // FP16 TC (272 Tensor Cores × 2176 CUDA)
        int8_tops: 114.8,     // INT8 TC ~2x FP16
        has_tensor_cores: true,
        has_rt_cores: true,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false),
        notes: "256-bit 448GB/s. 34 RT cores, 272 Tensor cores. Best RTX 2060 variant for throughput.",
    },

    // --- RTX 2060 12GB (late 2021) ---
    // Same TU106-410 die as Super, but 192-bit bus and higher boost clock
    GpuSpec {
        name: "RTX 2060 12GB",
        arch: "Turing TU106-300",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 2176,
        vram_gb: 12.0,
        mem_bus_bits: 192,
        mem_bw_gbs: 360.0,   // 15 Gbps × 192-bit
        fp32_tflops: 7.3,    // 2176 cores × 1680 MHz × 2 = 7.3
        fp16_tflops: 58.2,
        int8_tops: 116.0,
        has_tensor_cores: true,
        has_rt_cores: true,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false),
        notes: "12GB but 192-bit bus (360 GB/s). Same cores as Super, more VRAM, lower BW than Super. \
                Fits larger GGUF models (13B Q4_K_M = 8GB). 2060 SUPER faster for throughput.",
    },

    // --- Quadro T400 (Turing, entry workstation) ---
    GpuSpec {
        name: "Quadro T400",
        arch: "Turing TU117",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 384,
        vram_gb: 4.0,        // 2GB or 4GB variants
        mem_bus_bits: 64,
        mem_bw_gbs: 80.0,    // 64-bit 10 Gbps
        fp32_tflops: 1.094,
        fp16_tflops: 2.19,   // No TC listed in official specs! f16x2 only
        int8_tops: 4.4,      // dp4a estimated
        has_tensor_cores: false,  // TU117 in T400 does NOT have TC!
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true), // No TC on TU117
        notes: "CAUTION: TU117 in T400 has NO Tensor Cores despite SM 7.5! \
                30W TDP, 80 GB/s BW. Inference only for tiny models (1-3B). \
                dp4a works (SM 6.1+), but extremely low throughput. \
                Q4_K_M 3B = ~2.4GB fits. ~22 tok/s for 3B Q4_K_M.",
    },

    // Additional Quadro T for completeness
    GpuSpec {
        name: "Quadro T600",
        arch: "Turing TU117",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 640,
        vram_gb: 4.0,
        mem_bus_bits: 128,
        mem_bw_gbs: 160.0,
        fp32_tflops: 1.71,
        fp16_tflops: 3.42,
        int8_tops: 6.8,
        has_tensor_cores: false,  // TU117 — no TC
        has_rt_cores: false,
        fma_broken: Some(false),
        fp16_tc_broken: Some(true),
        notes: "TU117, no TC. 4GB 128-bit 160 GB/s. Better than T400, still entry level.",
    },

    // RTX 2060 for context (standard)
    GpuSpec {
        name: "RTX 2070",
        arch: "Turing TU106",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 2304,
        vram_gb: 8.0,
        mem_bus_bits: 256,
        mem_bw_gbs: 448.0,
        fp32_tflops: 7.5,
        fp16_tflops: 57.8,
        int8_tops: 115.0,
        has_tensor_cores: true,
        has_rt_cores: true,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false),
        notes: "8GB 256-bit. Full Turing TC. Good for 7B/13B models.",
    },
    GpuSpec {
        name: "RTX 2080 Ti",
        arch: "Turing TU102",
        sm_major: 7, sm_minor: 5,
        cuda_cores: 4352,
        vram_gb: 11.0,
        mem_bus_bits: 352,
        mem_bw_gbs: 616.0,
        fp32_tflops: 13.4,
        fp16_tflops: 107.9,
        int8_tops: 215.8,
        has_tensor_cores: true,
        has_rt_cores: true,
        fma_broken: Some(false),
        fp16_tc_broken: Some(false),
        notes: "Best consumer Turing. 11GB 352-bit. TC INT8 ~215 TOPS. Excellent for 13B+ models.",
    },
];

impl GpuSpec {
    /// Recommended GGUF format given available VRAM
    pub fn gguf_recommendations(&self) -> Vec<GgufRec> {
        let vram = self.vram_gb;
        let _bw = self.mem_bw_gbs;
        let _has_fast_dp4a = self.int8_tops > 5.0;

        // Model sizes in GB for each format (approx for 7B)
        // Formula: params * bpw / 8
        vec![
            GgufRec { format: "Q8_0",    bpw: 8.5,  size_7b: 8.0,  path: GpuPath::Dp4a,      notes: "Fastest on GPU (dp4a), near-lossless" },
            GgufRec { format: "Q6_K",    bpw: 6.6,  size_7b: 6.1,  path: GpuPath::CublasF32, notes: "Best quality after Q8_0, cuBLAS FP32 path" },
            GgufRec { format: "Q5_K_M",  bpw: 5.7,  size_7b: 5.3,  path: GpuPath::CublasF32, notes: "Good quality/size, FP32 cuBLAS" },
            GgufRec { format: "IQ4_XS",  bpw: 4.25, size_7b: 4.0,  path: GpuPath::Dp4a,      notes: "Pareto-optimal dp4a, needs imatrix" },
            GgufRec { format: "Q4_K_M",  bpw: 4.8,  size_7b: 4.5,  path: GpuPath::Dp4a,      notes: "Main recommendation: dp4a speed + quality" },
            GgufRec { format: "IQ4_NL",  bpw: 4.5,  size_7b: 4.2,  path: GpuPath::Dp4a,      notes: "Non-linear 4-bit, dp4a fast" },
            GgufRec { format: "Q4_K_S",  bpw: 4.4,  size_7b: 4.1,  path: GpuPath::Dp4a,      notes: "Small variant, dp4a" },
            GgufRec { format: "IQ3_M",   bpw: 3.66, size_7b: 3.4,  path: GpuPath::CublasF32, notes: "3-bit Pareto, FP32 cuBLAS" },
            GgufRec { format: "Q3_K_M",  bpw: 3.74, size_7b: 3.5,  path: GpuPath::CublasF32, notes: "3-bit K-quant, FP32 cuBLAS" },
            GgufRec { format: "IQ2_M",   bpw: 2.70, size_7b: 2.5,  path: GpuPath::CublasF32, notes: "2-bit, use only if nothing else fits" },
        ]
        .into_iter()
        .filter(|r| {
            // Check if model fits with 20% overhead for KV cache
            let fits = r.size_7b * 1.2 <= vram as f32;
            // If path is Dp4a but dp4a is broken/slow, note it
            fits
        })
        .collect()
    }

    /// Estimated tokens/sec for 7B model, batch=1 (memory-bound)
    pub fn est_tokens_per_sec(&self, bpw: f32) -> f32 {
        // bytes_per_param = bpw / 8
        // total_bytes = 7B * bytes_per_param
        // tok/s ≈ bandwidth / bytes_per_token_pass
        // Simplified: bw_gbs * 1e9 / (7e9 * bpw/8)
        let bytes_per_token = 7.0e9_f32 * bpw / 8.0;
        self.mem_bw_gbs * 1.0e9 / bytes_per_token
    }

    pub fn detect_cmp(&self) -> bool {
        self.fma_broken == Some(true) || self.fp16_tc_broken == Some(true)
    }
}

#[derive(Debug, Clone)]
pub struct GgufRec {
    pub format: &'static str,
    pub bpw: f32,
    pub size_7b: f32,
    pub path: GpuPath,
    pub notes: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuPath {
    Dp4a,       // INT8 dp4a MMQ — fastest on Pascal/Turing
    CublasF32,  // dequant → cuBLAS COMPUTE_32F
    Fp32Gemv,   // dequant → FP32 GEMV (slowest, memory-bound only)
}

impl GpuPath {
    pub fn label(&self) -> &'static str {
        match self {
            GpuPath::Dp4a      => "dp4a  MMQ",
            GpuPath::CublasF32 => "cuBLAS F32",
            GpuPath::Fp32Gemv  => "FP32  GEMV",
        }
    }
}
