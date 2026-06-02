# nbenchmark v2.0 -see details of you retro card! )

<img width="527" height="594" alt="image" src="https://github.com/user-attachments/assets/149af52e-4de3-42ac-b860-53823e31dbc8" />

90 Watt vs 125 Watt P104-100 8GB: +10% TFLOPS in specific path:
<img width="1789" height="914" alt="image" src="https://github.com/user-attachments/assets/209ed631-c8d6-4a80-ae40-0b2b1d413324" />

**Full Pascal + Turing GPU benchmark** — PTX kernels, cuBLAS probe, GGUF format advisor.

Reveals broken FMA / Tensor Cores on NVIDIA CMP mining cards.  
Maps all GGUF quantization formats to actual GPU instruction paths.

## Target Hardware

### Pascal (SM 6.x) — no Tensor Cores
| Card | VRAM | BW | FP32 | INT8 dp4a | Notes |
|------|------|----|------|-----------|-------|
| P106-100 | 6GB 192-bit | 192 GB/s | 4.6 TF | ~18 TOPS | GTX 1060 equivalent |
| P104-100 | 8GB 256-bit | 256 GB/s | 7.3 TF | ~29 TOPS | |
| **P102-100** | **10GB 320-bit** | **320 GB/s** | **5.6 TF*** | **41 TOPS*** | **tested** |
| Tesla P100 | 16GB HBM2 | 732 GB/s | 9.3 TF | ~37 TOPS | full FP64! |
| GTX 1080 Ti | 11GB 352-bit | 484 GB/s | 11.3 TF | ~45 TOPS | |

*measured values

### Turing (SM 7.5)
| Card | VRAM | BW | FP32 | FP16 TC | INT8 TC | Notes |
|------|------|----|------|---------|---------|-------|
| **CMP 50HX** | 10GB | 320 GB/s | ~5.9 TF* | **BAD!*** | **1.7 TOPS*** | **BROKEN FMA+TC** |
| RTX 2060 6GB | 6GB 192-bit | 336 GB/s | 6.5 TF | 52 TF | 104 TOPS | full TC |
| **RTX 2060 SUPER** | **8GB 256-bit** | **448 GB/s** | **7.2 TF** | **57 TF** | **115 TOPS** | best BW |
| **RTX 2060 12GB** | **12GB 192-bit** | **360 GB/s** | **7.3 TF** | **58 TF** | **116 TOPS** | most VRAM |
| **Quadro T400** | 2/4GB | **80 GB/s** | 1.1 TF | N/A | N/A | **NO TC (TU117!)** |
| RTX 2070 | 8GB 256-bit | 448 GB/s | 7.5 TF | 58 TF | 115 TOPS | |
| RTX 2080 Ti | 11GB 352-bit | 616 GB/s | 13.4 TF | 108 TF | 216 TOPS | best Turing |

## Key Differences Between Cards

### RTX 2060 6GB vs SUPER vs 12GB
```
              6GB      SUPER    12GB
CUDA cores:  1920     2176     2176
Bus:         192-bit  256-bit  192-bit   ← 12GB same bus as 6GB!
BW:          336 GB/s 448 GB/s 360 GB/s  ← SUPER fastest
VRAM:        6 GB     8 GB     12 GB     ← 12GB most memory
FP16 TC:     52 TF    57 TF    58 TF
INT8 TC:     104 TOPS 115 TOPS 116 TOPS

Conclusion: SUPER is faster than 12GB for throughput.
            12GB is better when model size matters (13B+ at Q4_K_M).
```

### Quadro T400 — Special Case
- **TU117 die: NO Tensor Cores** despite SM 7.5!
- Only 80 GB/s bandwidth (64-bit bus) — extremely limited
- Suitable only for 1-3B models
- dp4a (INT8) works, but throughput is low (~4 TOPS)
- Best GGUF: Q4_K_M for 1.5-3B, or Q8_0 for <1B

## Build

```bash
# Requirements: CUDA 12.x toolkit, Rust 1.75+
# Ubuntu:
sudo apt install libcublas-dev cuda-toolkit-12

cargo build --release
```

## Usage

```bash
# Quick diagnostic
./nbenchmark cmp-detect

# Full report with GGUF recommendations
./nbenchmark full-report

# Individual tests
./nbenchmark fp32
./nbenchmark fp32-fma          # reveals CMP breakage
./nbenchmark f16               # f16x2 mul+add (best on CMP 50HX!)
./nbenchmark int8              # all dp4a variants
./nbenchmark int4              # Q4_K/Q4_0 path simulation
./nbenchmark bitwise           # 1-bit binary nets
./nbenchmark int32             # accumulator baseline

# cuBLAS probe (original nbenchmark core test)
./nbenchmark cublas-probe
./nbenchmark cublas-probe --m 4096 --gemm-iters 50

# Turing only
./nbenchmark wmma-probe        # direct PTX Tensor Core test

# Info tables
./nbenchmark gguf-table        # all GGUF formats → GPU paths
./nbenchmark gpu-table         # known GPU specs

# With specific device
./nbenchmark full-report --device 1
./nbenchmark cublas-probe --n 134217728 --reps 16384
Full benchmark for NVIDIA Pascal (SM 6.x) and Turing (SM 7.5) GPUs.
Reveals broken FMA / Tensor Cores on NVIDIA CMP mining cards.
Maps all GGUF quantization formats to GPU instruction paths.

TARGET HARDWARE:
Pascal: P106-100 (6GB), P104-100 (8GB), P102-100 (10GB), Tesla P100 (16GB)
Turing: CMP 50HX, RTX 2060 (6GB), RTX 2060 SUPER (8GB),
RTX 2060 12GB, Quadro T400, RTX 2070, RTX 2080 Ti

KNOWN RESULTS:
P102-100 (Pascal):  FP32 mul+add ~5.6 TF | FP32 FMA ~10.5 TF | INT8 ~41 TOPS
CMP 50HX (Turing):  FP32 mul+add ~5.9 TF | FP32 FMA ~0.42 TF (BROKEN!) | INT8 ~1.7 TOPS
RTX 2060 (Turing):  FP32 ~6.5 TF | FP16 TC ~52 TF | INT8 TC ~104 TOPS
RTX 2060 SUPER:     FP32 ~7.2 TF | FP16 TC ~57 TF | BW 448 GB/s
RTX 2060 12GB:      FP32 ~7.3 TF | FP16 TC ~58 TF | BW 360 GB/s | VRAM 12GB
Quadro T400:        FP32 ~1.1 TF | NO TC (TU117!) | BW 80 GB/s | VRAM 4GB

GGUF GPU PATHS:
dp4a MMQ:    Q8_0, Q4_0, Q4_K_M, IQ4_XS, IQ4_NL     → fastest on Pascal!
cuBLAS F32:  Q5_K_M, Q6_K, Q3_K, IQ2-3 series        → 7x slower than dp4a
FP32 GEMV:   TQ1_0, TQ2_0, IQ1_S/M                   → avoid on GPU
FORBIDDEN:   BF16, TF32, FP8 (not available SM<8.0)


Usage: nbenchmark [OPTIONS] <COMMAND>

Commands:
  fp32          FP32 explicit mul+add (no FMA) — safe baseline
  fp32-fma      FP32 FMA — reveals CMP breakage if 14x slower than fp32
  fp64          FP64 mul+add — diagnose hardware FP64 (GP100 vs consumer)
  f16           FP16 f16x2 explicit mul+add — best path on CMP 50HX
  f16-fma       FP16 f16x2 FMA
  int8          INT8 dp4a all variants: s8s8 / u8u8 / s8u8 (Q8_0/Q8_1 paths)
  int4          INT4 emulated via dp4a (Q4_K/Q4_0 GPU path)
  int32         INT32 MAD — accumulator throughput baseline
  bitwise       Bitwise XNOR + POPC — 1-bit / binary neural nets
  all-ptx       All PTX: fp32/fp64/fp16/int8/int4/bitwise/int32
  fma-compare   FP32 vs FMA comparison (CMP diagnostic)
  cublas-probe  KEY: cuBLAS GEMM probe — all compute modes, correctness check
  wmma-probe    WMMA direct Tensor Core test (SM 7.5+ only, bypasses cuBLAS)
  cmp-detect    Quick CMP auto-detect: broken FMA + TC?
  gguf-table    Print GGUF format → GPU path mapping table
  gpu-table     Print known GPU specs comparison table
  full-report   Everything: all PTX + FMA compare + cuBLAS + WMMA + recommendations
  help          Print this message or the help of the given subcommand(s)

Options:
      --n <N>
          Number of elements for PTX benchmarks
          
          [default: 67108864]

      --reps <REPS>
          Arithmetic repetitions per GPU thread
          
          [default: 16384]

      --m <M>
          GEMM matrix dimension M=N=K for cuBLAS probe
          
          [default: 2048]

      --gemm-iters <GEMM_ITERS>
          GEMM iterations for stable timing
          
          [default: 50]

      --device <DEVICE>
          CUDA device index (0 = first GPU)
          
          [default: 0]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
  ```

## GGUF Format → GPU Path

| Category | Formats | GPU Path | Pascal Speed |
|----------|---------|----------|-------------|
| **dp4a MMQ** ★★★★★ | Q8_0, Q4_0, Q4_K_M, IQ4_XS, IQ4_NL | `dp4a.s32.s32` | ~41 TOPS |
| **cuBLAS F32** ★★ | Q5_K, Q6_K, Q3_K, IQ2/IQ3 | `COMPUTE_32F` | ~6.9 TFLOPS |
| **FP32 GEMV** ★ | TQ1_0, TQ2_0, IQ1_S/M | dequant+FP32 | slow |
| **❌ N/A** | BF16, TF32, FP8 | not supported | — |

**Key insight**: dp4a MMQ is **7x faster** than cuBLAS FP32 on Pascal!  
Q4_K_M is faster than Q5_K_M despite having fewer bits, because different GPU path.

## Pareto-Optimal GGUF Formats

```
By speed+quality:  Q8_0 > IQ4_XS* > Q4_K_M > IQ3_M > IQ2_M
By VRAM+quality:   IQ4_XS* > Q4_K_M > IQ3_M > UD-Q2_K_XL
(* requires imatrix for best results)

AVOID: BF16, Q4_1, Q5_1, IQ1_S, TQ1_0, cuBLAS COMPUTE_16F on Pascal
```

## CMP 50HX Recommended Settings

```bash
# Broken FMA: use pedantic math
GGML_CUDA_NO_FMA=1 ./llama-server -m model.Q4_K_M.gguf -ngl 99

# Alternative: exploit f16x2 mul+add (22 TFLOPS on CMP 50HX!)
# For small models in F16 format:
./llama-server -m model.F16.gguf -ngl 99

# cuBLAS must avoid FMA:
# cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)
# cublasGemmEx(..., CUBLAS_COMPUTE_32F, ...)
```

## Credits

- Original nbenchmark by korziner
- Based on habr.com article by Олег@WebSlave (Нижегородская обл.)
- TurboQuant research: Zandieh et al., ICLR 2026
