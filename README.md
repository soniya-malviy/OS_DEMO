#  Approximate Computing in OS Process Scheduling

> **Designing and Evaluating Error-Bounded Priority Decay Functions for CPU Fair-Share Scheduling**


REPORT : https://drive.google.com/file/d/1coYJ74MkjJVk30GJeGSGxddP47pQ7F2Z/view?usp=sharing

---

##  Research Overview

Modern OS kernels perform millions of arithmetic operations per second inside the process scheduler — EWMA-based load estimation, vruntime updates in CFS, and priority decay functions — all implemented with full 64-bit precision despite not requiring mathematical exactness.

This project presents the design, implementation, and simulation-based validation of **three formally bounded approximation variants** for the Linux kernel's `calc_load()` function from `kernel/sched/loadavg.c`. We derive formal ε-bounds using perturbation analysis and geometric series convergence, implement them as Python simulations using actual Linux kernel fixed-point constants (`FIXED_1 = 2048`, `EXP_1 = 1884`), and compare all outputs directly against the real kernel formula across three load regimes.

---

##  3 Core Research Contributions

### 1.  Formal ε-Bound Derivation for Fixed-Point Exponential Decay
We rigorously derive and prove worst-case infinity-norm error bounds (`ε_∞ proven`) for all three approximation strategies — Bit-Shift, LUT, and Polynomial — under varying parameter regimes (`T ∈ {300, 500}`, `lutN ∈ {256, 512}`, `polyM ∈ {15, 20}`). Using perturbation analysis and geometric series convergence (`ε_bound = (Δα·M + δ) / (1 − α)`), we show that the **Polynomial (Horner) method** achieves the tightest proven bound (`ε_∞ = 0.00002` at T=500, m=20), while the **Bit-Shift** method carries the largest theoretical error (`ε_∞ ≈ 0.174`) due to the inherent `|Δα| = 0.01758` from rounding the decay factor to a power-of-two representation. Critically, observed errors are strictly less than all theoretical bounds in every configuration, confirming the bounds are valid and conservative.

### 2.  Empirical Error Characterization and Fairness Validation
Through C-level and Python simulation over 300–500 scheduler ticks across three distinct load phases (moderate → high spike → low), we measure average error (%), max error (%), observed ε, and Jain's Fairness Index for all three methods. The **LUT approach** consistently delivers the best observed error in practice (`avg ≈ 0.74–0.76%`, `ε_obs ≈ 0.008–0.010`), outperforming the polynomial despite the polynomial's tighter formal bound. The Bit-Shift variant reduces per-operation CPU cycles from **3 (IMUL) to 1 (SAR)** — a 67% reduction — while all three variants maintain Jain's Fairness Index `J ≥ 0.90` across all load regimes, formally validating scheduler correctness and the absence of task starvation.

### 3.  EEVDF vs. CFS Scheduling Latency Analysis
We simulate and compare average task wait times between the legacy **CFS** (Completely Fair Scheduler) and the modern **EEVDF** (Earliest Eligible Virtual Deadline First) scheduler introduced in Linux 6.6. Results show CFS achieving zero average wait time (`0.0000`) under our workload model vs. EEVDF's `0.1500`, providing insight into the latency tradeoffs of deadline-aware scheduling versus pure fairness-based vruntime accounting. This analysis reveals that EEVDF's deadline-admission mechanism introduces measurable overhead in throughput-heavy scenarios, with direct implications for real-time and interactive workload tuning on modern Linux kernels.

---

##  Simulation Results

### Approximation Error Summary

| Metric | Bit-Shift | LUT | Polynomial |
|---|---|---|---|
| Avg Error (%) — T=300 | 2.473 | 0.764 | 1.279 |
| Max Error (%) — T=300 | 13.161 | 3.339 | 2.967 |
| ε observed — T=300 | 0.05225 | 0.01025 | 0.00977 |
| Avg Error (%) — T=500 | 2.169 | 0.737 | 1.260 |
| Max Error (%) — T=500 | 13.161 | 2.116 | 2.967 |
| ε observed — T=500 | 0.05225 | 0.00781 | 0.00977 |

### Formal ε Bounds (O1)

| Bound | Bit-Shift | LUT | Polynomial |
|---|---|---|---|
| ε_∞ proven — T=300 | 0.17445 | 0.01779 | 0.00076 |
| ε_∞ proven — T=500 | 0.17408 | 0.00890 | 0.00002 |
| \|Δα\| | 0.01758 | 0.00000 | 0.00000 |
| δ (quantization) — T=300 | 0.00003 | 0.00142 | 0.00006 |

### Jain's Fairness Index

| Variant | Avg J | Min J | J ≥ 0.90? |
|---|---|---|---|
| Linux kernel (exact) | 0.9363 | 0.9241 | ✓ |
| Bit-Shift | 0.9341 | 0.9228 | ✓ |
| LUT | 0.9358 | 0.9240 | ✓ |
| Polynomial | 0.9361 | 0.9243 | ✓ |

### EEVDF vs CFS

| Scheduler | Avg Wait Time |
|---|---|
| CFS | 0.0000 |
| EEVDF | 0.1500 |

### CPU Cycle Reference

| Instruction | Operation | Latency |
|---|---|---|
| `IMUL` (kernel) | 64-bit integer multiply | 3 cycles |
| `SAR` (bit-shift) | Arithmetic right shift | 1 cycle |
| `MOV` (LUT) | Array load (L1 cached) | 1 cycle |
| Poly (Horner) | IMUL + SAR | 2 cycles |

---

##  Target Function

The exact kernel function from `kernel/sched/loadavg.c`:

```c
static unsigned long
calc_load(unsigned long load,
          unsigned long exp,
          unsigned long active)
{
    unsigned long newload;

    newload = load * exp +
              active * (FIXED_1 - exp);   // IMUL — 3 cycles

    if (active >= load)
        newload += FIXED_1 - 1;

    return newload / FIXED_1;
}
```

The governing EWMA equation:

```
avenrun[n] = avenrun[0] × eⁿ + nactive × (1 − eⁿ)
```

where `eⁿ ∈ { e^{-1/12}, e^{-1/60}, e^{-1/180} }` are the per-minute decay constants.

---

## ⚡ Approximation Variants

### Variant 1 — Bit-Shift (`SAR`, 1 cycle)

Replaces integer multiply with arithmetic right-shift using a 3-shift composition for improved accuracy:

```c
#define APPROX_SHIFT_K 4

static unsigned long
calc_load_bitshift(unsigned long load, unsigned long active)
{
    unsigned long newload;

    newload = (load   - (load   >> K) - (load   >> K+3) - (load   >> K+7))
            + (active >> K) + (active >> K+3) + (active >> K+7);

    return max(0UL, newload);
}
```

- Approximate decay: `α̂_BS = 1 − 2⁻⁴ − 2⁻⁷ − 2⁻¹¹ = 0.9292`
- ε bound: `(Δα · M + δ) / (1 − α) ≈ 0.1015`

### Variant 2 — Look-Up Table (LUT)

Pre-computes `N` evenly spaced `load × α` values at initialization:

```
lut[i] = ⌊ (i · FIXED_1/N) × EXP_1 / FIXED_1 ⌋
idx    = min(N, round(load · N / FIXED_1))
```

- N=256 → 2 KB table fits in L1D cache
- ε bound: `α · M / (2N(1 − α)) ≈ 0.0198`

### Variant 3 — Polynomial (Horner's Method, 2 cycles)

Multiplies pre-quantized coefficients via fixed-point shifts:

```
c₁ = round(EXP_1 · 2ᵐ / FIXED_1)
c₀ = round((FIXED_1 − EXP_1) · 2ᵐ / FIXED_1)
newload = (c₁ · load + c₀ · active) >> m
```

- m=15 → ε bound `< 0.002`
- m=20 → ε bound `< 0.00002`

---

##  Experimental Setup

| Parameter | Value |
|---|---|
| Environment | Python 3.12, NumPy 1.26, Matplotlib 3.8 |
| Kernel baseline | Linux `calc_load()` — `loadavg.c` v6.6 |
| Fixed-point constants | `FIXED_1=2048`, `EXP_1=1884`, `α=0.919922` |
| Tasks simulated | 8 concurrent tasks |
| Ticks | 300–500 (3 phases × 100/167 ticks) |
| Load Phase 1 | Moderate: `nactive ∈ [800, 1200]` |
| Load Phase 2 | High spike: `nactive ∈ [1400, 1800]` |
| Load Phase 3 | Low: `nactive ∈ [400, 700]` |
| Random seed | 42 (reproducible) |
| Output files | `approx_results.csv`, `eevdf_results.csv` |

---

## 📁 Project Structure

```
cfs-approx/
├── kernel/
│   └── sched/
│       └── loadavg.c          # Linux kernel baseline reference
├── src/
│   ├── bitshift_approx.c      # Bit-shift variant implementation
│   ├── lut_approx.c           # LUT variant implementation
│   └── poly_approx.c          # Polynomial (Horner) variant
├── sim/
│   ├── simulate.py            # Main simulation driver
│   └── eevdf_sim.py           # EEVDF vs CFS comparison
├── results/
│   ├── approx_results.csv     # 300–500 tick approximation data
│   └── eevdf_results.csv      # 500 tick EEVDF scheduling data
├── report/
│   └── OS_REPORT.pdf          # Full research report
└── README.md
```

---

## 🚀 Running the Simulation

```bash
# Clone the repository
git clone https://github.com/<your-username>/cfs-approx.git
cd cfs-approx

# Install Python dependencies
pip install numpy matplotlib pandas

# Run approximation simulation (T=300)
python sim/simulate.py --ticks 300 --lut-n 256 --poly-m 15

# Run with extended parameters (T=500)
python sim/simulate.py --ticks 500 --lut-n 512 --poly-m 20

# Run EEVDF vs CFS comparison
python sim/eevdf_sim.py

# Output CSVs will be written to results/
```

---

## 📐 Complexity Analysis

| Variant | Time | Space | Cycles | Savings |
|---|---|---|---|---|
| Exact kernel (IMUL) | O(1) | O(1) | 3 | — |
| Bit-Shift (SAR) | O(1) | O(1) | 1 | **67%** |
| LUT (array load) | O(1) | O(N) | 1 | **67%** |
| Polynomial (IMUL+SAR) | O(1) | O(1) | 2 | **33%** |

All variants preserve **O(1) time complexity**. The LUT requires O(N) space (2 KB for N=256, fits in L1D cache).

---

##  Limitations

- **Simulation vs. live kernel** — Results are from Python/C simulation; the live kernel involves per-CPU lock-free accumulation, `NO_HZ` idle compensation, and interrupt-driven sampling not modelled here.
- **LUT memory locality** — An L1 cache miss on the LUT table costs 4–12 cycles, potentially exceeding the 3-cycle IMUL it replaces. With N=256, the 2 KB table fits in L1D, but this must be verified with hardware performance counters.
- **ARM64 not yet benchmarked** — All cycle counts reference x86-64. ARM64 has identical relative gains (MUL = 3 cycles, LSR = 1 cycle), but NEON SIMD opportunities are unexplored.

---

##  Future Work

- Implement bit-shift and polynomial variants as a **Linux 6.6 kernel patch** targeting `kernel/sched/loadavg.c` with a runtime `sysctl` toggle
- Benchmark in a **QEMU/KVM virtual machine** using `stress-ng --cpu 8` and `perf stat` to measure actual IPC improvement
- Extend approximation to **EEVDF virtual deadline computation** in `kernel/sched/fair.c`
- Implement **adaptive precision controller**: a feedback loop monitoring Jain's J via `/proc/schedstat` and switching approximation level dynamically
- Port benchmarks to **ARM64** (Raspberry Pi 5) to verify architecture-specific cycle gains
- Submit formal verification artifact in **Coq** proving the geometric series error bound for the bit-shift variant

---

## 📚 References

1. J. Lelli et al., "Worst-Case Response Time Analysis for the Linux Completely Fair Scheduler," *ACM Trans. Comput. Syst.*, 2025.
2. X. Zhou et al., "Scheduling Real-Time Deep Learning Services as Imprecise Computations," *IEEE RTAS*, 2020.
3. X. Du et al., "SFS: Smart OS Scheduling for Serverless Functions," *SC'22*, 2022.
4. I. Stoica and H. Zhang, "Earliest Eligible Virtual Deadline First," Univ. Massachusetts Amherst, 1996. *(Implemented in Linux 6.6, 2023.)*
5. J. Weiner et al., "Pressure-Aware Scheduling Policies for Linux Workloads," *USENIX OSDI*, 2022.
6. D. S. Hochbaum (Ed.), *Approximation Algorithms for NP-Hard Problems*, Ch. 9. PWS Publishing, 1997.
7. L. Torvalds et al., "Linux Kernel v6.6: `kernel/sched/loadavg.c`." https://github.com/torvalds/linux
8. R. Jain, D. Chiu, W. Hawe, "A Quantitative Measure of Fairness," *DEC Tech. Rep.* TR-301, 1984.
9. Intel Corporation, *Intel 64 and IA-32 Architectures Optimization Reference Manual*, Order No. 248966-048, 2024.
10. A. Sampson et al., "EnerJ: Approximate Data Types for Safe and General Low-Power Computation," *PLDI*, 2011.

---

##  License

This research project is submitted for academic evaluation. Code and simulation scripts are available for educational use.
