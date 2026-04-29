# Approximate Computing for OS Kernel Internals / Approximate Computing in OS Process Scheduling: Designing and Evaluating Error-Bounded Priority Decay Functions for CPU Fair-Share Scheduling

## REPORT 

https://drive.google.com/file/d/16bJhK0Cwaa1ny4N1c16WatTz2cVOhVfo/view?usp=sharing

## Research Overview

This project explores replacing the standard 3-cycle integer multiplication (`IMUL`) in the Linux kernel's load-average calculation (`calc_load`) with 1-cycle approximations. The goal is to reduce scheduler overhead on high-core-count systems while maintaining formal error bounds.

### Key Optimizations
- **Bit-Shift Approximation**: Replaces EWMA multiplication with arithmetic right shifts (SAR).
- **Look-Up Tables (LUT)**: Pre-computed quantization for $O(1)$ performance.
- **Horner Polynomials**: Tunable precision using fixed-point polynomial evaluation.

## Features

- **Interactive Simulation**: Live load-trace generation with real-time comparison between kernel logic and proposed approximations.
- **Formal ε Bounds**: Visual verification of proven upper error bounds (O1 complexity).
- **Scheduling Class Comparison**: Comparative latency analysis between standard CFS and EEVDF (Earliest Eligible Virtual Deadline First).
- **Instruction Latency**: Breakdown of x86 cycle costs for different implementations.
- **Dual Theme Support**: Premium dark and light modes with adaptive Chart.js visualizations.
- **Kernel-Ready C Code**: Direct C-code patches for `kernel/sched/loadavg.c`.

## Tech Stack

- **Core**: Vanilla JavaScript (ES6+), Semantic HTML5
- **Styling**: Modern CSS3 (Variables, Grid, Flexbox, Backdrop filters)
- **Charts**: [Chart.js](https://www.chartjs.org/) for high-performance data visualization
- **Typography**: JetBrains Mono & Syne (via Google Fonts)

## 📁 Project Structure

```text
cfs-approx/
├── css/
│   └── main.css     # Theme-aware styles & design system
├── js/
│   └── app.js       # Simulation logic & Chart.js controllers
├── data/            # Static research data & constants
└── index.html       # Research dashboard interface
```

## Getting Started

1. Clone the repository.
2. Open `cfs-approx/index.html` in any modern web browser.
3. Use the sidebar sliders to tune bit-shift parameters ($k$), LUT entries ($N$), and polynomial bits ($m$).
4. Toggle between Light and Dark themes using the **◐/☀** button in the header.

---
*Research codebase for kernel performance optimization.*
