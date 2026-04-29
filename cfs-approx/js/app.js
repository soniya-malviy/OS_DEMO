const Approx = (() => {
  // ── Kernel constants (include/linux/sched/loadavg.h) ──
  const FIXED_1 = 1 << 11;   // 2048
  const EXP_1 = 1884;      // e^(-1/12) fixed-point
  const EXP_5 = 2014;      // e^(-1/60)
  const EXP_15 = 2037;      // e^(-1/180)

  // ── Exact kernel formula (kernel/sched/loadavg.c) ─────
  function kernelEWMA(load, active, exp_n) {
    let newload = load * exp_n + active * (FIXED_1 - exp_n);
    if (active >= load) newload += FIXED_1 - 1;
    return Math.floor(newload / FIXED_1);
  }

  // ── Approximation Variant 1: Bit-shift ────────────────
  function approxBitShift(load, active, k) {
    return (load - (load >> k)) + (active >> k);
  }

  // ── Approximation Variant 2: LUT ──────────────────────
  function buildLUT(exp_n, N) {
    const lut = new Float64Array(N + 1);
    for (let i = 0; i <= N; i++) {
      // scale i back to fixed-point space before computing
      const L = Math.round(i * FIXED_1 / N);
      lut[i] = Math.floor(L * exp_n / FIXED_1);
    }
    return lut;
  }
  function approxLUT(load, active, exp_n, lut, N) {
    // scale load (fixed-point) to table index (0..N)
    const idx = Math.min(N, Math.round(load * N / FIXED_1));
    return lut[idx] + Math.floor(active * (FIXED_1 - exp_n) / FIXED_1);
  }

  // ── Approximation Variant 3: Polynomial (Horner) ──────
  function approxPoly(load, active, exp_n, m) {
    const scale = 1 << m;
    const c1 = Math.round(exp_n * scale / FIXED_1); // Q-m coefficient
    const c0 = Math.round((FIXED_1 - exp_n) * scale / FIXED_1);
    const r = Math.floor((c1 * load) >> m);
    const corr = Math.floor((c0 * active) >> m);
    return r + corr;
  }

  // ── Formal ε bounds (O1) ──────────────────────────────
  function computeEpsBounds(k, N, m, alpha, M) {
    const alphaHat_BS = 1 - 1 / (1 << k);
    const deltaAlpha_BS = Math.abs(alpha - alphaHat_BS);
    const delta_BS = 1 / (1 << m);
    const eps_BS = (deltaAlpha_BS * M + delta_BS) / (1 - alpha);

    const delta_LUT = alpha * M / (2 * N);
    const eps_LUT = delta_LUT / (1 - alpha);

    const alphaHat_P = Math.round(alpha * (1 << m)) / (1 << m);
    const deltaAlpha_P = Math.abs(alpha - alphaHat_P);
    const delta_POLY = 2 / (1 << m);
    const eps_POLY = (deltaAlpha_P * M + delta_POLY) / (1 - alpha);

    return { eps_BS, eps_LUT, eps_POLY, alphaHat_BS, deltaAlpha_BS, delta_BS, delta_LUT, deltaAlpha_P, delta_POLY };
  }

  // ── Main simulation ───────────────────────────────────
  function runSimulation(config) {
    const { T = 300, k = 4, lutN = 256, polyM = 15 } = config;
    const alpha = EXP_1 / FIXED_1;

    const nr_active = new Int32Array(T);
    let rng = 42;
    const rand = (lo, hi) => {
      rng = ((rng * 1664525 + 1013904223) >>> 0);
      return lo + (rng >>> 0) % (hi - lo);
    };
    for (let t = 0; t < 100; t++) nr_active[t] = rand(800, 1200);
    for (let t = 100; t < 200; t++) nr_active[t] = rand(1400, 1800);
    for (let t = 200; t < 300; t++) nr_active[t] = rand(400, 700);

    const lut = buildLUT(EXP_1, lutN);

    const kernel = new Float64Array(T);
    const bs = new Float64Array(T);
    const lutArr = new Float64Array(T);
    const poly = new Float64Array(T);

    let lk = nr_active[0], lbs = nr_active[0],
      ll = nr_active[0], lp = nr_active[0];

    for (let t = 0; t < T; t++) {
      const act = nr_active[t];
      lk = kernelEWMA(lk, act, EXP_1);
      lbs = Math.max(0, approxBitShift(lbs, act, k));
      ll = Math.max(0, approxLUT(ll, act, EXP_1, lut, lutN));
      lp = Math.max(0, approxPoly(lp, act, EXP_1, polyM));

      kernel[t] = lk / FIXED_1;
      bs[t] = lbs / FIXED_1;
      lutArr[t] = ll / FIXED_1;
      poly[t] = lp / FIXED_1;
    }

    const errBS = kernel.map((v, i) => Math.abs(v - bs[i]));
    const errLUT = kernel.map((v, i) => Math.abs(v - lutArr[i]));
    const errPoly = kernel.map((v, i) => Math.abs(v - poly[i]));
    const errPctBS = errBS.map((e, i) => kernel[i] > 0 ? e / kernel[i] * 100 : 0);
    const errPctLUT = errLUT.map((e, i) => kernel[i] > 0 ? e / kernel[i] * 100 : 0);
    const errPctPoly = errPoly.map((e, i) => kernel[i] > 0 ? e / kernel[i] * 100 : 0);

    const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
    const max = a => Math.max(...a);

    let maxLoadFP = 0;
    for (let t = 0; t < T; t++) {
      if (lk > maxLoadFP) maxLoadFP = lk;  // lk is fixed-point
    }
    const eps = computeEpsBounds(k, lutN, polyM, alpha, maxLoadFP / FIXED_1);
    return {
      T, nr_active: Array.from(nr_active).map(v => v / FIXED_1),
      kernel: Array.from(kernel),
      bs: Array.from(bs),
      lut: Array.from(lutArr),
      poly: Array.from(poly),
      errBS: Array.from(errBS),
      errLUT: Array.from(errLUT),
      errPoly: Array.from(errPoly),
      errPctBS: Array.from(errPctBS),
      errPctLUT: Array.from(errPctLUT),
      errPctPoly: Array.from(errPctPoly),
      stats: {
        avgErrBS: avg(errPctBS).toFixed(3),
        avgErrLUT: avg(errPctLUT).toFixed(3),
        avgErrPoly: avg(errPctPoly).toFixed(3),
        maxErrBS: max(errPctBS).toFixed(3),
        maxErrLUT: max(errPctLUT).toFixed(3),
        maxErrPoly: max(errPctPoly).toFixed(3),
        epsilonBS: max(errBS).toFixed(5),
        epsilonLUT: max(errLUT).toFixed(5),
        epsilonPoly: max(errPoly).toFixed(5),
      },
      bounds: eps,
      alpha,
      FIXED_1, EXP_1, k, lutN, polyM
    };
  }

  return { runSimulation, kernelEWMA, approxBitShift, FIXED_1, EXP_1 };
})();

const EEVDF = (() => {
  const LATENCY = 4e6;    // 4ms target latency (ns)
  const PERIOD = 1e8;     // 100ms period

  function calcEligibility(weight, load) {
    const lat = LATENCY / 1000;
    const per = PERIOD / 1000;
    const latency = Math.min(lat, per / Math.sqrt(weight / 1024));
    return latency;
  }

  function pickNextEEVDF(tasks, now) {
    let min = null;
    let minVe = Infinity;
    for (const t of tasks) {
      if (!t.eligible || t.start > now) continue;
      const ve = t.lag + (now - t.start) / t.eligibility;
      if (ve < minVe) { minVe = ve; min = t; }
    }
    return min;
  }

  function runEEVDFSimulation() {
    const T = 500;
    const N = 50;
    const cfsLatency = [];
    const eevdfLatency = [];
    const cfsWait = [];
    const eevdfWait = [];
    let now = 0;
    let tskId = 0;

    class Task {
      constructor() {
        this.id = tskId++;
        this.weight = 256 + Math.floor(Math.random() * 2048);
        this.period = 50 + Math.floor(Math.random() * 200);
        this.burst = 2 + Math.floor(Math.random() * 8);
        this.start = 0;
        this.wait = 0;
      }
      get eligibility() { return calcEligibility(this.weight, 1); }
    }

    const tasks = [];
    for (let i = 0; i < N; i++) tasks.push(new Task());

    const cfsQueue = [...tasks].sort((a, b) => a.weight - b.weight);
    const eevdfQueue = tasks.map(t => ({ ...t, lag: 0, start: 0, eligible: false }));

    for (now = 0; now < T; now++) {
      if (Math.random() < 0.1) { const t = new Task(); eevdfQueue.push({ ...t, lag: 0, start: now, eligible: true }); cfsQueue.push(t); }

      let waitCFS = 0, waitEEVDF = 0;
      if (cfsQueue.length > 0) {
        const t = cfsQueue[0];
        waitCFS = now - t.start;
        t.start = now + t.burst;
        if (t.start < T) cfsQueue[0] = t;
      }
      if (eevdfQueue.length > 0) {
        const eligible = eevdfQueue.filter(t => t.eligible && t.start <= now);
        if (eligible.length > 0) {
          const t = eligible.sort((a, b) => (a.lag + (now - a.start) / a.eligibility) - (b.lag + (now - b.start) / b.eligibility))[0];
          waitEEVDF = now - t.start;
          t.start = now + t.burst;
          t.lag += now - t.start + t.burst;
        }
      }

      cfsWait.push(Math.max(0, waitCFS));
      eevdfWait.push(Math.max(0, waitEEVDF));
      cfsLatency.push(cfsWait[now] || 0);
      eevdfLatency.push(eevdfWait[now] || 0);
    }

    return {
      T,
      cfsLatency,
      eevdfLatency,
      avgCFS: cfsWait.reduce((s, v) => s + v, 0) / T,
      avgEEVDF: eevdfWait.reduce((s, v) => s + v, 0) / T,
    };
  }

  return { runEEVDFSimulation };
})();

const Charts = (() => {
  let PALETTE = {
    kernel: '#2563EB',
    bs: '#EA580C',
    lut: '#059669',
    poly: '#7C3AED',
    eevdf: '#6D28D9',
    cfs: '#475569',
    active: '#64748B',
    err: '#DC2626',
    bound: '#4338CA',
    grid: 'rgba(148,163,184,0.1)',
  };

  const THEMES = {
    light: {
      kernel: '#2563EB', bs: '#EA580C', lut: '#059669', poly: '#7C3AED',
      eevdf: '#6D28D9', cfs: '#475569', active: '#64748B', err: '#DC2626',
      bound: '#4338CA', grid: 'rgba(148,163,184,0.1)', text: '#475569'
    },
    dark: {
      kernel: '#3B82F6', bs: '#F97316', lut: '#10B981', poly: '#A855F7',
      eevdf: '#8B5CF6', cfs: '#6B7280', active: '#94A3B8', err: '#EF4444',
      bound: '#7C3AED', grid: 'rgba(148,163,184,0.15)', text: '#94A3B8'
    }
  };

  function updatePalette(theme) {
    PALETTE = { ...THEMES[theme] };
  }

  const baseOpts = (yLabel) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { maxTicksLimit: 8, font: { size: 10, family: 'JetBrains Mono, monospace' }, color: PALETTE.text },
        grid: { color: PALETTE.grid },
        border: { color: 'transparent' }
      },
      y: {
        ticks: {
          font: { size: 10, family: 'JetBrains Mono, monospace' }, color: PALETTE.text,
          callback: v => v.toFixed(2)
        },
        grid: { color: PALETTE.grid },
        border: { color: 'transparent' },
        title: { display: !!yLabel, text: yLabel, color: PALETTE.text, font: { size: 10 } }
      }
    }
  });

  let instances = {};

  function destroy(id) {
    if (instances[id]) { instances[id].destroy(); delete instances[id]; }
  }

  function buildComparisonChart(data) {
    destroy('chart-compare');
    const el = document.getElementById('chart-compare');
    if (!el) return;
    const ctx = el.getContext('2d');
    const labels = Array.from({ length: data.T }, (_, i) => i);

    instances['chart-compare'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Linux kernel', data: data.kernel, borderColor: PALETTE.kernel, borderWidth: 2, pointRadius: 0, tension: 0.3, fill: false },
          { label: 'Bit-shift', data: data.bs, borderColor: PALETTE.bs, borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [5, 3], fill: false },
          { label: 'LUT', data: data.lut, borderColor: PALETTE.lut, borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [2, 2], fill: false },
          { label: 'Polynomial', data: data.poly, borderColor: PALETTE.poly, borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [8, 3], fill: false },
        ]
      },
      options: {
        ...baseOpts('load average'),
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: document.body.classList.contains('dark-mode') ? '#0C1420' : '#FFFFFF',
            borderColor: document.body.classList.contains('dark-mode') ? '#111C2C' : '#E2E8F0',
            borderWidth: 1,
            titleColor: document.body.classList.contains('dark-mode') ? '#E2EAF4' : '#0F172A',
            bodyColor: document.body.classList.contains('dark-mode') ? '#64748B' : '#475569',
            titleFont: { family: 'JetBrains Mono, monospace', size: 11 },
            bodyFont: { family: 'JetBrains Mono, monospace', size: 10 },
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` }
          }
        }
      }
    });
  }

  function buildErrorChart(data) {
    destroy('chart-error');
    const el = document.getElementById('chart-error');
    if (!el) return;
    const ctx = el.getContext('2d');
    const labels = Array.from({ length: data.T }, (_, i) => i);

    instances['chart-error'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Bit-shift |Δ|', data: data.errBS, borderColor: PALETTE.bs, borderWidth: 1.5, pointRadius: 0, fill: { target: 'origin', above: PALETTE.bs + '18' }, tension: 0.2 },
          { label: 'LUT |Δ|', data: data.errLUT, borderColor: PALETTE.lut, borderWidth: 1.5, pointRadius: 0, fill: { target: 'origin', above: PALETTE.lut + '18' }, tension: 0.2 },
          { label: 'Poly |Δ|', data: data.errPoly, borderColor: PALETTE.poly, borderWidth: 1.5, pointRadius: 0, fill: { target: 'origin', above: PALETTE.poly + '18' }, tension: 0.2 },
        ]
      },
      options: baseOpts('absolute error')
    });
  }

  function buildPctChart(data) {
    destroy('chart-pct');
    const el = document.getElementById('chart-pct');
    if (!el) return;
    const ctx = el.getContext('2d');
    const labels = Array.from({ length: data.T }, (_, i) => i);

    instances['chart-pct'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'Bit-shift %', data: data.errPctBS, borderColor: PALETTE.bs, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.2 },
          { label: 'LUT %', data: data.errPctLUT, borderColor: PALETTE.lut, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.2 },
          { label: 'Polynomial %', data: data.errPctPoly, borderColor: PALETTE.poly, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.2 },
        ]
      },
      options: {
        ...baseOpts('% error'),
        scales: { ...baseOpts().scales, y: { ...baseOpts().scales.y, ticks: { ...baseOpts().scales.y.ticks, callback: v => v.toFixed(2) + '%' } } }
      }
    });
  }

  function buildCyclesChart() {
    destroy('chart-cycles');
    const el = document.getElementById('chart-cycles');
    if (!el) return;
    const ctx = el.getContext('2d');

    instances['chart-cycles'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['IMUL', 'SAR', 'LUT', 'Poly'],
        datasets: [{
          data: [3, 1, 1, 2],
          backgroundColor: [PALETTE.kernel + 'CC', PALETTE.bs + 'CC', PALETTE.lut + 'CC', PALETTE.poly + 'CC'],
          borderColor: [PALETTE.kernel, PALETTE.bs, PALETTE.lut, PALETTE.poly],
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text }, grid: { display: false } },
          y: { min: 0, max: 4, ticks: { stepSize: 1, font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text, callback: v => v + ' cyc' }, grid: { color: PALETTE.grid } }
        }
      }
    });
  }

  function buildEpsChart(data) {
    destroy('chart-eps');
    const el = document.getElementById('chart-eps');
    if (!el) return;
    const ctx = el.getContext('2d');
    const b = data.bounds;

    instances['chart-eps'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Bit-shift', 'LUT', 'Poly'],
        datasets: [
          { label: 'ε_∞ bound', data: [b.eps_BS, b.eps_LUT, b.eps_POLY], backgroundColor: [PALETTE.bs + 'AA', PALETTE.lut + 'AA', PALETTE.poly + 'AA'], borderColor: [PALETTE.bs, PALETTE.lut, PALETTE.poly], borderWidth: 1, borderRadius: 4 },
          { label: 'Observed', data: [parseFloat(data.stats.epsilonBS), parseFloat(data.stats.epsilonLUT), parseFloat(data.stats.epsilonPoly)], backgroundColor: ['#F9731622', '#10B98122', '#A855F722'], borderColor: [PALETTE.bs, PALETTE.lut, PALETTE.poly], borderWidth: 1, borderRadius: 4, borderDash: [4, 2] }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text }, grid: { display: false } },
          y: { ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text, callback: v => v.toFixed(3) }, grid: { color: PALETTE.grid } }
        }
      }
    });
  }

  function buildEEVDFChart(data) {
    destroy('chart-eevdf-latency');
    const el = document.getElementById('chart-eevdf-latency');
    if (!el) return;
    const ctx = el.getContext('2d');
    const labels = Array.from({ length: data.T }, (_, i) => i);

    instances['chart-eevdf-latency'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'CFS', data: data.cfsLatency, borderColor: PALETTE.cfs, borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
          { label: 'EEVDF', data: data.eevdfLatency, borderColor: PALETTE.eevdf, borderWidth: 2, pointRadius: 0, fill: false, tension: 0.3 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 8, font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text }, grid: { color: PALETTE.grid }, border: { color: 'transparent' } },
          y: { ticks: { font: { size: 10, family: 'JetBrains Mono' }, color: PALETTE.text }, grid: { color: PALETTE.grid }, border: { color: 'transparent' }, title: { display: true, text: 'wait time', color: PALETTE.text } }
        }
      }
    });
  }

  function buildAll(data, eevdfData) {
    buildComparisonChart(data);
    buildErrorChart(data);
    buildPctChart(data);
    buildCyclesChart();
    buildEpsChart(data);
    if (eevdfData) buildEEVDFChart(eevdfData);
  }

  return { buildAll, updatePalette, PALETTE };
})();

// App Controller
document.addEventListener('DOMContentLoaded', () => {
  const config = { T: 300, k: 4, lutN: 256, polyM: 15 };

  function updateUI() {
    const data = Approx.runSimulation(config);
    const eevdfData = EEVDF.runEEVDFSimulation();
    Charts.buildAll(data, eevdfData);

    // Update live stats
    document.getElementById('stat-avg-bs').textContent = data.stats.avgErrBS + '%';
    document.getElementById('stat-avg-lut').textContent = data.stats.avgErrLUT + '%';
    document.getElementById('stat-avg-poly').textContent = data.stats.avgErrPoly + '%';

    document.getElementById('bound-bs').textContent = data.bounds.eps_BS.toFixed(4);
    document.getElementById('bound-lut').textContent = data.bounds.eps_LUT.toFixed(4);
    document.getElementById('bound-poly').textContent = data.bounds.eps_POLY.toFixed(4);
  }

  // Event Listeners for Sliders
  const sliders = {
    'k': 'val-k',
    'lutN': 'val-n',
    'polyM': 'val-m',
    'T': 'val-t'
  };

  Object.entries(sliders).forEach(([id, displayId]) => {
    const el = document.getElementById('input-' + id);
    if (el) {
      el.addEventListener('input', (e) => {
        config[id] = parseInt(e.target.value);
        document.getElementById(displayId).textContent = e.target.value;
        updateUI();
      });
    }
  });

  // Navigation logic
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('data-target');

      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');

      document.querySelectorAll('.section').forEach(s => s.classList.remove('visible'));
      document.getElementById('section-' + target).classList.add('visible');
    });
  });

  // Theme Toggle Logic
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      const isDark = document.body.classList.contains('dark-mode');
      themeToggle.textContent = isDark ? '☀' : '◐';
      
      Charts.updatePalette(isDark ? 'dark' : 'light');
      updateUI(); // Re-run simulation and rebuild charts
    });
  }

  updateUI();
});
