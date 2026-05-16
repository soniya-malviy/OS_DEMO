/*
 * CFS Approximate Computing — C Implementation
 * =============================================
 * Converted from JavaScript (app.js) to pure C.
 *
 * This program simulates the Linux CFS load-average calculation
 * and compares the exact kernel EWMA against three approximation
 * strategies: Bit-Shift, LUT, and Horner Polynomial.
 *
 * It also runs an EEVDF scheduling simulation comparing CFS vs EEVDF
 * wakeup latencies.
 *
 * Build:  gcc -O2 -o cfs_approx cfs_approx.c -lm
 * Run:    ./cfs_approx
 *
 * Output: Prints summary statistics to stdout and writes detailed
 *         CSV data files for external plotting.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Kernel constants (include/linux/sched/loadavg.h) ── */
#define FIXED_1   (1 << 11)    /* 2048 */
#define EXP_1     1884         /* e^(-1/12) fixed-point  */
#define EXP_5     2014         /* e^(-1/60) fixed-point  */
#define EXP_15    2037         /* e^(-1/180) fixed-point */

/* ── Simulation defaults ── */
#define DEFAULT_T      300
#define DEFAULT_K      4
#define DEFAULT_LUT_N  256
#define DEFAULT_POLY_M 15

/* ── EEVDF constants ── */
#define EEVDF_LATENCY  4000000   /* 4ms target latency (ns) */
#define EEVDF_PERIOD   100000000 /* 100ms period (ns) */
#define EEVDF_T        500
#define EEVDF_N        50

/* ══════════════════════════════════════════════════════════
 *  Exact kernel formula (kernel/sched/loadavg.c)
 * ══════════════════════════════════════════════════════════ */
static long kernel_ewma(long load, long active, int exp_n)
{
    long newload = load * exp_n + active * (FIXED_1 - exp_n);
    if (active >= load)
        newload += FIXED_1 - 1;
    return newload / FIXED_1;
}

/* ══════════════════════════════════════════════════════════
 *  Approximation Variant 1: Bit-shift
 * ══════════════════════════════════════════════════════════ */
static long approx_bitshift(long load, long active, int k)
{
    return (load - (load >> k)) + (active >> k);
}

/* ══════════════════════════════════════════════════════════
 *  Approximation Variant 2: LUT (Look-Up Table)
 * ══════════════════════════════════════════════════════════ */
static double *build_lut(int exp_n, int N)
{
    double *lut = (double *)calloc(N + 1, sizeof(double));
    if (!lut) { fprintf(stderr, "ERROR: LUT allocation failed\n"); exit(1); }
    for (int i = 0; i <= N; i++) {
        long L = (long)((double)i * FIXED_1 / N + 0.5);
        lut[i] = (double)(L * exp_n / FIXED_1);
    }
    return lut;
}

static long approx_lut(long load, long active, int exp_n,
                        const double *lut, int N)
{
    int idx = (int)((double)load * N / FIXED_1 + 0.5);
    if (idx > N) idx = N;
    return (long)(lut[idx] + (double)(active * (FIXED_1 - exp_n)) / FIXED_1);
}

/* ══════════════════════════════════════════════════════════
 *  Approximation Variant 3: Polynomial (Horner)
 * ══════════════════════════════════════════════════════════ */
static long approx_poly(long load, long active, int exp_n, int m)
{
    long scale = 1L << m;
    long c1 = (long)((double)exp_n * scale / FIXED_1 + 0.5);
    long c0 = (long)((double)(FIXED_1 - exp_n) * scale / FIXED_1 + 0.5);
    long r    = (c1 * load) >> m;
    long corr = (c0 * active) >> m;
    return r + corr;
}

/* ══════════════════════════════════════════════════════════
 *  Formal ε bounds (O1)
 * ══════════════════════════════════════════════════════════ */
typedef struct {
    double eps_BS, eps_LUT, eps_POLY;
    double alphaHat_BS, deltaAlpha_BS, delta_BS;
    double delta_LUT;
    double deltaAlpha_P, delta_POLY;
} EpsBounds;

static EpsBounds compute_eps_bounds(int k, int N, int m,
                                     double alpha, double M)
{
    EpsBounds b;
    b.alphaHat_BS   = 1.0 - 1.0 / (1 << k);
    b.deltaAlpha_BS = fabs(alpha - b.alphaHat_BS);
    b.delta_BS      = 1.0 / (1 << m);
    b.eps_BS        = (b.deltaAlpha_BS * M + b.delta_BS) / (1.0 - alpha);

    b.delta_LUT = alpha * M / (2.0 * N);
    b.eps_LUT   = b.delta_LUT / (1.0 - alpha);

    double alphaHat_P = (double)((long)(alpha * (1 << m) + 0.5)) / (1 << m);
    b.deltaAlpha_P    = fabs(alpha - alphaHat_P);
    b.delta_POLY      = 2.0 / (1 << m);
    b.eps_POLY        = (b.deltaAlpha_P * M + b.delta_POLY) / (1.0 - alpha);

    return b;
}

/* ══════════════════════════════════════════════════════════
 *  Simple LCG PRNG (matches the JS version)
 * ══════════════════════════════════════════════════════════ */
static uint32_t rng_state = 42;

static int lcg_rand(int lo, int hi)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return lo + (int)(rng_state % (unsigned)(hi - lo));
}

/* ══════════════════════════════════════════════════════════
 *  Main CFS Approximation Simulation
 * ══════════════════════════════════════════════════════════ */
typedef struct {
    int    T, k, lutN, polyM;
    double alpha;

    double *nr_active;   /* scaled by FIXED_1 */
    double *kernel;
    double *bs;
    double *lut_arr;
    double *poly;

    double *errBS, *errLUT, *errPoly;
    double *errPctBS, *errPctLUT, *errPctPoly;

    /* summary stats */
    double avgErrBS, avgErrLUT, avgErrPoly;
    double maxErrBS, maxErrLUT, maxErrPoly;
    double epsilonBS, epsilonLUT, epsilonPoly;

    EpsBounds bounds;
} SimResult;

static void free_sim(SimResult *s)
{
    free(s->nr_active); free(s->kernel); free(s->bs);
    free(s->lut_arr);   free(s->poly);
    free(s->errBS);     free(s->errLUT);    free(s->errPoly);
    free(s->errPctBS);  free(s->errPctLUT); free(s->errPctPoly);
}

#define ALLOC_ARR(n) ((double *)calloc((n), sizeof(double)))

static SimResult run_simulation(int T, int k, int lutN, int polyM)
{
    SimResult r;
    memset(&r, 0, sizeof(r));
    r.T = T; r.k = k; r.lutN = lutN; r.polyM = polyM;
    r.alpha = (double)EXP_1 / FIXED_1;

    /* Generate nr_active workload */
    int *nr_raw = (int *)calloc(T, sizeof(int));
    rng_state = 42;
    for (int t = 0; t < T && t < 100; t++)  nr_raw[t] = lcg_rand(800, 1200);
    for (int t = 100; t < T && t < 200; t++) nr_raw[t] = lcg_rand(1400, 1800);
    for (int t = 200; t < T && t < 300; t++) nr_raw[t] = lcg_rand(400, 700);
    for (int t = 300; t < T; t++)             nr_raw[t] = lcg_rand(400, 1800);

    /* Build LUT */
    double *lut = build_lut(EXP_1, lutN);

    /* Allocate result arrays */
    r.nr_active = ALLOC_ARR(T);
    r.kernel    = ALLOC_ARR(T);
    r.bs        = ALLOC_ARR(T);
    r.lut_arr   = ALLOC_ARR(T);
    r.poly      = ALLOC_ARR(T);
    r.errBS     = ALLOC_ARR(T);
    r.errLUT    = ALLOC_ARR(T);
    r.errPoly   = ALLOC_ARR(T);
    r.errPctBS  = ALLOC_ARR(T);
    r.errPctLUT = ALLOC_ARR(T);
    r.errPctPoly= ALLOC_ARR(T);

    long lk = nr_raw[0], lbs = nr_raw[0], ll = nr_raw[0], lp = nr_raw[0];

    for (int t = 0; t < T; t++) {
        long act = nr_raw[t];
        lk  = kernel_ewma(lk, act, EXP_1);
        lbs = approx_bitshift(lbs, act, k);
        if (lbs < 0) lbs = 0;
        ll  = approx_lut(ll, act, EXP_1, lut, lutN);
        if (ll < 0) ll = 0;
        lp  = approx_poly(lp, act, EXP_1, polyM);
        if (lp < 0) lp = 0;

        r.nr_active[t] = (double)nr_raw[t] / FIXED_1;
        r.kernel[t]    = (double)lk  / FIXED_1;
        r.bs[t]        = (double)lbs / FIXED_1;
        r.lut_arr[t]   = (double)ll  / FIXED_1;
        r.poly[t]      = (double)lp  / FIXED_1;
    }

    /* Compute errors */
    double sumPctBS = 0, sumPctLUT = 0, sumPctPoly = 0;
    r.maxErrBS = r.maxErrLUT = r.maxErrPoly = 0;
    r.epsilonBS = r.epsilonLUT = r.epsilonPoly = 0;

    for (int t = 0; t < T; t++) {
        r.errBS[t]   = fabs(r.kernel[t] - r.bs[t]);
        r.errLUT[t]  = fabs(r.kernel[t] - r.lut_arr[t]);
        r.errPoly[t] = fabs(r.kernel[t] - r.poly[t]);

        r.errPctBS[t]   = r.kernel[t] > 0 ? r.errBS[t]   / r.kernel[t] * 100 : 0;
        r.errPctLUT[t]  = r.kernel[t] > 0 ? r.errLUT[t]  / r.kernel[t] * 100 : 0;
        r.errPctPoly[t] = r.kernel[t] > 0 ? r.errPoly[t] / r.kernel[t] * 100 : 0;

        sumPctBS   += r.errPctBS[t];
        sumPctLUT  += r.errPctLUT[t];
        sumPctPoly += r.errPctPoly[t];

        if (r.errPctBS[t]   > r.maxErrBS)   r.maxErrBS   = r.errPctBS[t];
        if (r.errPctLUT[t]  > r.maxErrLUT)  r.maxErrLUT  = r.errPctLUT[t];
        if (r.errPctPoly[t] > r.maxErrPoly) r.maxErrPoly = r.errPctPoly[t];

        if (r.errBS[t]   > r.epsilonBS)   r.epsilonBS   = r.errBS[t];
        if (r.errLUT[t]  > r.epsilonLUT)  r.epsilonLUT  = r.errLUT[t];
        if (r.errPoly[t] > r.epsilonPoly) r.epsilonPoly = r.errPoly[t];
    }

    r.avgErrBS   = sumPctBS   / T;
    r.avgErrLUT  = sumPctLUT  / T;
    r.avgErrPoly = sumPctPoly / T;

    /* Compute formal bounds */
    double maxLoadFP = 0;
    long lk2 = nr_raw[0];
    for (int t = 0; t < T; t++) {
        lk2 = kernel_ewma(lk2, nr_raw[t], EXP_1);
        if ((double)lk2 > maxLoadFP) maxLoadFP = (double)lk2;
    }
    r.bounds = compute_eps_bounds(k, lutN, polyM, r.alpha, maxLoadFP / FIXED_1);

    free(lut);
    free(nr_raw);
    return r;
}

/* ══════════════════════════════════════════════════════════
 *  EEVDF Scheduling Simulation
 * ══════════════════════════════════════════════════════════ */
typedef struct {
    int    id;
    int    weight;
    int    period;
    int    burst;
    int    start;
    int    wait;
    double lag;
    int    eligible;
    double eligibility;
} EEVDFTask;

static double calc_eligibility(int weight)
{
    double lat = (double)EEVDF_LATENCY / 1000.0;
    double per = (double)EEVDF_PERIOD / 1000.0;
    double e   = per / sqrt((double)weight / 1024.0);
    return lat < e ? lat : e;
}

typedef struct {
    int     T;
    double *cfsLatency;
    double *eevdfLatency;
    double  avgCFS;
    double  avgEEVDF;
} EEVDFResult;

static EEVDFResult run_eevdf_simulation(void)
{
    EEVDFResult res;
    res.T = EEVDF_T;
    res.cfsLatency   = ALLOC_ARR(EEVDF_T);
    res.eevdfLatency = ALLOC_ARR(EEVDF_T);

    int taskCount = EEVDF_N;
    int maxTasks  = EEVDF_N + EEVDF_T; /* room for dynamic arrivals */

    EEVDFTask *cfsTasks   = (EEVDFTask *)calloc(maxTasks, sizeof(EEVDFTask));
    EEVDFTask *eevdfTasks = (EEVDFTask *)calloc(maxTasks, sizeof(EEVDFTask));

    srand(12345);
    for (int i = 0; i < EEVDF_N; i++) {
        cfsTasks[i].id      = i;
        cfsTasks[i].weight  = 256 + rand() % 2048;
        cfsTasks[i].period  = 50 + rand() % 200;
        cfsTasks[i].burst   = 2 + rand() % 8;
        cfsTasks[i].start   = 0;
        cfsTasks[i].eligibility = calc_eligibility(cfsTasks[i].weight);

        eevdfTasks[i] = cfsTasks[i];
        eevdfTasks[i].lag      = 0;
        eevdfTasks[i].eligible = 1;
    }

    double sumCFS = 0, sumEEVDF = 0;
    int cfsCount = EEVDF_N, eevdfCount = EEVDF_N;

    for (int now = 0; now < EEVDF_T; now++) {
        /* Random task arrival */
        if ((rand() % 100) < 10 && cfsCount < maxTasks) {
            int id = cfsCount;
            cfsTasks[id].id     = id;
            cfsTasks[id].weight = 256 + rand() % 2048;
            cfsTasks[id].period = 50 + rand() % 200;
            cfsTasks[id].burst  = 2 + rand() % 8;
            cfsTasks[id].start  = now;
            cfsTasks[id].eligibility = calc_eligibility(cfsTasks[id].weight);
            cfsCount++;

            eevdfTasks[eevdfCount] = cfsTasks[id];
            eevdfTasks[eevdfCount].lag      = 0;
            eevdfTasks[eevdfCount].eligible = 1;
            eevdfCount++;
        }

        /* CFS: pick first (sorted by weight) */
        double waitCFS = 0;
        if (cfsCount > 0) {
            waitCFS = (double)(now - cfsTasks[0].start);
            cfsTasks[0].start = now + cfsTasks[0].burst;
        }

        /* EEVDF: pick by minimum virtual eligibility */
        double waitEEVDF = 0;
        int bestIdx = -1;
        double bestVe = 1e18;
        for (int i = 0; i < eevdfCount; i++) {
            if (!eevdfTasks[i].eligible || eevdfTasks[i].start > now)
                continue;
            double ve = eevdfTasks[i].lag +
                        (double)(now - eevdfTasks[i].start) /
                        eevdfTasks[i].eligibility;
            if (ve < bestVe) { bestVe = ve; bestIdx = i; }
        }
        if (bestIdx >= 0) {
            waitEEVDF = (double)(now - eevdfTasks[bestIdx].start);
            eevdfTasks[bestIdx].start = now + eevdfTasks[bestIdx].burst;
            eevdfTasks[bestIdx].lag  += (double)(now - eevdfTasks[bestIdx].start
                                                 + eevdfTasks[bestIdx].burst);
        }

        res.cfsLatency[now]   = waitCFS > 0 ? waitCFS : 0;
        res.eevdfLatency[now] = waitEEVDF > 0 ? waitEEVDF : 0;
        sumCFS   += res.cfsLatency[now];
        sumEEVDF += res.eevdfLatency[now];
    }

    res.avgCFS   = sumCFS   / EEVDF_T;
    res.avgEEVDF = sumEEVDF / EEVDF_T;

    free(cfsTasks);
    free(eevdfTasks);
    return res;
}

/* ══════════════════════════════════════════════════════════
 *  CSV Output
 * ══════════════════════════════════════════════════════════ */
static void write_approx_csv(const SimResult *s, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return; }

    fprintf(f, "tick,nr_active,kernel,bitshift,lut,poly,"
               "errBS,errLUT,errPoly,errPctBS,errPctLUT,errPctPoly\n");
    for (int t = 0; t < s->T; t++) {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                t, s->nr_active[t], s->kernel[t], s->bs[t],
                s->lut_arr[t], s->poly[t],
                s->errBS[t], s->errLUT[t], s->errPoly[t],
                s->errPctBS[t], s->errPctLUT[t], s->errPctPoly[t]);
    }
    fclose(f);
}

static void write_eevdf_csv(const EEVDFResult *e, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return; }

    fprintf(f, "tick,cfs_latency,eevdf_latency\n");
    for (int t = 0; t < e->T; t++) {
        fprintf(f, "%d,%.4f,%.4f\n", t, e->cfsLatency[t], e->eevdfLatency[t]);
    }
    fclose(f);
}

/* ══════════════════════════════════════════════════════════
 *  Pretty-print helpers
 * ══════════════════════════════════════════════════════════ */
#define SEP "─────────────────────────────────────────────────────"
#define HEADER_FMT  "  %-18s  %12s  %12s  %12s\n"
#define ROW_FMT     "  %-18s  %12.5f  %12.5f  %12.5f\n"

static void print_banner(void)
{
    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════╗\n");
    printf("  ║   CFS APPROXIMATE COMPUTING — C SIMULATION       ║\n");
    printf("  ║   kernel/sched/loadavg.c                         ║\n");
    printf("  ╚═══════════════════════════════════════════════════╝\n\n");
}

/* ══════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    int T     = DEFAULT_T;
    int k     = DEFAULT_K;
    int lutN  = DEFAULT_LUT_N;
    int polyM = DEFAULT_POLY_M;

    /* Optional CLI overrides: ./cfs_approx [T] [k] [lutN] [polyM] */
    if (argc >= 2) T     = atoi(argv[1]);
    if (argc >= 3) k     = atoi(argv[2]);
    if (argc >= 4) lutN  = atoi(argv[3]);
    if (argc >= 5) polyM = atoi(argv[4]);

    print_banner();

    printf("  Parameters: T=%d  k=%d  lutN=%d  polyM=%d\n", T, k, lutN, polyM);
    printf("  FIXED_1=%d  EXP_1=%d  α=%.6f\n\n", FIXED_1, EXP_1,
           (double)EXP_1 / FIXED_1);

    /* ── Run CFS approximation simulation ── */
    SimResult sim = run_simulation(T, k, lutN, polyM);

    printf("  %s\n", SEP);
    printf("  APPROXIMATION ERROR SUMMARY\n");
    printf("  %s\n", SEP);
    printf(HEADER_FMT, "Metric", "Bit-Shift", "LUT", "Polynomial");
    printf("  %s\n", SEP);
    printf(ROW_FMT, "Avg Error (%)", sim.avgErrBS, sim.avgErrLUT, sim.avgErrPoly);
    printf(ROW_FMT, "Max Error (%)", sim.maxErrBS, sim.maxErrLUT, sim.maxErrPoly);
    printf(ROW_FMT, "ε observed",    sim.epsilonBS, sim.epsilonLUT, sim.epsilonPoly);
    printf("  %s\n\n", SEP);

    printf("  %s\n", SEP);
    printf("  FORMAL ε BOUNDS (O1)\n");
    printf("  %s\n", SEP);
    printf(HEADER_FMT, "Bound", "Bit-Shift", "LUT", "Polynomial");
    printf("  %s\n", SEP);
    printf(ROW_FMT, "ε_∞ proven",  sim.bounds.eps_BS,  sim.bounds.eps_LUT,  sim.bounds.eps_POLY);
    printf(ROW_FMT, "ε_∞ observed", sim.epsilonBS,      sim.epsilonLUT,      sim.epsilonPoly);
    printf(ROW_FMT, "|Δα|",         sim.bounds.deltaAlpha_BS, 0.0, sim.bounds.deltaAlpha_P);
    printf(ROW_FMT, "δ (quant)",     sim.bounds.delta_BS, sim.bounds.delta_LUT, sim.bounds.delta_POLY);
    printf("  %s\n\n", SEP);

    /* ── Run EEVDF simulation ── */
    EEVDFResult eevdf = run_eevdf_simulation();

    printf("  %s\n", SEP);
    printf("  EEVDF vs CFS SCHEDULING\n");
    printf("  %s\n", SEP);
    printf("  Avg CFS   wait time:  %.4f\n", eevdf.avgCFS);
    printf("  Avg EEVDF wait time:  %.4f\n", eevdf.avgEEVDF);
    printf("  %s\n\n", SEP);

    /* ── Write CSV files ── */
    write_approx_csv(&sim, "approx_results.csv");
    write_eevdf_csv(&eevdf, "eevdf_results.csv");

    printf("  ✓ Written: approx_results.csv  (%d rows)\n", sim.T);
    printf("  ✓ Written: eevdf_results.csv   (%d rows)\n", eevdf.T);
    printf("\n  CPU Cycle Reference:\n");
    printf("    IMUL (kernel):  3 cycles\n");
    printf("    SAR  (bitshift): 1 cycle\n");
    printf("    MOV  (LUT):     1 cycle\n");
    printf("    Poly (Horner):  2 cycles\n\n");

    /* Cleanup */
    free_sim(&sim);
    free(eevdf.cfsLatency);
    free(eevdf.eevdfLatency);

    return 0;
}
