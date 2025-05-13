/*
 * Tiny Ising model.
 * Loosely based on  "q-state Potts model metastability
 * study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick,
 * Sergio A. Cannas
 * http://arxiv.org/abs/1101.0876
 *
 * Debugging: Ezequiel Ferrero
 */

#if defined(AVX2_INTRINSICS) || defined(AVX512F_INTRINSICS)
    #include "ising_intrinsics.h"
#elif defined(RB_UPDATE)
    #include "ising_rb.h"
#else
    #include "ising.h"
#endif


#include "params.h"
#include "wtime.h"

#include <assert.h>
#include <limits.h> // UINT_MAX
#include <math.h> // expf()
#include <stdio.h> // printf()
#include <stdlib.h> // rand()
#include <time.h> // time()

// Internal definitions and functions
// out vector size, it is +1 since we reach TEMP_
#define NPOINTS (1 + (int)((TEMP_FINAL - TEMP_INITIAL) / TEMP_DELTA))
#define N (L * L) // system size
#define SEED (time(NULL)) // random seed

#ifndef WHOAMI
#define WHOAMI "I don't know who i am"
#endif

// temperature, E, E^2, E^4, M, M^2, M^4
struct statpoint {
    double t;
    double e;
    double e2;
    double e4;
    double m;
    double m2;
    double m4;
};

    // clear the grid
#if defined(ZERO_PADDING)
int grid[L+2][L+2] = { { 0 } };
#else
int grid[L][L] = { { 0 } };
#endif

#if defined(ZERO_PADDING)
static void cycle(int grid[L+2][L+2],
                  const double min, const double max,
                  const double step, const unsigned int calc_step,
                  struct statpoint stats[])

#else

static void cycle(int grid[L][L],
                  const double min, const double max,
                  const double step, const unsigned int calc_step,
                  struct statpoint stats[])
#endif
{

    assert((0 < step && min <= max) || (step < 0 && max <= min));
    int modifier = (0 < step) ? 1 : -1;

    unsigned int index = 0;
    for (double temp = min; modifier * temp <= modifier * max; temp += step) {

        #if defined(BOLTZMANN)
        init_boltzmann(temp);
        #endif

        // equilibrium phase
        for (unsigned int j = 0; j < TRAN; ++j) {
            #if !defined(AVX2_INTRINSICS)
            update(temp, grid);
            #else
            update(grid);
            #endif

        }

        // measurement phase
        unsigned int measurements = 0;
        double e = 0.0, e2 = 0.0, e4 = 0.0, m = 0.0, m2 = 0.0, m4 = 0.0;
        for (unsigned int j = 0; j < TMAX; ++j) {
            #if !defined(AVX2_INTRINSICS)
            update(temp, grid);
            #else
            update(grid);
            #endif
            if (j % calc_step == 0) {
                double energy = 0.0, mag = 0.0;
                int M_max = 0;
                energy = calculate(grid, &M_max);
                mag = abs(M_max) / (float)N;
                e += energy;
                e2 += energy * energy;
                e4 += energy * energy * energy * energy;
                m += mag;
                m2 += mag * mag;
                m4 += mag * mag * mag * mag;
                ++measurements;
            }
        }
        assert(index < NPOINTS);
        stats[index].t = temp;
        stats[index].e += e / measurements;
        stats[index].e2 += e2 / measurements;
        stats[index].e4 += e4 / measurements;
        stats[index].m += m / measurements;
        stats[index].m2 += m2 / measurements;
        stats[index].m4 += m4 / measurements;
        ++index;
    }
}

#if defined(ZERO_PADDING)

static void init(int grid[L+2][L+2])
{
    for (unsigned int i = 1; i <= L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            grid[i][j] = 1;
        }
    }
}

#else
static void init(int grid[L][L])
{
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            grid[i][j] = 1;
        }
    }
}
#endif


int main(void)
{

   
    // parameter checking
    static_assert(TEMP_DELTA != 0, "Invalid temperature step");
    static_assert(((TEMP_DELTA > 0) && (TEMP_INITIAL <= TEMP_FINAL)) || ((TEMP_DELTA < 0) && (TEMP_INITIAL >= TEMP_FINAL)), "Invalid temperature range+step");
    static_assert(TMAX % DELTA_T == 0, "Measurements must be equidistant"); // take equidistant calculate()
    static_assert((L * L / 2) * 4ULL < UINT_MAX, "L too large for uint indices"); // max energy, that is all spins are the same, fits into a ulong


      // print header
    //   printf("# L: %i\n", L);
    //   printf("# Minimum Temperature: %f\n", TEMP_INITIAL);
    //   printf("# Maximum Temperature: %f\n", TEMP_FINAL);
    //   printf("# Temperature Step: %.12f\n", TEMP_DELTA);
    //   printf("# Equilibration Time: %i\n", TRAN);
    //   printf("# Measurement Time: %i\n", TMAX);
    //   printf("# Data Acquiring Step: %i\n", DELTA_T);
    //   printf("# Number of Points: %i\n", NPOINTS);
    //   fflush(stdout);
    // the stats
    struct statpoint stat[NPOINTS];
    for (unsigned int i = 0; i < NPOINTS; ++i) {
        stat[i].t = 0.0;
        stat[i].e = stat[i].e2 = stat[i].e4 = 0.0;
        stat[i].m = stat[i].m2 = stat[i].m4 = 0.0;
    }

  

    // configure RNG
    srand(SEED);

    #if(XOSHIRO256PP)
    init_xoshiro();
    #endif

    #if defined(AVX2_INTRINSICS)
    init_rng256((uint64_t)time(NULL));
    init_constants();
    #elif defined(AVX512F_INTRINSICS)
    init_rng_state((uint64_t)time(NULL));
    #endif

    // start timer
    double start = wtime();


    init(grid);

    // temperature increasing cycle
    cycle(grid, TEMP_INITIAL, TEMP_FINAL, TEMP_DELTA, DELTA_T, stat);

    // stop timer
    double elapsed = wtime() - start;
    FILE * time_stream = fopen("time", "w");
    printf("%lf", elapsed);
    fprintf(time_stream, "# Total Simulation Time (sec): %lf\n", elapsed);


    FILE * table_stream = fopen("table", "w");

    fprintf(table_stream, "# Temp,E,E^2,E^4,M,M^2,M^4\n");
    for (unsigned int i = 0; i < NPOINTS; ++i) {
        fprintf(table_stream, "%lf,%.10lf,%.10lf,%.10lf,%.10lf,%.10lf,%.10lf\n",
               stat[i].t,
               stat[i].e / ((double)N),
               stat[i].e2 / ((double)N * N),
               stat[i].e4 / ((double)N * N * N * N),
               stat[i].m,
               stat[i].m2,
               stat[i].m4);
    }

    return 0;
}
