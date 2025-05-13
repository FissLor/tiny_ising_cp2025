#include "ising_intrinsics.h"

#include <math.h>
#include <stdlib.h>

#include <stdio.h>

#if defined(AVX512F_INTRINSICS)


#include <immintrin.h>
#include <stdalign.h>

#include <stdint.h>
#include <time.h>
#include <immintrin.h>

// your global RNG state for 16 lanes
static __m512i rng_state;

// SplitMix64 round to scramble a 64‑bit value
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Call once before your first AVX512 sweep:
void init_rng_state(uint64_t seed) {
    uint64_t state = seed;
    uint32_t lanes[16];
    for (int i = 0; i < 16; ++i) {
        // generate a 64‑bit random, then truncate to 32 bits
        uint64_t r = splitmix64(&state);
        lanes[i] = (uint32_t)r | 1u;       // ensure nonzero for xorshift
    }
    // load into the AVX‑512 register (lane 0→element 0, …, lane 15→element 15)
    rng_state = _mm512_setr_epi32(
        lanes[0],  lanes[1],  lanes[2],  lanes[3],
        lanes[4],  lanes[5],  lanes[6],  lanes[7],
        lanes[8],  lanes[9],  lanes[10], lanes[11],
        lanes[12], lanes[13], lanes[14], lanes[15]
    );
}

#define STRIDE (L+2)

// your five ΔE values
static const int dE_vals[5] = { -8, -4, 0, 4, 8 };
// table of Boltzmann factors (aligned to 64 B)
alignas(64) float exp_dE[5];
void init_boltzmann(float T) {
  for(int k=0;k<5;++k) exp_dE[k] = expf(-dE_vals[k]/T);
}

// must compile with -mavx512f -mavx512dq -mavx512vl
void update(int grid[L+2][L+2]) {
    // (init_boltzmann(T) has already been called)
    for(int color=0;color<2;++color) {
        for(int i=1;i<=L;++i) {
            // starting column index for this color
            int j0 = ((i+color)&1) ? 1 : 2;

            // process 16 sites at a time: j0, j0+2, j0+4, …, j0+30
            for(int j=j0; j<=L; j+=16) {
                // build vector of absolute 1D indices = i*STRIDE + j + {0,2,4…30}
                __m512i base = _mm512_set1_epi32(i*STRIDE + j);
                __m512i offs = _mm512_setr_epi32(
                     0,  2,  4,  6,   8, 10, 12, 14,
                    16, 18, 20, 22,  24, 26, 28, 30
                );
                __m512i idx = _mm512_add_epi32(base, offs);

                // gather the spin and its 4 neighbors
                __m512i spin =  _mm512_i32gather_epi32(idx,     grid,4);
                __m512i up   =  _mm512_i32gather_epi32(
                                  _mm512_sub_epi32(idx,_mm512_set1_epi32( STRIDE)),
                                  grid,4);
                __m512i dn   =  _mm512_i32gather_epi32(
                                  _mm512_add_epi32(idx,_mm512_set1_epi32( STRIDE)),
                                  grid,4);
                __m512i lt   =  _mm512_i32gather_epi32(
                                  _mm512_sub_epi32(idx,_mm512_set1_epi32(1)),
                                  grid,4);
                __m512i rt   =  _mm512_i32gather_epi32(
                                  _mm512_add_epi32(idx,_mm512_set1_epi32(1)),
                                  grid,4);

                // neighbor sum and ΔE = 2 * spin * nb
                __m512i nb = _mm512_add_epi32(
                                _mm512_add_epi32(up, dn),
                                _mm512_add_epi32(lt, rt)
                             );
                __m512i dE = _mm512_mullo_epi32(
                                _mm512_set1_epi32(2),
                                _mm512_mullo_epi32(spin, nb)
                             );

                // lookup Boltzmann factor: index = (dE + 8) >> 2
                __m512i tbl_idx = _mm512_srli_epi32(
                                     _mm512_add_epi32(dE, _mm512_set1_epi32(8)),
                                     2
                                  );
                __m512 prob = _mm512_i32gather_ps(tbl_idx, exp_dE, 4);

                // vector xorshift RNG → new rng_state, 16 lanes of 32‑bit ints
                __m512i x = rng_state;
                x = _mm512_xor_si512(x, _mm512_slli_epi32(x,13));
                x = _mm512_xor_si512(x, _mm512_srli_epi32(x,17));
                x = _mm512_xor_si512(x, _mm512_slli_epi32(x,5));
                rng_state = x;

                // convert to floats in [0,1):
                __m512 randf = _mm512_cvtepi32_ps(
                                  _mm512_and_epi32(x, _mm512_set1_epi32(RAND_MAX))
                                ) * (1.0f/(float)RAND_MAX);

                // build mask: ΔE ≤ 0  OR  randf < prob
                __mmask16 m1 = _mm512_cmp_epi32_mask(dE,
                                  _mm512_setzero_si512(),
                                  _MM_CMPINT_LE);
                __mmask16 m2 = _mm512_cmp_ps_mask(randf, prob,
                                  _CMP_LT_OS);
                __mmask16 m = m1 | m2;

                // compute flipped spins = –spin
                __m512i neg = _mm512_sub_epi32(
                                 _mm512_setzero_si512(),
                                 spin
                               );
                // blend: where m=1 pick neg, else spin
                __m512i out = _mm512_mask_blend_epi32(m, spin, neg);

                // scatter result back into grid
                _mm512_i32scatter_epi32(grid, idx, out, 4);
            }
        }
    }
}

#elif defined(AVX2_INTRINSICS)

#include <immintrin.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "xoshiropp.h"

// ————— Globals —————

// Precomputed Boltzmann factors (aligned to 32 bytes for AVX2 gather)
static float exp_dE[5] __attribute__((aligned(32)));  
// 8‑lane xorshift RNG state
static __m256i rng_state256;


// precomputed constant vectors:
static __m256i V_RANDMAX;      // = _mm256_set1_epi32(RAND_MAX)
static __m256  V_INV_R;        // = _mm256_set1_ps(1.0f/(float)RAND_MAX)
static __m256i V_TWO;          // = _mm256_set1_epi32(2)
static __m256i V_ADD8;         // = _mm256_set1_epi32(8)
static __m256  V_ZERO_F;       // = _mm256_set1_ps(0.0f)


// ————— SplitMix64 for seeding —————
static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Initialize the 8‑lane RNG with nonzero seeds
void init_rng256(uint64_t seed) {
    uint64_t st = seed;
    uint32_t lanes[8];
    for (int i = 0; i < 8; ++i) {
        // generate 64‑bit random, truncate to 32‑bit, ensure nonzero
        lanes[i] = (uint32_t)splitmix64(&st) | 1u;
    }
    rng_state256 = _mm256_setr_epi32(
        lanes[0],lanes[1],lanes[2],lanes[3],
        lanes[4],lanes[5],lanes[6],lanes[7]
    );
}

void init_xoshiro(){
    xoshiro_state[0] = rand();
    xoshiro_state[1] = rand();
    xoshiro_state[2] = rand();
    xoshiro_state[3] = rand();

} 

// Precompute exp(−ΔE/T) for ΔE∈{−8,−4,0,4,8}; call once per new T
void init_boltzmann(float T) {
    const int dE_vals[5] = { -8, -4, 0, 4, 8 };
    for (int k = 0; k < 5; ++k) {
        exp_dE[k] = expf(-dE_vals[k] / T);
    }
}

// Call once at startup (after init_rng256 and init_boltzmann)
void init_constants(void) {
    V_RANDMAX = _mm256_set1_epi32(RAND_MAX);
    V_INV_R   = _mm256_set1_ps(1.0f/(float)RAND_MAX);
    V_TWO     = _mm256_set1_epi32(2);
    V_ADD8    = _mm256_set1_epi32(8);
    V_ZERO_F  = _mm256_set1_ps(0.0f);
}

// ————— AVX2 red–black Metropolis update —————
void update(int grid[L+2][L+2]) {
    const int maxJ = L - 14;  // highest j for which j+14 ≤ L

    for (int color = 0; color < 2; ++color) {
        for (int i = 1; i <= L; ++i) {
            int *row  = &grid[i][0];
            int *rowU = &grid[i-1][0];
            int *rowD = &grid[i+1][0];

            // start at j0 so that (i+j)%2 == color
            int j0 = ((i + color) & 1) ? 1 : 2;
            int j;

            // --- vectorized blocks (8 lanes, stride=2 → span 16 cols) ---
            for (j = j0; j <= maxJ; j += 16) {
                __m256i idx = _mm256_setr_epi32(
                    j, j+2, j+4, j+6,
                    j+8, j+10, j+12, j+14
                );
                // gather center + neighbors
                __m256i c  = _mm256_i32gather_epi32(row,  idx, 4);
                __m256i up = _mm256_i32gather_epi32(rowU, idx, 4);
                __m256i dn = _mm256_i32gather_epi32(rowD, idx, 4);
                __m256i il = _mm256_sub_epi32(idx, _mm256_set1_epi32(1));
                __m256i ir = _mm256_add_epi32(idx, _mm256_set1_epi32(1));
                __m256i lt = _mm256_i32gather_epi32(row, il, 4);
                __m256i rt = _mm256_i32gather_epi32(row, ir, 4);

                // ΔE = 2 * c * (up + dn + lt + rt)
                __m256i nb = _mm256_add_epi32(
                                _mm256_add_epi32(up, dn),
                                _mm256_add_epi32(lt, rt)
                             );
                __m256i dE = _mm256_mullo_epi32(
                                V_TWO,
                                _mm256_mullo_epi32(c, nb)
                             );

                // lookup P = exp_dE[(dE+8)>>2]
                __m256i ti   = _mm256_srli_epi32(
                                   _mm256_add_epi32(dE, V_ADD8), 2
                               );
                __m256  prob = _mm256_i32gather_ps(exp_dE, ti, 4);

                // xorshift RNG → x
                __m256i x = rng_state256;
                x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
                x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
                x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
                rng_state256 = x;

                // normalize → randf ∈ [0,1)
                __m256i xm    = _mm256_and_si256(x, V_RANDMAX);
                __m256  randf = _mm256_mul_ps(
                                    _mm256_cvtepi32_ps(xm),
                                    V_INV_R
                                );

                // mask = (dE ≤ 0) || (randf < prob)
                __m256  m1   = _mm256_cmp_ps(
                                   _mm256_cvtepi32_ps(dE),
                                   V_ZERO_F,
                                   _CMP_LE_OQ
                               );
                __m256  m2   = _mm256_cmp_ps(randf, prob, _CMP_LT_OQ);
                __m256  mask = _mm256_or_ps(m1, m2);
                __m256i mskI = _mm256_castps_si256(mask);

                // blend: keep spin or flip
                __m256i neg = _mm256_sub_epi32(_mm256_setzero_si256(), c);
                __m256i out = _mm256_blendv_epi8(c, neg, mskI);

                // scatter via temporary array
                int tmp[8];
                _mm256_storeu_si256((__m256i*)tmp, out);
                for (int k = 0; k < 8; ++k) {
                    row[j + 2*k] = tmp[k];
                }
            }

            // --- scalar tail for leftover j’s ---
            for (; j <= L; j += 2) {
                int  S   = row[j];
                int  nb2 = rowU[j] + rowD[j] + row[j-1] + row[j+1];
                int  dE2 = 2 * S * nb2;
                float P  = exp_dE[(dE2 + 8) >> 2];
                int   r  = next() % RAND_MAX;
                float p  = r / (float)RAND_MAX;
                if (dE2 <= 0 || p < P) {
                    row[j] = -S;
                }
            }
        }
    }
}
#endif


double calculate(int grid[L+2][L+2], int* M_max)
{
    int E = 0;
    for (unsigned int i = 1; i <= L; ++i) {
        for (unsigned int j = 1; j <= L; ++j) {
            int spin = grid[i][j];
            int spin_neigh_n = grid[i -1][j];
            int spin_neigh_e = grid[i][j + 1];
            int spin_neigh_w = grid[i][j-1];
            int spin_neigh_s = grid[i + 1][j];

            E += (spin * spin_neigh_n) + (spin * spin_neigh_e) + (spin * spin_neigh_w) + (spin * spin_neigh_s);
            *M_max += spin;
        }
    }
    return -((double)E / 2.0);
}