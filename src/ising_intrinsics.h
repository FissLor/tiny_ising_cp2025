#if !defined(ISING_INTRINSICS_H)
#define ISING_INTRINSICS_H

#include "params.h"
#include <stdint.h>


#if defined(BOLTZMANN)
void init_boltzmann( float temp );
#endif


#if defined(AVX512F_INTRINSICS)
void update( int grid[L+2][L+2]);

#elif defined(AVX2_INTRINSICS)
void init_constants(void);
void update( int grid[L+2][L+2]);
void init_rng256(uint64_t seed);
void init_xoshiro();

#endif

double calculate(int grid[L+2][L+2], int* M_max);

#endif