#include "params.h"
#include <stdint.h>
#if defined(BOLTZMANN)
void init_boltzmann( float temp );
#endif

#if defined(ZERO_PADDING)

#if defined(AVX512F_INTRINSICS)
void update( int grid[L+2][L+2]);
#elif defined(AVX2_INTRINSICS)
void init_constants(void);
void update( int grid[L+2][L+2]);
void init_rng256(uint64_t seed);

#else
void update(const float temp, int grid[L+2][L+2]);
#endif

double calculate(int grid[L+2][L+2], int* M_max);

#else
void update(const float temp, int grid[L][L]);
double calculate(int grid[L][L], int* M_max);
#endif
void init_xoshiro();
