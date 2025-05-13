#if !defined(ISING_RB_H)
#define ISING_RB_H

#include "params.h"
#include <stdint.h>

#if defined(BOLTZMANN)
void init_boltzmann( float temp );
#endif

#if defined(ZERO_PADDING)

void update(const float temp, int grid[L+2][L+2]);


double calculate(int grid[L+2][L+2], int* M_max);

#else
void update(const float temp, int grid[L][L]);
double calculate(int grid[L][L], int* M_max);
#endif
void init_xoshiro();


#endif