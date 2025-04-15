#include "params.h"


#if defined(ZERO_PADDING)

void update(const float temp, int grid[L+2][L+2]);
double calculate(int grid[L+2][L+2], int* M_max);

#else
void update(const float temp, int grid[L][L]);
double calculate(int grid[L][L], int* M_max);
#endif
void init_xoshiro();
