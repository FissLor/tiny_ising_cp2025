

#include "ising_rb.h"
#include <math.h>
#include <stdlib.h>
#if  defined(BOLTZMANN)

#include "xoshiropp.h"




void init_xoshiro(){
    xoshiro_state[0] = rand();
    xoshiro_state[1] = rand();
    xoshiro_state[2] = rand();
    xoshiro_state[3] = rand();

} 

static float exp_dE[5];  
static int   dE_vals[5] = { -8, -4, 0, 4, 8 };

void init_boltzmann( float temp ){
  for(int k = 0; k < 5; ++k){
    exp_dE[k] = expf( -dE_vals[k] / temp );
  }
}


void update(const float temp,
                      int   grid[L+2][L+2]  )
{
    // (you should already have called init_boltzmann(temp))

    for(int color = 0; color < 2; ++color){
        for(int i = 1; i <= L; ++i){
            int j0 = ((i + color) & 1) ? 2 : 1;

            #pragma omp simd 
            for(int j = j0; j <= L; j += 2){
                int  S      = grid[i][j];
                int  nb     = grid[i-1][j]
                            + grid[i+1][j]
                            + grid[i][j-1]
                            + grid[i][j+1];
                int  dE     = 2 * S * nb;

                // lookup Boltzmann probability
                float Pacc;
                switch(dE){
                  case  8: Pacc = exp_dE[4]; break;
                  case  4: Pacc = exp_dE[3]; break;
                  case  0: Pacc = exp_dE[2]; break;
                  case -4: Pacc = exp_dE[1]; break;
                  case -8: Pacc = exp_dE[0]; break;
                  default: Pacc = 0.0f;      // unreachable
                }

                // still a scalar RNG + branch here…
                float r = (next()% RAND_MAX) / (float)RAND_MAX;
                if( dE <= 0 || r < Pacc )
                    grid[i][j] = -S;
            }
        }
    }
}



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

#else

#include "xoshiropp.h"

void init_xoshiro(){
    xoshiro_state[0] = rand();
    xoshiro_state[1] = rand();
    xoshiro_state[2] = rand();
    xoshiro_state[3] = rand();

}

void update(const float temp, int grid[L+2][L+2]) {

    
    for (int color = 0; color < 2; ++color) {
        // color = 0 → “rojos”, color = 1 → “negros”
        for (int i = 1; i <= L; i+=2) {
            // Recorre solo los sitios de ese color con paso 2
            #pragma omp simd
            for (int j = 1; j <= L; j += 2) {
                int S = grid[i][j];
                int nb = grid[i-1][j]
                       + grid[i+1][j]
                       + grid[i][j-1]
                       + grid[i][j+1];
                int delta_E = 2 * S * nb;

                uint64_t r =  next()% RAND_MAX;
                float p = (float)r / (float)RAND_MAX;
                if (delta_E <= 0 || p < expf(-delta_E / temp)) {
                    grid[i][j] = -S;
                }
            }
        }
        for (int i = 2; i <= L; i+=2) {
            // Recorre solo los sitios de ese color con paso 2
            #pragma omp simd
            for (int j = 2; j <= L; j += 2) {
                int S = grid[i][j];
                int nb = grid[i-1][j]
                       + grid[i+1][j]
                       + grid[i][j-1]
                       + grid[i][j+1];
                int delta_E = 2 * S * nb;

                // uint64_t r =  next()% RAND_MAX;
                float p = (float)(next()% RAND_MAX) / (float)RAND_MAX;
                if (delta_E <= 0 || p < expf(-delta_E / temp)) {
                    grid[i][j] = -S;
                }
            }
        }
    }
}
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
#endif