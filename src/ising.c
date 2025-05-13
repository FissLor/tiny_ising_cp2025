#include "ising.h"

#include <math.h>
#include <stdlib.h>


#if defined(XOSHIRO256PP) && defined(ZERO_PADDING)
#include "xoshiropp.h"

void init_xoshiro(){
    xoshiro_state[0] = rand();
    xoshiro_state[1] = rand();
    xoshiro_state[2] = rand();
    xoshiro_state[3] = rand();

}


void update(const float temp, int grid[L+2][L+2])
{
    // typewriter update
    for (unsigned int i = 1; i <= L; ++i) {
        for (unsigned int j = 1; j < L; ++j) {
            int spin_old = grid[i][j];
            int spin_new = (-1) * spin_old;

            // computing h_before
            int spin_neigh_n = grid[i -1][j];
            int spin_neigh_e = grid[i][j + 1];
            int spin_neigh_w = grid[i][j-1];
            int spin_neigh_s = grid[i + 1][j];
            int h_before = -(spin_old * spin_neigh_n) - (spin_old * spin_neigh_e) - (spin_old * spin_neigh_w) - (spin_old * spin_neigh_s);

            // h after taking new spin
            int h_after = -(spin_new * spin_neigh_n) - (spin_new * spin_neigh_e) - (spin_new * spin_neigh_w) - (spin_new * spin_neigh_s);

            int delta_E = h_after - h_before;
            
            int rand = next() % RAND_MAX;
            float p = rand / (float)RAND_MAX;
            if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                grid[i][j] = spin_new;
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
#elif defined(XOSHIRO256PP)

#include "xoshiropp.h"

void init_xoshiro(){
    xoshiro_state[0] = rand();
    xoshiro_state[1] = rand();
    xoshiro_state[2] = rand();
    xoshiro_state[3] = rand();

}
void update(const float temp, int grid[L][L])
{
    // typewriter update
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int spin_old = grid[i][j];
            int spin_new = (-1) * spin_old;

            // computing h_beforer
            int spin_neigh_n = grid[(i + L - 1) % L][j];
            int spin_neigh_e = grid[i][(j + 1) % L];
            int spin_neigh_w = grid[i][(j + L - 1) % L];
            int spin_neigh_s = grid[(i + 1) % L][j];
            int h_before = -(spin_old * spin_neigh_n) - (spin_old * spin_neigh_e) - (spin_old * spin_neigh_w) - (spin_old * spin_neigh_s);

            // h after taking new spin
            int h_after = -(spin_new * spin_neigh_n) - (spin_new * spin_neigh_e) - (spin_new * spin_neigh_w) - (spin_new * spin_neigh_s);

            int delta_E = h_after - h_before;

            int rand = next() % RAND_MAX;
            float p = rand / (float)RAND_MAX;
            if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                grid[i][j] = spin_new;
            }
        }
    }
}
double calculate(int grid[L][L], int* M_max)
{
    int E = 0;
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int spin = grid[i][j];
            int spin_neigh_n = grid[(i + 1) % L][j];
            int spin_neigh_e = grid[i][(j + 1) % L];
            int spin_neigh_w = grid[i][(j + L - 1) % L];
            int spin_neigh_s = grid[(i + L - 1) % L][j];

            E += (spin * spin_neigh_n) + (spin * spin_neigh_e) + (spin * spin_neigh_w) + (spin * spin_neigh_s);
            *M_max += spin;
        }
    }
    return -((double)E / 2.0);
}

#elif defined(ZERO_PADDING)


void update(const float temp, int grid[L+2][L+2])
{
    // typewriter update
    for (unsigned int i = 1; i <= L; ++i) {
        for (unsigned int j = 1; j < L; ++j) {
            int spin_old = grid[i][j];
            int spin_new = (-1) * spin_old;

            // computing h_before
            int spin_neigh_n = grid[i -1][j];
            int spin_neigh_e = grid[i][j + 1];
            int spin_neigh_w = grid[i][j-1];
            int spin_neigh_s = grid[i + 1][j];
            int h_before = -(spin_old * spin_neigh_n) - (spin_old * spin_neigh_e) - (spin_old * spin_neigh_w) - (spin_old * spin_neigh_s);

            // h after taking new spin
            int h_after = -(spin_new * spin_neigh_n) - (spin_new * spin_neigh_e) - (spin_new * spin_neigh_w) - (spin_new * spin_neigh_s);

            int delta_E = h_after - h_before;
            float p = rand() / (float)RAND_MAX;
            if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                grid[i][j] = spin_new;
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
// #if defined(default_rand)
void update(const float temp, int grid[L][L])
{
    // typewriter update
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int spin_old = grid[i][j];
            int spin_new = (-1) * spin_old;

            // computing h_before
            int spin_neigh_n = grid[(i + L - 1) % L][j];
            int spin_neigh_e = grid[i][(j + 1) % L];
            int spin_neigh_w = grid[i][(j + L - 1) % L];
            int spin_neigh_s = grid[(i + 1) % L][j];
            int h_before = -(spin_old * spin_neigh_n) - (spin_old * spin_neigh_e) - (spin_old * spin_neigh_w) - (spin_old * spin_neigh_s);

            // h after taking new spin
            int h_after = -(spin_new * spin_neigh_n) - (spin_new * spin_neigh_e) - (spin_new * spin_neigh_w) - (spin_new * spin_neigh_s);

            int delta_E = h_after - h_before;
            float p = rand() / (float)RAND_MAX;
            if (delta_E <= 0 || p <= expf(-delta_E / temp)) {
                grid[i][j] = spin_new;
            }
        }
    }
}
double calculate(int grid[L][L], int* M_max)
{
    int E = 0;
    for (unsigned int i = 0; i < L; ++i) {
        for (unsigned int j = 0; j < L; ++j) {
            int spin = grid[i][j];
            int spin_neigh_n = grid[(i + 1) % L][j];
            int spin_neigh_e = grid[i][(j + 1) % L];
            int spin_neigh_w = grid[i][(j + L - 1) % L];
            int spin_neigh_s = grid[(i + L - 1) % L][j];

            E += (spin * spin_neigh_n) + (spin * spin_neigh_e) + (spin * spin_neigh_w) + (spin * spin_neigh_s);
            *M_max += spin;
        }
    }
    return -((double)E / 2.0);
}

#endif



