#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../include/cpu_linalg.hpp"

const double ZERO_EPSILON = 1e-10;
const int ZERO_HP_MARKER_VALUE = 32;



int marg2d_remove_zeros_and_coalign(double* work_A, double* work_p, double* work_A2, double* work_p2, int* c_map, int* cs_map, int m)
{
    bool F[m];
    memset(F, 1, m * sizeof(bool));
    int F_idxs[m];

    // Normalize all HPs in work_A using 1-norm
    // Scale p[i] by the 1-norm of work_A[i,:]
    //checking for zero HPs, and marking these accordingly
    for(int i = 0; i < m; i++)
    {
        double* ai = work_A + i*2;
        double fabs_ai0 = fabs(ai[0]);
        double fabs_ai1 = fabs(ai[1]);
        bool is_zero = (fabs_ai0 < ZERO_EPSILON) && (fabs_ai1 < ZERO_EPSILON);
        // Mark with special zero index of ZERO_HP_MARKER_VALUE in F_Map if this is the case
        if(is_zero)
        {
            c_map[i] = ZERO_HP_MARKER_VALUE;
            cs_map[i] = ZERO_HP_MARKER_VALUE;
            F_idxs[i] = ZERO_HP_MARKER_VALUE;
            F[i] = 0;
            continue;
        }
        else 
        {
            double sum_a = fabs_ai0 + fabs_ai1;
            ai[0] /= sum_a;
            ai[1] /= sum_a;
            work_p[i] *= sum_a;
        }
    }

    // Loop over all non-zero rows of work_A
    int unique_count = 0;
    for(int i = 0; i < m; i++)
    {
        if(F[i])
        {
            c_map[i] = unique_count;
            cs_map[i] = 1;
            F_idxs[i] = i;
            work_p2[unique_count] = work_p[i];
            // Zeros check
            double* ai = work_A + i*2;    
            for(int j = i+1; j < m; j++)
            {
                if(F[j])
                {
                    double* aj = work_A + j*2;
                    // Check for positive coals first
                    bool pos_coal = (fabs(ai[0] - aj[0]) < COALIGN_MU_EPS) && (fabs(ai[1] - aj[1]) < COALIGN_MU_EPS);
                    if(pos_coal)
                    {
                        c_map[j] = unique_count;
                        cs_map[j] = 1;
                        F[j] = 0;
                        F_idxs[j] = i;
                        work_p2[unique_count] += work_p[j];
                        continue;
                    }
                    bool neg_coal = (fabs(ai[0] + aj[0]) < COALIGN_MU_EPS) && (fabs(ai[1] + aj[1]) < COALIGN_MU_EPS);
                    if(neg_coal)
                    {
                        c_map[j] = unique_count;
                        cs_map[j] = -1;
                        F[j] = 0;
                        F_idxs[j] = i;
                        work_p2[unique_count] += work_p[j];
                        continue;
                    }
                }
            }
            unique_count++;
        }
    }
    if(unique_count < m)
    {
        unique_count = 0;
        for(int i = 0; i < m; i++)
        {
            if(F_idxs[i] == i)
            {
                work_A2[unique_count*2+0] = work_A[i*2+0];
                work_A2[unique_count*2+1] = work_A[i*2+1];
                unique_count++;
            }
        }
    }
    else
        memcpy(work_A2, work_A, m * 2 * sizeof(double));

    return unique_count;
} 


int main()
{
    const int m = 10;
    const int d = 2;
    double work_A[m*d] = 
    {
        1, 0,
        0, 0,
        0, 0,
        1, 0,
        0, 1,
        1, 1,
        -1, 1,
        0, 1,
        0, 0,
        1, -1
    };
    double work_p[m] = {1,2,3,4,5,6,7,8,9,10};
    double work_A2[m*d];
    double work_p2[m];
    int c_map[m];
    int cs_map[m];
    int m_new = marg2d_remove_zeros_and_coalign(work_A, work_p, work_A2, work_p2, c_map, cs_map, m);

    printf("Original HPA A is:\n");
    print_mat(work_A, m, d);
    printf("Original p-vector is:\n");
    print_mat(work_p, 1, m);
    printf("Newly modified HPA is:\n");
    print_mat(work_A2, m_new, d);
    printf("Newly modified p-vector is:\n");
    print_mat(work_p2, 1, m_new);
    printf("Coalign Index Map is:\n");
    print_mat(c_map, 1, m);
    printf("Coalign Sign Map is:\n");
    print_mat(cs_map, 1, m);
    return 0;
}