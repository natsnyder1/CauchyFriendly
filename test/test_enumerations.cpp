#include "../include/cell_enumeration.hpp"
#include "../include/inc_enu_glpk.hpp"
#include "../include/cpu_timer.hpp"

void test_single_enumerations()
{
    const int m = 10;
    const int d = 3;
    double A[m*d];
    double root_point[d];
    
    for(int i = 0; i < m*d; i++)
        A[i] = 2*random_uniform()-1;
    for(int i = 0; i < d; i++)
        root_point[i] = random_uniform();
    
    const int cells_gen = cell_count_general(m, d);
    int B_glpk[cells_gen]; // overallocation
    int* B_nat; // overallocation
    int cell_count_glpk;
    int cell_count_nat;
    // Run GLPK Inc-Enu
    bool is_encoded = true;
    bool is_sort_B = true;
    run_inc_enu(B_glpk, &cell_count_glpk, A, root_point, 1, m, d, is_encoded, is_sort_B);
    printf("GLPK Inc Enu for HPA has %d cells:\n", cell_count_glpk);
    print_mat(B_glpk, 1, cell_count_glpk);
    // Run Nats Cell Enumeration
    const bool is_half_enum = false;
    const bool is_warn_checks = false;
    B_nat = make_enumeration_matrix(A, m, d, &cell_count_nat, NULL, is_half_enum, is_warn_checks);
    sort_encoded_B(B_nat, cell_count_nat);
    printf("Nats Cell Enum for HPA has %d cells:\n", cell_count_nat);
    print_mat(B_nat, 1, cell_count_nat);
    if(cell_count_nat != cell_count_glpk)
    {
        printf(YEL "  Warning: cell counts for glpk and nats cell enum do not match!" NC "\n");
        printf(YEL "  GLPK cell count = %d vs. Nats cell count = %d" NC "\n", cell_count_glpk, cell_count_nat);
        exit(1);
    }
    else
    {
        char yes[4] = "YES";
        char no[3] = "NO";
        char* choice = !is_Bs_different(B_glpk, B_nat, cell_count_nat) ? yes : no;
        printf("Do enumeration matrices match: %s\n", choice);
    }
    free(B_nat);
}

void test_enumeration_timing()
{
    const int iters = 10000;
    const int m = 10;
    const int d = 2;
    const int cells_cen = cell_count_central(m, d);
    double root_point[d];
    for(int i = 0; i < d; i++)
        root_point[i] = random_uniform();
    double* As = (double*) malloc(iters * m * d * sizeof(double));
    null_ptr_check(As);
    for(int i = 0; i < m*d*iters; i++)
        As[i] = 2*random_uniform()-1;
    
    printf("Timing cell enumeration for %d HPAs each of size %d x %d\n", iters, m, d);
    DiffCellEnumHelper dce_helper;
    dce_helper.init(m, d, 4);
    CPUTimer tmr;

    
    // Testing speed of inc-enu for GLPK
    int* Bs_glpk = (int*) malloc(iters * cells_cen * sizeof(int));
    null_ptr_check(Bs_glpk);
    int* cell_counts_glpk = (int*) malloc(iters * sizeof(int));
    null_ptr_check(cell_counts_glpk);
    bool is_encoded = true;
    bool is_sort_B = false;
    tmr.tic();
    run_inc_enu(Bs_glpk, cell_counts_glpk, As, root_point, iters, m, d, is_encoded, is_sort_B);
    tmr.toc(false);
    int time_ms_glpk = tmr.cpu_time_used;
    printf("GLPK Cell Enum took: %d ms\n", time_ms_glpk);

    // Testing spped of nats cell enumeration 
    int** Bs_nat = (int**) malloc(iters * sizeof(int*));
    null_dptr_check((void**)Bs_nat);
    int* cell_counts_nat = (int*) malloc(iters * sizeof(int));
    null_ptr_check(cell_counts_nat);
    bool is_half_enum = false;
    bool is_warn_check = false;
    tmr.tic();
    for(int i = 0; i < iters; i++)
        Bs_nat[i] = make_enumeration_matrix(As + i*m*d, m, d, cell_counts_nat + i, &dce_helper, is_half_enum, is_warn_check);
    tmr.toc(false);
    int time_ms_nat = tmr.cpu_time_used;
    printf("Nats Cell Enum took: %d ms\n", time_ms_nat);

    // Compare the results together
    bool results_same = true;
    for(int i = 0; i < iters; i++)
    {
        int* B_glpk = Bs_glpk + i * cells_cen;
        int cell_count_glpk = cell_counts_glpk[i];
        int* B_nat = Bs_nat[i];
        int cell_count_nat = cell_counts_nat[i];
        if(cell_count_nat != cell_count_glpk)
        {
            printf(YEL "---- Warn: On iteration %d, cells counts do not match! (GLPK=%d vs Nat=%d) ----" NC "\n", i, cell_count_glpk, cell_count_nat);
            results_same = false;
            break;
        }
        else 
        {
            sort_encoded_B(B_glpk, cell_count_glpk);
            sort_encoded_B(B_nat, cell_count_nat);
            if( is_Bs_different(B_glpk, B_nat, cell_count_nat) )
            {
                results_same = false;
                printf(YEL "---- Warn: On iteration %d, # cells = %d match, but the sign vectors do not! ----" NC "\n", i, cell_count_glpk);
                printf("HPA is:\n");
                print_mat(As + i * m * d, m, d, 16);
                printf("GLPK Cell Enu for HPA:\n");
                print_mat(B_glpk, 1, cell_count_glpk);
                printf("Nat Cell Enu for HPA:\n");
                print_mat(B_nat, 1, cell_count_nat);
                break;
            }
        }
    }

    char yes[4] = "YES";
    char no[3] = "NO";
    char* choice = results_same ? yes : no;
    printf("Are results of all enumerations the same: %s\n", choice);
    if(results_same)
        printf("Speed-up factor is: %.3lf\n", ((double)time_ms_glpk/((double) time_ms_nat)) );

    free(As);
    free(Bs_glpk);
    free(cell_counts_glpk);
    for(int i = 0; i < iters; i++)
        free(Bs_nat[i]);
    free(Bs_nat);
    free(cell_counts_nat);
    dce_helper.deinit();
}

int main()
{
    //test_single_enumerations();
    test_enumeration_timing();
}