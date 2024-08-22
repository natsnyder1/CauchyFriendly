#include "../include/random_variables.hpp"
#include "../include/lapacke_linalg.hpp"
#include "../include/cauchy_linalg.hpp"

// This test file compares lapacke's symmetric eigenvalue solver ...
// ... located in "../include/lapacke_linalg.hpp"
// to the standalone solver ...
// ... located in "../include/eig_solve.hpp"
// ... which is included as part of "../include/cauchy_linalg.hpp" (math lib with no external library includes)


void test_eigs_default()
{
    int             i, j;
	static double   u[5], v[5];
	static double   a[5][5] = {{1.0, 6.0, -3.0, -1.0, 7.0},
	{8.0, -15.0, 18.0, 5.0, 4.0}, {-2.0, 11.0, 9.0, 15.0, 20.0},
	{-13.0, 2.0, 21.0, 30.0, -6.0}, {17.0, 22.0, -5.0, 3.0, 6.0}};

	static double b[5][5]={ {10.0,1.0,2.0,3.0,4.0},
							{1.0,9.0,-1.0,2.0,-3.0},
							{2.0,-1.0,7.0,3.0,-5.0},
							{3.0,2.0,3.0,12.0,-1.0},
							{4.0,-3.0,-5.0,-1.0,15.0}};


	printf("Non-Symmetric MAT A IS:\n");
	for (i = 0; i <= 4; i++) {
		for (j = 0; j <= 4; j++)
			printf("%.8lf ", a[i][j]);
		printf("\n");
	}
	printf("\nEig Solve:\nEigenvalues:\n");
	evs_n_eigen(a[0], 5, u, v);
	for (i = 0; i <= 4; i++)
		printf("%.8lf +J %.8lf\n", u[i], v[i]);
    printf("\nEigenvectors (THIS IS INCOMPLETE!):\n");
	for (i = 0; i <= 4; i++) {
		for (j = 0; j <= 4; j++)
			printf("%.8lf ", a[i][j]);
		printf("\n");
	}

    printf("\nSymmetric MAT B IS:\n");
	for (i = 0; i <= 4; i++) {
		for (j = 0; j <= 4; j++)
			printf("%.8lf ", b[i][j]);
		printf("\n");
	}
	printf("\nSym Eig Solve:\nEigenvalues:\n");
	evs_n_eigen_symm(b[0], 5, u);
	for (i = 0; i <= 4; i++)
		printf("%.8lf\n", u[i]);
	printf("\nEigenvectors:\n");
	for (i = 0; i <= 4; i++) {
		for (j = 0; j <= 4; j++)
			printf("%.8lf ", b[i][j]);
		printf("\n");
	}
	printf("\n");
}

struct arg_sort_struct
{
	int idx;
	double val;
};

int arg_sort_cmp(const void* _as1, const void* _as2)
{
	arg_sort_struct* as1 = (arg_sort_struct*) _as1;
	arg_sort_struct* as2 = (arg_sort_struct*) _as2;
	double a = as1->val;
	double b = as2->val;
	if(a > b)
		return 1;
	else if( a < b )
		return -1;
	else
		return 0;
}

void compare_solutions(double* evals, double* evecs, double* evals_lp, double* evecs_lp, int n, bool with_print_sorted, bool with_print_diff)
{
	double* work = (double*) malloc(n*n*sizeof(double));
	// Arg Sort Evals of Stand Alone 
	arg_sort_struct* arg_sort = (arg_sort_struct*) malloc(n*sizeof(arg_sort_struct));
	for(int i = 0; i < n; i++)
	{
		arg_sort[i].idx = i; 
		arg_sort[i].val = evals[i];
	}
	qsort(arg_sort, n, sizeof(arg_sort_struct), arg_sort_cmp);
	memcpy(work, evecs, n*n*sizeof(double));
	for(int i = 0; i < n; i++)
	{
		evals[i] = arg_sort[i].val;
		// Move column at index arg_sort[i].idx to i-th column
		int idx = arg_sort[i].idx;
		for(int j = 0; j < n; j++)
			evecs[j*n + i] = work[j*n + idx];
	}
	if(with_print_sorted)
	{
		printf("Sorted Evals Standalone:\n");
		print_vec(evals, n);
		printf("Sorted Evecs Standalone:\n");
		print_mat(evecs, n, n);
	}

	// Arg Sort Evals of Lapacke
	for(int i = 0; i < n; i++)
	{
		arg_sort[i].idx = i; 
		arg_sort[i].val = evals_lp[i];
	}
	qsort(arg_sort, n, sizeof(arg_sort_struct), arg_sort_cmp);
	memcpy(work, evecs_lp, n*n*sizeof(double));
	for(int i = 0; i < n; i++)
	{
		evals_lp[i] = arg_sort[i].val;
		// Move column at index arg_sort[i].idx to i-th column
		int idx = arg_sort[i].idx;
		for(int j = 0; j < n; j++)
			evecs_lp[j*n + i] = work[j*n + idx];
	}
	if(with_print_sorted)
	{
		printf("Sorted Evals Lapacke:\n");
		print_mat(evals_lp, 1, n);
		printf("Sorted Evecs Lapacke:\n");
		print_mat(evecs_lp, n, n);
	}

	// Find max eigenvalue diff
	double max_diff_eval = 0;
	double max_diff_evec = 0;
	double print_precision = 2;
	for(int i = 0; i < n; i++)
	{
		work[i] = fabs(evals[i] - evals_lp[i]);
		if(work[i] > max_diff_eval)
			max_diff_eval = work[i];
	}
	printf("Max Eval Difference: %.4E\n", max_diff_eval);
	if(with_print_diff)
	{
		printf("Eigenvalue Differences:\n");
		print_mat(work, 1, n, print_precision);
	}
	// Find max eigenvector diff
	int n2 = n*n;
	for(int i = 0; i < n2; i++)
	{
		double ev = evecs[i];
		double ev_lp = evecs_lp[i];
		double sgn_diff = sgn(ev * ev_lp);
		work[i] = fabs(ev - sgn_diff * ev_lp);
		if(work[i] > max_diff_evec)
			max_diff_evec = work[i];
	}
	printf("Max Evec Difference: %.4E\n", max_diff_evec);
	if(with_print_diff)
	{
		printf("Eigenvector Differences:\n");
		print_mat(work, n, n, print_precision);
	}
	free(arg_sort);
	free(work);
}

// This is the format which the cauchy estimator would use
// Tests the stand alone eigenvalue solver against the 
void test_sym_eigs()
{
	// Seed -- Can comment out
	uint seed = time(NULL);
	srand(seed);
	printf("Seeding with: %u\n", seed);

    const int n = 7;
	const int n2 = n*n;
	bool with_print = true;
	bool with_print_sorted = false;
	bool with_print_diff = false;
	/* 
	const bool using_heap = false;
	// Example given with code at first
    double B[n*n]={  10.0,1.0,2.0,3.0,4.0,
                    1.0,9.0,-1.0,2.0,-3.0,
                    2.0,-1.0,7.0,3.0,-5.0,
                    3.0,2.0,3.0,12.0,-1.0,
                    4.0,-3.0,-5.0,-1.0,15.0};
    double evals[n];
    double evecs[n*n];
	double evals_lp[n];
    double evecs_lp[n*n];
	*/
	
	// Allocate, fill and make symmetric
	const bool using_heap = true;
	double* B = (double*) malloc(n*n*sizeof(double));
	double* work = (double*) malloc(n*n*sizeof(double));
	double* evals = (double*) malloc(n*sizeof(double));
	double* evecs = (double*) malloc(n*n*sizeof(double));
	double* evals_lp = (double*) malloc(n*sizeof(double));
    double* evecs_lp = (double*) malloc(n*n*sizeof(double));
	printf("Generating (%d x %d) symmetric matrix from random values\n", n, n);
	for(int i = 0; i < n2; i++)
		work[i] = random_normal(0, 1);
	inner_mat_prod(work, B, n, n);

	// Print B
	if(with_print)
	{
		printf("\nSymmetric MAT B IS:\n");
		print_mat(B, n, n);
	}
	// --- Run --- //

	// No-Lapacke (stand alone)
	sym_eig(B, evals, evecs, n);
	if(with_print)
	{
		printf("\nStandalone Sym Eig Solve:\nEigenvalues:\n");
		print_vec(evals, n);
		printf("\nStandalone Eigenvectors:\n");
		print_mat(evecs, n, n);
	}

	// With-Lapacke
	lapacke_sym_eig(B, evals_lp, evecs_lp, n);
	if(with_print)
	{
		printf("\nLapacke Sym Eig Solve:\nEigenvalues:\n");
		print_vec(evals_lp, n);
		printf("\nLapacke Eigenvectors:\n");
		print_mat(evecs_lp, n, n);
	}

	// Compare Outputs 
	printf("Sorting Eigenvalue solutions in ascending order, then comparing:\n\tEigenvector column solutions are kept consistent with the e-val swaps\n");
	compare_solutions(evals, evecs, evals_lp, evecs_lp, n, with_print_sorted, with_print_diff);
	printf("Note Eigenvector solutions can point in opposite directions of one another, so long as its consistent across entire vector\nThats all, folks!\n");
	// Free
	if(using_heap)
	{
		free(B);
		free(work);
		free(evals);
		free(evecs);
		free(evals_lp);
		free(evecs_lp);
	}
}

int main(void)
{
	//test_eigs_default();
    test_sym_eigs();
    return 0;
}