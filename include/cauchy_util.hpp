#ifndef _CAUCHY_UTIL_HPP_
#define _CAUCHY_UTIL_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_linalg.hpp"
#include "eig_solve.hpp"
#include "gtable.hpp"
#include "array_logging.hpp"
#include "dynamic_models.hpp"

// Speyer's initialization routine assumes that Var is a positive definite matrix
// This is made sure of by the numerical checks each window conducts on its computed covariance matrix
// It may be the case that Var has very small eigenvalues, and is very close to being positive semidefinite
// It may also be the case that Var was computed to have positive eigenvalues, but due to eigenvalue solver instability ...
// ... they may come out (small) negative or zero now when checked in this function
// If any eigenvalue of the covariance is detected to be less than COV_EIGENVALUE_TOLERANCE ...
// ... the diagonal of the covariance gets boosted by COV_EIGENVALUE_TOLERANCE - min(eig(Var))
// On enterance of this function, if window_var_boost != NULL ... 
// ... the diagonal of the covariance is boosted by this vector. This helps make the covariance less correlated (more Pos Def) for initialization
// This function could also implement other boosting strategies, such as multiplicative boosting (this is not yet implemented, however).
void speyers_window_init(const int N, double* x1_hat, double* Var, 
                  double* H, double gamma, double z_1, 
                  double* A_0, double* p_0, double* b_0, 
                  const int window_initializer_idx, const int window_initializee_idx, double* window_var_boost)
{
    double* work = (double*) malloc(N*N*sizeof(double));
    double* work2= (double*) malloc(N*N*sizeof(double));
	double* eigs = (double*) malloc(N * sizeof(double));
    memset(eigs, 0, N*sizeof(double));

    // Boost variance on enterance to this function, if window_var_boost != NULL
    bool print_var = false;
    if(window_var_boost != NULL)
        for(int i = 0; i < N; i++)
            Var[i*N+i] += window_var_boost[i];
    
    // Check for positive definiteness of Var
    
    // LAPACKE CODE
    //memcpy(work, Var, N*N*sizeof(double));
    //memset(work2, 0, N*sizeof(double));
    //LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, work, N, work2);
    //double min_eig = work2[0];
    // STAND ALONE EVAL/VEC CODE
    sym_eig(Var, work2, work, N);
    double min_eig = array_min<double>(work2, N);

    for(int i = 0; i < N; i++)
    {    
	    if(work2[i] < COV_EIGENVALUE_TOLERANCE)
        {    
            printf(YEL "[WARN (Window manager) Speyer Initialization:] Initializer: Win %d, Initializee: Win %d, eig[%d]=%.3E of Covariance very small or negative!"
                NC "\n", window_initializer_idx, window_initializee_idx, i, work2[i]);
            print_var = true;
        }
        if( work2[i] < min_eig )
	  min_eig = work2[i];
    }
    if(print_var)
    {
      double min_eig = array_min(work2, N);
      double min_boost = COV_EIGENVALUE_TOLERANCE - min_eig;
      printf("Eig-Values of Covar:\n");
      print_mat(work2, 1, N, 16);
      printf("Covar of initializer is\n");
      print_mat(Var, N, N, 16);
      printf("Making Covariance more positive definate for initialization!\n");
      for(int i = 0; i < N; i++)
          Var[i*N+i] += min_boost;
    }

    // Begin Speyer Initialization
    inner_mat_prod(H, work, 1, N);
    matmatmul(Var, work, work2, N, N, N, N);
    matmatmul(work2, Var, work, N, N, N, N);
    double Hx1_hat = dot_prod(H, x1_hat, N);
    double scale = gamma * gamma + (z_1 - Hx1_hat) * (z_1 - Hx1_hat);
    scale_mat(work, 1.0 / scale, N, N);
    add_mat(A_0, Var, work, 1.0, N, N);

    // var_work2 = Var + Var @ H^T @ H @ Var / ( gamma^2 + (z_1 - H @ x1_hat)^2)

    // Form A_0 using lapack's (d)ouble (sy)metric (ev)al/vec routine 
    // LAPACKE CODE
    //LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, A_0, N, eigs);
    // STAND ALONE EVAL/VEC CODE
    memcpy(work, A_0, N*N*sizeof(double));
    sym_eig(work, eigs, A_0, N); // A_0 holds the eigenvectors (transposed to rows below)

    // Form b_0
    double scale2 = (z_1 - Hx1_hat) / scale;
    matvecmul(Var, H, work, N, N);
    scale_mat(work, scale2, N, N);
    sub_vecs(x1_hat, work, b_0, N);

    // Form p_0
    matvecmul(A_0, H, work2, N, N, true); // H @ A = A_T @ H_T
    matvecmul(Var, H, work, N, N);
    double HVarH_T = dot_prod(H, work, N);
    double scale3 = (scale + HVarH_T) / gamma;
    for(int i = 0; i < N; i++)
    {
        p_0[i] = eigs[i] / scale3; // p_i_bar / h_i_bar
        p_0[i] *= work2[i]; // p_i_bar
        p_0[i] /= sgn(work2[i]); // p_i
    }
    reflect_array(A_0,N,N);
	free(work);
	free(work2);
	free(eigs);
}


double normalize_l1(double* x, const int n)
{
	double norm_factor = 0;
	for(int i = 0; i < n; i++)
		norm_factor += fabs(x[i]);
	for(int i = 0; i < n; i++)
		x[i] /= norm_factor;
	return norm_factor;
}

// Coaligns the Gamma matrix given as part of the coalignment routine
// returns tmp_Gamma, tmp_beta, and tmp_cmcc -> paramas ready for coalignment
int precoalign_Gamma_beta(double* Gamma, double* beta, int cmcc, int d, double* tmp_Gamma, double* tmp_beta)
{
	// Need to pre-coalign and normalize Gamma and beta
	// Convert Gamma from column storage (d x cmcc) to row storage (cmcc x d)
	for(int i = 0; i < cmcc; i++)
	{
		for(int j = 0; j < d; j++)
			tmp_Gamma[i*d + j] = Gamma[j*cmcc + i];
		tmp_beta[i] = beta[i];
	}
	// Note that Gamma is treated as hyperplanes, where the columns of Gamma (rows of tmp_Gamma) are each hyperplane 
	for(int i = 0; i < cmcc; i++)
	{
		double norm_factor = normalize_l1(tmp_Gamma + i*d, d);
		tmp_beta[i] *= norm_factor;
	}
	// Coalign tmp_Gamma, in the strange event they are coaligned
	#if _WIN32
		bool F[MAX_HP_SIZE];
	#else
		bool F[cmcc];
	#endif
	memset(F, 1, cmcc * sizeof(bool));
	int unique_count = 0;
	for(int i = 0; i < cmcc-1; i++)
	{
		if(F[i])
		{
			double* Gam_root = tmp_Gamma + i*d;
			for(int j = i+1; j < cmcc; j++)
			{
				if(F[j])
				{
					double* Gam_compare = tmp_Gamma + j*d;
					bool pos_gate = true;
					bool neg_gate = true;
					for(int l = 0; l < d; l++)
					{
						double root = Gam_root[l];
						double compare = Gam_compare[l];
						if(pos_gate)
							pos_gate &= fabs(root - compare) < COALIGN_MU_EPS;
						if(neg_gate)
							neg_gate &= fabs(root + compare) < COALIGN_MU_EPS;
						if(!(pos_gate || neg_gate) )
							break;
					}   
					assert(!(pos_gate && neg_gate));
					if(pos_gate || neg_gate)
					{
						F[j] = 0;
						tmp_beta[i] += tmp_beta[j];
					}
				}
			}
			unique_count += 1;
		}
	}
	int tmp_cmcc = 0;
	for(int i = 0; i < cmcc; i++)
		tmp_cmcc += F[i];

	// Only if we have coalignments do we need to move memory
	if( tmp_cmcc != cmcc)
	{
		unique_count = 1;
		for(int j = 1; j < cmcc; j++)
		{
			if(F[j])
			{
				if(unique_count < j)
				{
					memcpy(tmp_Gamma + unique_count*d, tmp_Gamma + j*d, d*sizeof(double));
					tmp_beta[unique_count] = tmp_beta[j];
				}
				unique_count++;
			}
		}
		return tmp_cmcc;
	}
	else
		return cmcc;
}

// Helper Elements for Fast Incremental Enumeration and Fast Term Reduction
struct FastTermRedHelper
{
  int max_num_terms;
  int d;
  // Term reduction helpers
  int* F;
  int* F_TR;
  // Fast term reduction helpers
  double** ordered_points;
  int** forward_map;
  int** backward_map;

  void init(int _d, int _max_num_terms)
  {
    max_num_terms = _max_num_terms;
    d = _d;
    F = (int*) malloc( max_num_terms * sizeof(int));
    null_ptr_check(F);
    for(int i = 0; i < max_num_terms; i++)
      F[i] = i;
    // Term reduction flags
    F_TR = (int*)malloc(max_num_terms * sizeof(int));
    null_ptr_check(F_TR);
    init_ordered_point_maps();
    assert_TR_search_index_pattern();
  }

  void realloc_helpers(int _max_num_terms)
  {
    F = (int*) realloc(F, _max_num_terms * sizeof(int));
    null_ptr_check(F);
    if(_max_num_terms > max_num_terms)
      for(int i = max_num_terms; i < _max_num_terms; i++)
        F[i] = i;
    max_num_terms = _max_num_terms;
    // Allocate memory for the helper element arrays
    F_TR = (int*) realloc(F_TR, max_num_terms * sizeof(int));
    null_ptr_check(F_TR);
    realloc_ordered_point_maps();
    assert_TR_search_index_pattern();
  }

  void assert_TR_search_index_pattern()
  {
    // Make absolutely sure search index pattern is set correctly
	#if _WIN32
			bool F_sip[MAX_HP_SIZE];
	#else
		  bool F_sip[d];
	#endif
	memset(F_sip, 0, d*sizeof(bool));
    for(int i = 0; i < d; i++)
    {
      if(TR_SEARCH_IDXS_ORDERING[i] >= d)
      {
        printf(RED "[ERROR FastTermRedHelper initialization:]\n"
                   "  TR_SEARCH_IDXS_ORDERING in cauchy_type.hpp not set correctly for d=%d states!\n"
                   "  Please modify this array to have unique integers [0,...,%d] in a specified order\n"
                   "  Exiting! Please Fix!\n"
               NC  "\n", d, d-1);
        exit(1);
      }
      else
        F_sip[TR_SEARCH_IDXS_ORDERING[i]] = 1;
    }
    for(int i = 0; i < d; i++)
    {
      if(F_sip[i] == 0)
      {
        // indices not set correctly
        printf(RED "[ERROR FastTermRedHelper initialization:]\n"
                   "  TR_SEARCH_IDXS_ORDERING in cauchy_type.hpp not set correctly for d=%d states!\n"
                   "  Please modify this array to have unique integers [0,...,%d] in a specified order\n"
                   "  Exiting! Please Fix!\n"
               NC  "\n", d, d-1);
        exit(1);
      }
    }
  }

  void init_ordered_point_maps()
  {
      int n = max_num_terms;
      ordered_points = (double**) malloc(d * sizeof(double*));
      null_dptr_check((void**)ordered_points);
      forward_map = (int**) malloc(d * sizeof(int*));
      null_dptr_check((void**)forward_map);
      backward_map = (int**) malloc(d * sizeof(int*));
      null_dptr_check((void**)backward_map);
      for(int i = 0; i < d; i++)
      {
          ordered_points[i] = (double*) malloc(n * sizeof(double));
          null_ptr_check(ordered_points[i]);
          forward_map[i] = (int*) malloc(n * sizeof(int));
          null_ptr_check(forward_map[i]);
          backward_map[i] = (int*) malloc(n * sizeof(int));
          null_ptr_check(backward_map[i]);
      }
  }

  void realloc_ordered_point_maps()
  {
      int n = max_num_terms;
      for(int i = 0; i < d; i++)
      {
        ordered_points[i] = (double*) realloc(ordered_points[i], n * sizeof(double));
        null_ptr_check(ordered_points[i]);
        forward_map[i] = (int*) realloc(forward_map[i], n * sizeof(int));
        null_ptr_check(forward_map[i]);
        backward_map[i] = (int*) realloc(backward_map[i], n * sizeof(int));
        null_ptr_check(backward_map[i]);
      }
  }

  void dealloc_ordered_point_maps()
  {
      for(int i = 0; i < d; i++)
      {
          free(ordered_points[i]);
          free(forward_map[i]);
          free(backward_map[i]);
      }
      free(ordered_points);
      free(forward_map);
      free(backward_map);
  }

  void deinit()
  {
    free(F);
    free(F_TR);
    dealloc_ordered_point_maps();
  }
};

// Creates a forwards pointing flag array from the backward pointing flag array that is created by term reduction
struct ForwardFlagArray
{
	int** Fs;
	int* F_counts;
	int F_count;
	int* F_TR;
	int num_terms_after_reduction;
	int* chunk_sizes;
	int* reduced_terms_per_chunk;
	int max_threads;

	ForwardFlagArray(int* F, int num_F, int _max_threads = 0)
	{
	  F_count = num_F;
	  F_TR = F;
	  Fs = (int**) malloc(F_count * sizeof(int*));
	  null_dptr_check((void**)Fs);
	  F_counts = (int*) malloc(F_count * sizeof(int));
	  null_ptr_check(F_counts);
	  max_threads = _max_threads;
	  chunk_sizes = (int*) malloc(max_threads * sizeof(int));
	  reduced_terms_per_chunk = (int*) malloc(max_threads * sizeof(int));
	  if(max_threads != 0)
	  {
		null_ptr_check(chunk_sizes);
		null_ptr_check(reduced_terms_per_chunk);
	  }
	  for(int i = 0; i < F_count; i++)
	  {
		  Fs[i] = NULL;
		  F_counts[i] = 0;
	  }
	  // Fill ForwardFlagArray
	  num_terms_after_reduction = 0;
	  const int INIT_ARRAY_SIZE = 10;
	  for(int j = 0; j < num_F; j++)
	  {
		if(F[j] != j)
		{
		  int i = F[j];
		  if( (F_counts[i] % INIT_ARRAY_SIZE) == 0 )
		  {
			Fs[i] = (int*) realloc(Fs[i], (F_counts[i] + INIT_ARRAY_SIZE) * sizeof(int));
			null_ptr_check(Fs[i]);
		  }
		  Fs[i][F_counts[i]++] = j;
		}
		else
		  num_terms_after_reduction++;
	  }
	  // Realloc the array to minimal size
	  for(int j = 0; j < num_F; j++)
	  {
		  if(F_counts[j])
		  {
			Fs[j] = (int*) realloc(Fs[j], F_counts[j] * sizeof(int));
			null_ptr_check(Fs[j]);
		  }
	  }
	}

	int get_balanced_threaded_flattening_indices(const int num_threads, int* start_idxs, int* end_idxs, int win_num, int step, int total_steps, bool is_print=false)
	{
	  if(num_threads > max_threads)
	  {
		//chunk_sizes = (int*) realloc(chunk_sizes, num_threads * sizeof(int));
		//null_ptr_check(chunk_sizes);
		//reduced_terms_per_chunk = (int*) realloc(reduced_terms_per_chunk, num_threads * sizeof(int));
		//null_ptr_check(reduced_terms_per_chunk);
		//max_threads = num_threads;
		assert(false);
	  }
	  int chunk_size = F_count / num_threads;
	  assert(chunk_size > 0);
	  int chunk_count = 0;
	  int start_idx = 0;
	  int idx_count = 0;
	  int new_num_threads = num_threads;
	  memset(reduced_terms_per_chunk, 0, max_threads * sizeof(int));
	  for(int i = 0; i < F_count; i++)
	  {
		if(F_TR[i] == i)
		{
		  if(chunk_count < chunk_size)
		  {
			reduced_terms_per_chunk[idx_count] += 1;
			if(Fs[i] == NULL)
			  chunk_count += 1;
			else
			  chunk_count += F_counts[i]+1;
		  }
		  else 
		  {
			assert(idx_count < num_threads);
			start_idxs[idx_count] = start_idx;
			end_idxs[idx_count] = i;
			start_idx = i;
			chunk_sizes[idx_count] = chunk_count;
			chunk_count = (Fs[i] == NULL) ? 1 : F_counts[i]+1;
			idx_count += 1;
			if(idx_count == num_threads-1)
			  chunk_size += F_count % num_threads;
			if(idx_count < num_threads)
			  reduced_terms_per_chunk[idx_count] += 1;
		  }
		}
	  }
	  if(idx_count < num_threads-1)
	  {
		printf("[ForwardFlagArray Win Num %d, Step %d/%d:] Lowering thread count from %d to %d (Before FTR: %d, Max After FTR: %d)\n", win_num, step, total_steps, num_threads, idx_count+1, F_count, num_terms_after_reduction);
		//is_print = true;
		int sum_chunks = 0;
		int sum_rtpc = 0;
		if(idx_count == 0)
		{
		  chunk_sizes[idx_count] = F_count;
		  reduced_terms_per_chunk[idx_count] = num_terms_after_reduction;
		  new_num_threads = idx_count+1;
		  start_idxs[idx_count] = 0;
		  end_idxs[idx_count] = F_count;
		}
		else 
		{
		  for(int i = 0; i < idx_count; i++)
		  {
			sum_chunks += chunk_sizes[i];
			sum_rtpc += reduced_terms_per_chunk[i];
		  }
		  if( (sum_chunks < F_count) || (sum_rtpc < num_terms_after_reduction) )
		  { 
			chunk_sizes[idx_count] = F_count - sum_chunks;
			reduced_terms_per_chunk[idx_count] = num_terms_after_reduction - sum_rtpc;
			new_num_threads = idx_count+1;
			start_idxs[idx_count] = end_idxs[idx_count-1];
			end_idxs[idx_count] = F_count;
		  }
		  else if((sum_chunks > F_count) || (sum_rtpc > num_terms_after_reduction))
		  {
			printf("...Still an error in Forward flag array balance threads...Exiting!\n");
			exit(1);
		  }
		  else 
		  {
			start_idxs[idx_count] = end_idxs[idx_count-1];
			end_idxs[idx_count] = F_count;
			new_num_threads = idx_count;
		  }
		}
		//printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Lowering thread count to %d...\n", win_num, step, total_steps, new_num_threads);
	  }
	  if(idx_count > num_threads)
	  {
		printf("Error in chunking up flattening indices (idx_count > num_threads)! BUG! Please Fix!\n");
		exit(1);
	  }
	  if(idx_count == (num_threads-1))
	  {
		if(end_idxs[idx_count-1] == F_count)
		{
		  printf("[ForwardFlagArray Win Num %d, Step %d/%d:] Lowering thread count from %d to %d (Before FTR: %d, Max After FTR: %d)\n", win_num, step, total_steps, num_threads, idx_count+1, F_count, num_terms_after_reduction);
		  //is_print = true;
		  //printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Too many threads allocated...\n", win_num, step, total_steps);
		  new_num_threads = idx_count;
		  //printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Lowering thread count to %d...\n", win_num, step, total_steps, new_num_threads);
		}
		else 
		{
		  start_idxs[idx_count] = end_idxs[idx_count-1];
		  end_idxs[idx_count] = F_count;
		  chunk_sizes[idx_count] = chunk_count;
		}
	  }
	  if(is_print)
	  {
		printf("---Indices for efficient flattening!---\n");
		for(int i = 0; i < new_num_threads; i++)
		{
		  printf("Start to End: [%d,%d) --> %d/%d terms in chunk, %d/%d reduced terms in chunk!\n", start_idxs[i], end_idxs[i], chunk_sizes[i], F_count, reduced_terms_per_chunk[i], num_terms_after_reduction);
		}
	  }
	  // checks to make sure that the above code worked
	  int sum_chunks = 0;
	  int sum_rt = 0;
	  for(int i = 0; i < new_num_threads; i++)
	  {
		sum_chunks += chunk_sizes[i];
		sum_rt += reduced_terms_per_chunk[i];
	  }
	  assert(sum_chunks == F_count);
	  assert(sum_rt == num_terms_after_reduction);
	  if(new_num_threads >= 2)
	  {
		assert(end_idxs[new_num_threads-1] == F_count);
		assert(start_idxs[new_num_threads-1] == end_idxs[new_num_threads-2]);
	  }
	  return new_num_threads;
	}

	void print_forwards_flag_array()
	{
	  for(int i = 0; i < F_count; i++)
	  {
		  if(F_counts[i] > 0)
		  {
			int* F = Fs[i];
			int num_combos = F_counts[i];
			printf("%d:", i);
			for(int j = 0; j < num_combos; j++)
				printf("%d,", F[j]);
			printf("\n");
		  }
	  }
	}

	~ForwardFlagArray()
	{
		for(int i = 0; i < F_count; i++)
		{
			if(F_counts[i] > 0)
				free(Fs[i]);
		}
		free(F_counts);
		free(Fs);
		free(chunk_sizes);
		free(reduced_terms_per_chunk);
	}
};

// Provides a chunked, packed storage method for the g and b tables
struct ChunkedPackedTableStorage
{
	GTABLE* chunked_gtables; // Gtables are arranged in chunks, partitioned tighly
	GTABLE* chunked_gtable_ps; // Parent Gtables are arranged in chunks, partitioned tighly
	BKEYS* chunked_btables; // Btables are arranged in chunks, partitioned tighly
	BKEYS* chunked_btable_ps; // Parent Btables are arranged in chunks, partitioned tighly
	uint page_limits[4]; // 0: gtables, 1: gtable_ps, 2: btables, 3: btable_ps
	uint current_page_idxs[4]; // the current index of the page pointer
	BYTE_COUNT_TYPE page_size_bytes; // size of each page
	BYTE_COUNT_TYPE page_size_gtable_num_elements; // number of GTABLE_TYPE elements which can fit into a page for the gtables
	BYTE_COUNT_TYPE page_size_btable_num_elements; // number of elements which can fit into a page for the btables
	BYTE_COUNT_TYPE* used_elems_per_page[4]; // gtables, gtable_ps, btables, btable_ps
	int mem_init_strategy; // 0: malloc, 1: calloc 2: page touching (after a malloc)
	int SYSTEM_PAGE_SIZE;

	void init(const uint pages_at_start, BYTE_COUNT_TYPE _page_size_bytes, const int _mem_init_strategy = 0, const int _SYSTEM_PAGE_SIZE = 4096)
	{
		page_size_bytes = _page_size_bytes; // number of bytes in the page allocated
		mem_init_strategy = _mem_init_strategy;
		SYSTEM_PAGE_SIZE = _SYSTEM_PAGE_SIZE;
		page_size_gtable_num_elements = page_size_bytes / sizeof(GTABLE_TYPE);
		page_size_btable_num_elements = page_size_bytes / sizeof(BKEYS_TYPE);
		//assert(page_size >= 1024*1024); // pages should be > 100KB, but not actually necessary
		for(int i = 0; i < 4; i++)
		{
			page_limits[i] = pages_at_start;
			used_elems_per_page[i] = (BYTE_COUNT_TYPE*) calloc(page_limits[i], sizeof(BYTE_COUNT_TYPE));
			if(pages_at_start > 0)
				null_ptr_check(used_elems_per_page[i]);
			current_page_idxs[i] = 0;			
		}
		chunked_gtables = (GTABLE*) malloc(pages_at_start * sizeof(GTABLE));
		if(pages_at_start > 0)
			null_dptr_check((void**)chunked_gtables);
		chunked_gtable_ps = (GTABLE*) malloc(pages_at_start * sizeof(GTABLE));
		if(pages_at_start > 0)
			null_dptr_check((void**)chunked_gtable_ps);
		chunked_btables = (BKEYS*) malloc(pages_at_start * sizeof(BKEYS));
		if(pages_at_start > 0)
			null_dptr_check((void**)chunked_btables);
		chunked_btable_ps = (BKEYS*) malloc(pages_at_start * sizeof(BKEYS));
		if(pages_at_start > 0)
			null_dptr_check((void**)chunked_btable_ps);
		for(uint page_idx = 0; page_idx < pages_at_start; page_idx++)
		{
			chunked_gtable_ps[page_idx] = (GTABLE) page_alloc();
			null_ptr_check(chunked_gtable_ps[page_idx]);
			chunked_gtables[page_idx] = (GTABLE) page_alloc();
			null_ptr_check(chunked_gtables[page_idx]);
			chunked_btables[page_idx] = (BKEYS) page_alloc();
			null_ptr_check(chunked_btables[page_idx]);
			chunked_btable_ps[page_idx] = (BKEYS) page_alloc();
			null_ptr_check(chunked_btable_ps[page_idx]);
		}
	}

	void* page_alloc()
	{
		switch (mem_init_strategy)
		{
			case 0: // Malloc
			{
				return malloc(page_size_bytes);
			}
			case 1: // Calloc
			{
				return calloc(page_size_bytes, 1);
			}
			case 2: // System Page Touching
			{
				uint8_t* buf = (uint8_t*) malloc(page_size_bytes);
				for(BYTE_COUNT_TYPE i = 0; i < page_size_bytes; i+= SYSTEM_PAGE_SIZE)
					buf[i] = 0;
				return buf;
			}
			default:
			{
				printf("PAGE ALLOC METHOD NOT IMPLEMENTED!\n");
				exit(1);
			}
		}
	}

	// extends the gtable pages and the btable pages
	// Computed as table_bytes = max_cells_cen_shape * GTABLE_MULTIPLIER * terms_of_shape_after_reduction * sizeof(GTABLE_TYPE)
	// max_cells_cen_shape is brought in as s_c(m,d) / (1 + HALF_STORAGE)
	void extend_gtables(BYTE_COUNT_TYPE max_cells_cen_shape, BYTE_COUNT_TYPE terms_in_shape_after_reduction)
	{
		BYTE_COUNT_TYPE bytes_table = ((BYTE_COUNT_TYPE) (GTABLE_SIZE_MULTIPLIER * max_cells_cen_shape)) * sizeof(GTABLE_TYPE);
		BYTE_COUNT_TYPE req_table_bytes = bytes_table * terms_in_shape_after_reduction;

		// find how many bytes are left in the current gtable page
		const uint page_idx = current_page_idxs[0];
		BYTE_COUNT_TYPE bytes_remaining; // in current gtable page
		if(page_idx == page_limits[0])
			bytes_remaining = 0;
		else
		{
			bytes_remaining = page_size_bytes - (used_elems_per_page[0][page_idx] * sizeof(GTABLE_TYPE));
			// find how many bytes are left in the available pages
			BYTE_COUNT_TYPE bytes_upper = (page_limits[0] - (page_idx+1)) * page_size_bytes;
			bytes_remaining += bytes_upper;
		}
		// only add additional pages if the amount of memory is less than required
		if(req_table_bytes < bytes_remaining)
			return;
		// find max number of pages to add 
		uint additional_pages = (req_table_bytes - bytes_remaining + page_size_bytes - 1) / page_size_bytes;
		uint num_new_total_pages = additional_pages + page_limits[0];
		// extend page limits by additional pages 
		chunked_gtables = (GTABLE*) realloc(chunked_gtables, num_new_total_pages * sizeof(GTABLE));
		null_dptr_check((void**)chunked_gtables);
		used_elems_per_page[0] = (BYTE_COUNT_TYPE*) realloc(used_elems_per_page[0], num_new_total_pages * sizeof(BYTE_COUNT_TYPE));
		null_ptr_check(used_elems_per_page[0]);
		
		for(uint new_page_idx = page_limits[0]; new_page_idx < num_new_total_pages; new_page_idx++)
		{
			chunked_gtables[new_page_idx] = (GTABLE) page_alloc();
			null_ptr_check(chunked_gtables[new_page_idx]);
			used_elems_per_page[0][new_page_idx] = 0;
		}
		page_limits[0] = num_new_total_pages;
	}
	
	// extends the btable pages and the btable pages
	// Computed as btable_bytes = max_cells_cen_shape * terms_of_shape_after_reduction * sizeof(BKEYS_TYPE)
	// max_cells_cen_shape is brought in as s_c(m,d) / (1 + HALF_STORAGE)
	void extend_btables(BYTE_COUNT_TYPE max_cells_cen_shape, BYTE_COUNT_TYPE terms_in_shape_after_reduction)
	{
		BYTE_COUNT_TYPE bytes_table = (BYTE_COUNT_TYPE)(max_cells_cen_shape * sizeof(BKEYS_TYPE) );
		BYTE_COUNT_TYPE req_table_bytes = bytes_table * terms_in_shape_after_reduction;

		// find how many bytes are left in the current btable page
		const uint page_idx = current_page_idxs[2];
		BYTE_COUNT_TYPE bytes_remaining;
		if(page_idx == page_limits[2])
			bytes_remaining = 0;
		else
		{
			bytes_remaining = page_size_bytes - (used_elems_per_page[2][page_idx] * sizeof(BKEYS_TYPE));
			// find how many bytes are left in the available pages
			BYTE_COUNT_TYPE bytes_upper = (page_limits[2] - (page_idx+1)) * page_size_bytes;
			bytes_remaining += bytes_upper;
		}
		// only add additional pages if the amount of memory is less than required
		if(req_table_bytes < bytes_remaining)
			return;
		// find max number of pages to add 
		uint additional_pages = (req_table_bytes - bytes_remaining + page_size_bytes - 1) / page_size_bytes;
		uint num_new_total_pages = additional_pages + page_limits[2];
		// extend page limits by additional pages 
		chunked_btables = (BKEYS*) realloc(chunked_btables, num_new_total_pages * sizeof(BKEYS));
		null_dptr_check((void**)chunked_btables);
		used_elems_per_page[2] = (BYTE_COUNT_TYPE*) realloc(used_elems_per_page[2], num_new_total_pages * sizeof(BYTE_COUNT_TYPE));
		null_ptr_check(used_elems_per_page[2]);
		for(uint new_page_idx = page_limits[2]; new_page_idx < num_new_total_pages; new_page_idx++)
		{
			chunked_btables[new_page_idx] = (BKEYS) page_alloc();
			null_ptr_check(chunked_btables[new_page_idx]);
			used_elems_per_page[2][new_page_idx] = 0;
		}
		page_limits[2] = num_new_total_pages;
	}

	// extends the btable pages and the btable pages
	// Compute req_btable_bytes = HALF_STORE / 2 * s_c(m, d) * terms * sizeof(BKEYS_TYPE)
	void extend_bp_tables(BYTE_COUNT_TYPE req_table_bytes)
	{
		// find how many bytes are left in the current btable page
		const uint page_idx = current_page_idxs[3];
		BYTE_COUNT_TYPE bytes_remaining;
		if(page_idx == page_limits[3])
			bytes_remaining = 0;
		else
		{
			bytes_remaining = page_size_bytes - (used_elems_per_page[3][page_idx] * sizeof(BKEYS_TYPE));
			// find how many bytes are left in the available pages
			BYTE_COUNT_TYPE bytes_upper = (page_limits[3] - (page_idx+1)) * page_size_bytes;
			bytes_remaining += bytes_upper;
		}
		// only add additional pages if the amount of memory is less than required
		if(req_table_bytes < bytes_remaining)
			return;
		// find max number of pages to add 
		uint additional_pages = (req_table_bytes - bytes_remaining + page_size_bytes - 1) / page_size_bytes;
		uint num_new_total_pages = additional_pages + page_limits[3];
		// extend page limits by additional pages 
		chunked_btable_ps = (BKEYS*) realloc(chunked_btable_ps, num_new_total_pages * sizeof(BKEYS));
		null_dptr_check((void**)chunked_btable_ps);
		used_elems_per_page[3] = (BYTE_COUNT_TYPE*) realloc(used_elems_per_page[3], num_new_total_pages * sizeof(BYTE_COUNT_TYPE));
		null_ptr_check(used_elems_per_page[3]);
		for(uint new_page_idx = page_limits[3]; new_page_idx < num_new_total_pages; new_page_idx++)
		{
			chunked_btable_ps[new_page_idx] = (BKEYS) page_alloc();
			null_ptr_check(chunked_btable_ps[new_page_idx]);
			used_elems_per_page[3][new_page_idx] = 0;
		}
		page_limits[3] = num_new_total_pages;
	}

	// This function assumes that there is no way to increment over the number of allocated pages
	void set_term_gtable_pointer(GTABLE* gtable, int cells_gtable, const bool mem_ptr_incr = true)
	{
		BYTE_COUNT_TYPE elems_gtable = cells_gtable * GTABLE_SIZE_MULTIPLIER;
		uint page_idx = current_page_idxs[0];
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[0][page_idx];
		if( (elems_used_in_page + elems_gtable) > page_size_gtable_num_elements )
		{
			page_idx = ++current_page_idxs[0];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			if(page_idx == page_limits[0])
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN set_term_gtable_pointer:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
						   YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
						   NC "\n");
				}
				BYTE_COUNT_TYPE gtable_type_size = GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE);
				BYTE_COUNT_TYPE added_elems = (page_size_bytes + gtable_type_size -1) / gtable_type_size;
				extend_gtables(added_elems, 3); // Enlarge memory range by 3 page sizes.
			}
		}
		*gtable = chunked_gtables[page_idx] + elems_used_in_page;
		if(mem_ptr_incr)
			used_elems_per_page[0][page_idx] += elems_gtable;
	}

	// This function assumes that there is no way to increment over the number of allocated pages
	void set_term_btable_pointer(BKEYS* btable, int cells_gtable, const bool mem_ptr_incr = true)
	{
		uint page_idx = current_page_idxs[2];
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[2][page_idx];
		if( (elems_used_in_page + cells_gtable) > page_size_btable_num_elements )
		{
			page_idx = ++current_page_idxs[2];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			if(page_idx == page_limits[2])
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN set_term_btable_pointer:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
						   YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
						   NC "\n");
				}
				BYTE_COUNT_TYPE added_elems = (page_size_bytes + sizeof(BKEYS_TYPE) -1) / sizeof(BKEYS_TYPE);
				extend_btables(added_elems, 3); // Enlarge memory range by 3 page sizes.
			}
		}
		*btable = chunked_btables[page_idx] + elems_used_in_page;
		if(mem_ptr_incr)
			used_elems_per_page[2][page_idx] += cells_gtable;
	}
	
	// This function assumes that there is no way to increment over the number of allocated pages
	void set_term_bp_table_pointer(BKEYS* bp_table, int cells_gtable, const bool mem_ptr_incr = true)
	{
		uint page_idx = current_page_idxs[3];
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[3][page_idx];
		if( (elems_used_in_page + cells_gtable) > page_size_btable_num_elements )
		{
			page_idx = ++current_page_idxs[3];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			if(page_idx == page_limits[3])
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN set_term_bp_table_pointer:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
							YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
							NC "\n");
				}
				extend_bp_tables(3 * page_size_bytes); // Enlarge memory range by 3 page sizes.
			}
		}
		*bp_table = chunked_btable_ps[page_idx] + elems_used_in_page;
		if(mem_ptr_incr)
			used_elems_per_page[3][page_idx] += cells_gtable;
	}
	
	GTABLE get_term_gtable_pointer()
	{
		int cur_gtable_page_idx = current_page_idxs[0];
		return chunked_gtables[cur_gtable_page_idx] + used_elems_per_page[0][cur_gtable_page_idx];
	}

	BKEYS get_term_btable_pointer()
	{
		int cur_btable_page_idx = current_page_idxs[2];
		return chunked_btables[cur_btable_page_idx] + used_elems_per_page[2][cur_btable_page_idx];
	}
	
	// only call this after calling set_term_gtable_pointer with mem_ptr_incr as false
	void incr_chunked_gtable_ptr(int cells_gtable)
	{
		BYTE_COUNT_TYPE elems_gtable = cells_gtable * GTABLE_SIZE_MULTIPLIER;
		used_elems_per_page[0][current_page_idxs[0]] += elems_gtable;
	}

	// only call this after calling set_term_btable_pointer with mem_ptr_incr as false
	void incr_chunked_btable_ptr(int cells_gtable)
	{
		used_elems_per_page[2][current_page_idxs[2]] += cells_gtable;
	}
	
	// only call this after calling set_term_bp_table_pointer with mem_ptr_incr as false
	void incr_chunked_bp_table_ptr(int cells_gtable)
	{
		used_elems_per_page[3][current_page_idxs[3]] += cells_gtable;
	}
	
	// called at the end of the term reduction
	// the gtable memory area now becomes the gtable_p memory area
	// the btable memory area now becomes the btable_p memory area
	// the page info for gtable is swapped to gtable_p page info
	// the page info for btable is swapped to btable_p page info
	// the gtable memory area is cleared (if WITH_GB_TABLE_REALLOC is set to true)
	// the btable memory area is cleared (if WITH_GB_TABLE_REALLOC is set to true)
	void swap_gtables()
	{
		// Swap and (possibly) clear the gtables
		ptr_swap<GTABLE>(&chunked_gtables, &chunked_gtable_ps);
		ptr_swap<BYTE_COUNT_TYPE>( &(used_elems_per_page[1]), &(used_elems_per_page[0]) );
		val_swap<uint>( &(page_limits[1]), &(page_limits[0]) );
		val_swap<uint>( &(current_page_idxs[1]), &(current_page_idxs[0]) );
		// Now that the gtables memory areas are referenced by chunked_gtable_ps ...
		// ... free the "gtable" (old gtables_p) memory area to save memory
		if(WITH_GB_TABLE_REALLOC)
		{
			for(uint page_idx = 0; page_idx < page_limits[0]; page_idx++)
				free(chunked_gtables[page_idx]);
			current_page_idxs[0] = 0;
			used_elems_per_page[0] = (BYTE_COUNT_TYPE*)realloc(used_elems_per_page[0], sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page[0]);
		  	used_elems_per_page[0][0] = 0;
			page_limits[0] = 0;

			// Now shrink gtable_ps to its minimally required size
			for(uint page_idx = current_page_idxs[1]+1; page_idx < page_limits[1]; page_idx++)
				free(chunked_gtable_ps[page_idx]);
		  	page_limits[1] = current_page_idxs[1]+1;
		  	used_elems_per_page[1] = (BYTE_COUNT_TYPE*)realloc(used_elems_per_page[1], page_limits[1]*sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page[1]);
			// keep current_page_idxs[1] as is
		}
		// Otherwise, keep the allocated memory area and simply reset the counters and memory counts
		else
		{
			current_page_idxs[0] = 0;
			memset(used_elems_per_page[0], 0, page_limits[0] * sizeof(BYTE_COUNT_TYPE));
			// In this case the page_limit is not reset
		}

	}

	void swap_btables()
	{
		// Swap and (possibly) clear the btables
		ptr_swap<BKEYS>(&chunked_btables, &chunked_btable_ps);
		ptr_swap<BYTE_COUNT_TYPE>( &(used_elems_per_page[3]), &(used_elems_per_page[2]) );
		val_swap<uint>( &(page_limits[3]), &(page_limits[2]) );
		val_swap<uint>( &(current_page_idxs[3]), &(current_page_idxs[2]) );
		// Now that the btables memory areas are referenced by btable_ps ...
		// ... free the "btable_ps" (old btables) memory area to save memory
		if(WITH_GB_TABLE_REALLOC)
		{
			for(uint page_idx = 0; page_idx < page_limits[3]; page_idx++)
				free(chunked_btable_ps[page_idx]);
			current_page_idxs[3] = 0;
			used_elems_per_page[3] = (BYTE_COUNT_TYPE*)realloc(used_elems_per_page[3], sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page[3]);
			used_elems_per_page[3][0] = 0;
			page_limits[3] = 0;

			// Now shrink btables to its minimally required size
			for(uint page_idx = current_page_idxs[2]+1; page_idx < page_limits[2]; page_idx++)
				free(chunked_btables[page_idx]);
		  	page_limits[2] = current_page_idxs[2]+1;
		  	used_elems_per_page[2] = (BYTE_COUNT_TYPE*)realloc(used_elems_per_page[2], page_limits[2]*sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page[2]);
			// keep current_page_idxs[3] as is
		}
		// Otherwise, keep the allocated memory area and simply reset the counters and memory counts
		else
		{
			current_page_idxs[3] = 0;
			memset(used_elems_per_page[3], 0, page_limits[3] * sizeof(BYTE_COUNT_TYPE));
			// In this case the page_limit is not reset
		}
	}

	void deinit()
	{
		// free chunked gtables
		for(uint page_idx = 0; page_idx < page_limits[0]; page_idx++)
			free(chunked_gtables[page_idx]);
		free(chunked_gtables);
		free(used_elems_per_page[0]);
		page_limits[0] = 0;
		// free chunked gtable_ps
		for(uint page_idx = 0; page_idx < page_limits[1]; page_idx++)
			free(chunked_gtable_ps[page_idx]);
		free(chunked_gtable_ps);
		free(used_elems_per_page[1]);
		page_limits[1] = 0;
		// free chunked btables
		for(uint page_idx = 0; page_idx < page_limits[2]; page_idx++)
			free(chunked_btables[page_idx]);
		free(chunked_btables);
		free(used_elems_per_page[2]);
		page_limits[2] = 0;
		// free chunked btable_ps
		for(uint page_idx = 0; page_idx < page_limits[3]; page_idx++)
			free(chunked_btable_ps[page_idx]);
		free(chunked_btable_ps);
		free(used_elems_per_page[3]);
		page_limits[3] = 0;
	}

};

// Provides a chunked, packed storage method for the A/p/q/b elements, along with the coalign (sign) maps
template<typename T>
struct ChunkedPackedElement
{
	T** chunked_elems;
	uint page_limit;
	uint current_page_idx;
	BYTE_COUNT_TYPE* used_elems_per_page;
	BYTE_COUNT_TYPE page_size_bytes; // size of each page
	BYTE_COUNT_TYPE page_size_num_elements;
	int mem_init_strategy; // 0: malloc, 1: calloc 2: page touching (after a malloc)
	int SYSTEM_PAGE_SIZE;

	void init(const uint pages_at_start, BYTE_COUNT_TYPE _page_size_bytes, const int _mem_init_strategy = 0, const int _SYSTEM_PAGE_SIZE = 4096)
	{
		page_size_bytes = _page_size_bytes; // number of bytes in the page allocated
		mem_init_strategy = _mem_init_strategy;
		SYSTEM_PAGE_SIZE = _SYSTEM_PAGE_SIZE;
		page_size_num_elements = page_size_bytes / sizeof(T);

		page_limit = pages_at_start;
		used_elems_per_page = (BYTE_COUNT_TYPE*) calloc(page_limit, sizeof(BYTE_COUNT_TYPE));
		if(page_limit > 0)
			null_ptr_check(used_elems_per_page);
		current_page_idx = 0;
		chunked_elems = (T**) malloc(pages_at_start * sizeof(T*));
		if(page_limit > 0)
			null_dptr_check((void**)chunked_elems);
		for(uint page_idx = 0; page_idx < pages_at_start; page_idx++)
		{
			chunked_elems[page_idx] = (T*) page_alloc();	
			null_ptr_check(chunked_elems[page_idx]);
		}
	}

	void* page_alloc()
	{
		switch (mem_init_strategy)
		{
			case 0: // Malloc
			{
				return malloc(page_size_bytes);
			}
			case 1: // Calloc
			{
				return calloc(page_size_bytes, 1);
			}
			case 2: // System Page Touching
			{
				uint8_t* buf = (uint8_t*) malloc(page_size_bytes);
				for(BYTE_COUNT_TYPE i = 0; i < page_size_bytes; i+= SYSTEM_PAGE_SIZE)
					buf[i] = 0;
				return buf;
			}
			default:
			{
				printf("PAGE ALLOC METHOD NOT IMPLEMENTED!\n");
				exit(1);
			}
		}
	}

	// extends the elem pages
	void extend_elems(BYTE_COUNT_TYPE req_bytes)
	{
		// find how many bytes are left in the current page
		const uint page_idx = current_page_idx;
		BYTE_COUNT_TYPE bytes_remaining; // in current elem page
		if(page_idx == page_limit)
			bytes_remaining = 0;
		else
		{
			bytes_remaining = page_size_bytes - (used_elems_per_page[page_idx] * sizeof(T));
			// find how many bytes are left in the available pages
			BYTE_COUNT_TYPE bytes_upper = (page_limit - (page_idx+1)) * page_size_bytes;
			bytes_remaining += bytes_upper;
		}
		// only add additional pages if the amount of memory is less than required
		if(req_bytes < bytes_remaining)
			return;
		// find max number of pages to add 
		uint additional_pages = (req_bytes - bytes_remaining + page_size_bytes - 1) / page_size_bytes;
		uint num_new_total_pages = additional_pages + page_limit;
		// extend page limits by additional pages 
		chunked_elems = (T**) realloc(chunked_elems, num_new_total_pages * sizeof(T*));
		null_dptr_check((void**)chunked_elems);
		used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(used_elems_per_page, num_new_total_pages * sizeof(BYTE_COUNT_TYPE));
		null_ptr_check(used_elems_per_page);
		
		for(uint new_page_idx = page_limit; new_page_idx < num_new_total_pages; new_page_idx++)
		{
			chunked_elems[new_page_idx] = (T*) page_alloc();
			null_ptr_check(chunked_elems[new_page_idx]);
			used_elems_per_page[new_page_idx] = 0;
		}
		page_limit = num_new_total_pages;
	}

	void copy_then_set_elem_ptr(T** elem, BYTE_COUNT_TYPE num_elems)
	{
		// Set A ptr
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[current_page_idx];
		if( (elems_used_in_page + num_elems) > page_size_num_elements )
		{
			current_page_idx++;
			elems_used_in_page = 0;
			//assert(current_page_idx < page_limit); // REMOVE
			if(current_page_idx == page_limit)
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN ChunkedPackedElement:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
						   YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
						   NC "\n");
				}
				extend_elems(3 * page_size_bytes); // Enlarge memory range by 3 page sizes.
			}
		}
		T* elem_addr = chunked_elems[current_page_idx] + elems_used_in_page;
		memcpy(elem_addr, *elem, num_elems * sizeof(T));
		*elem = elem_addr;
		used_elems_per_page[current_page_idx] += num_elems;
	}

	void set_elem_ptr_then_copy_vector(T** elem, T* vector, BYTE_COUNT_TYPE num_elems)
	{
		// Set A ptr
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[current_page_idx];
		if( (elems_used_in_page + num_elems) > page_size_num_elements )
		{
			current_page_idx++;
			elems_used_in_page = 0;
			//assert(current_page_idx < page_limit); // REMOVE
			if(current_page_idx == page_limit)
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN ChunkedPackedElement:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
						   YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
						   NC "\n");
				}
				extend_elems(3 * page_size_bytes); // Enlarge memory range by 3 page sizes.
			}
		}
		T* elem_addr = chunked_elems[current_page_idx] + elems_used_in_page;
		*elem = elem_addr;
		memcpy(elem_addr, vector, num_elems * sizeof(T));
		used_elems_per_page[current_page_idx] += num_elems;
	}

	void set_elem_ptr(T** elem, BYTE_COUNT_TYPE num_elems)
	{
		// Set A ptr
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[current_page_idx];
		if( (elems_used_in_page + num_elems) > page_size_num_elements )
		{
			current_page_idx++;
			elems_used_in_page = 0;
			//assert(current_page_idx < page_limit); // REMOVE
			if(current_page_idx == page_limit)
			{
				if(WITH_WARNINGS)
				{
					printf(YEL "[WARN ChunkedPackedElement:]\n  Allocated memory after extending memory range was found to be insufficient.\n"
						   YEL "  Consider increasing CP_STORAGE_PAGE_SIZE in cauchy_types.hpp! Extending Memory Range Again!"
						   NC "\n");
				}
				extend_elems(3 * page_size_bytes); // Enlarge memory range by 3 page sizes.
			}
		}
		T* elem_addr = chunked_elems[current_page_idx] + elems_used_in_page;
		*elem = elem_addr;
		used_elems_per_page[current_page_idx] += num_elems;
	}

	void pop_elem_ptr(BYTE_COUNT_TYPE num_elems)
	{
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[current_page_idx];
		if(num_elems > elems_used_in_page)
		{
			used_elems_per_page[current_page_idx] = 0;
			assert(current_page_idx > 0);
			current_page_idx--;
			assert( used_elems_per_page[current_page_idx] >= (num_elems-elems_used_in_page) );
			used_elems_per_page[current_page_idx] -= (num_elems-elems_used_in_page);
		}
		else 
		{
			used_elems_per_page[current_page_idx] -= num_elems;
		}
	}

	void unallocate_unused_space()
	{
		for(uint i = current_page_idx+1; i < page_limit; i++)
			free(chunked_elems[i]);
		//assert(page_limit > 0);
		uint used_page_idxs = (page_limit == 0) ? 0 : page_limit - 1;
		if(current_page_idx < used_page_idxs )
		{
			page_limit = current_page_idx + 1;
			chunked_elems = (T**) realloc(chunked_elems, page_limit * sizeof(T*));
			null_dptr_check((void**)chunked_elems);
			used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(used_elems_per_page, page_limit * sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page);
		}
	}

	void reset()
	{
		if(WITH_COALIGN_REALLOC)
		{
			for(uint page_idx = 0; page_idx < page_limit; page_idx++)
				free(chunked_elems[page_idx]);
			current_page_idx = 0;
			used_elems_per_page = (BYTE_COUNT_TYPE*)realloc(used_elems_per_page, sizeof(BYTE_COUNT_TYPE));
			null_ptr_check(used_elems_per_page);
			chunked_elems = (T**) realloc(chunked_elems, sizeof(T*));
			null_dptr_check((void**)chunked_elems);
		  	used_elems_per_page[0] = 0;
			page_limit = 0;
		}
		else 
		{
			current_page_idx = 0;
			memset(used_elems_per_page, 0, page_limit * sizeof(BYTE_COUNT_TYPE));
			// In this case the page_limit is not reset
		}
	}

	void reset_page_idxs()
	{
		current_page_idx = 0;
		memset(used_elems_per_page, 0, page_limit * sizeof(BYTE_COUNT_TYPE));
	}

	BYTE_COUNT_TYPE get_total_byte_count()
	{
		BYTE_COUNT_TYPE bytes_total = 0;
		for(uint i = 0; i < page_limit; i++)
			bytes_total += used_elems_per_page[i] * sizeof(T);
		return bytes_total;
	}

	void deinit()
	{
		for(uint page_idx = 0; page_idx < page_limit; page_idx++)
			free(chunked_elems[page_idx]);
		free(chunked_elems);
		free(used_elems_per_page);
		page_limit = 0;
	}

};

struct CoalignmentElemStorage
{
	ChunkedPackedElement<double> chunked_As;
	ChunkedPackedElement<double> chunked_ps;
	ChunkedPackedElement<double> chunked_qs;
	ChunkedPackedElement<double> chunked_bs;
	ChunkedPackedElement<uint8_t> chunked_c_maps;
	ChunkedPackedElement<int8_t> chunked_cs_maps;


	void init(const uint pages_at_start, BYTE_COUNT_TYPE page_size_bytes, int mem_init_strategy = 0)
	{
		chunked_As.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_ps.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_qs.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_bs.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_c_maps.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_cs_maps.init(pages_at_start, page_size_bytes, mem_init_strategy);
	}

	void extend_storage(BYTE_COUNT_TYPE ps_bytes, BYTE_COUNT_TYPE bs_bytes, const int d)
	{
		chunked_As.extend_elems(ps_bytes * d);
		chunked_ps.extend_elems(ps_bytes);
		chunked_qs.extend_elems(ps_bytes);
		chunked_bs.extend_elems(bs_bytes);
		chunked_c_maps.extend_elems(bs_bytes / d);
		chunked_cs_maps.extend_elems(bs_bytes / d);
	}

	void set_term_ptrs(CauchyTerm* term, int m_precoalign)
	{
		BYTE_COUNT_TYPE m = term->m;
		BYTE_COUNT_TYPE d = term->d;
		chunked_As.copy_then_set_elem_ptr(&(term->A), m*d);
		chunked_ps.copy_then_set_elem_ptr(&(term->p), m);
		chunked_qs.copy_then_set_elem_ptr(&(term->q), m);
		chunked_bs.copy_then_set_elem_ptr(&(term->b), d);
		if(term->m < m_precoalign)
		{
			chunked_c_maps.copy_then_set_elem_ptr(&(term->c_map), m_precoalign);
			chunked_cs_maps.copy_then_set_elem_ptr(&(term->cs_map), m_precoalign);
		}
		else 
		{
			term->c_map=NULL;
			term->cs_map=NULL;
		}
	}

	void unallocate_unused_space()
	{
		chunked_As.unallocate_unused_space();
		chunked_ps.unallocate_unused_space();
		chunked_qs.unallocate_unused_space();
		chunked_bs.unallocate_unused_space();
		chunked_c_maps.unallocate_unused_space();
		chunked_cs_maps.unallocate_unused_space();
	}

	void reset()
	{
		chunked_As.reset();
		chunked_ps.reset();
		chunked_qs.reset();
		chunked_bs.reset();
		chunked_c_maps.reset();
		chunked_cs_maps.reset();
	}

	void reset_page_idxs()
	{
		chunked_As.reset_page_idxs();
		chunked_ps.reset_page_idxs();
		chunked_qs.reset_page_idxs();
		chunked_bs.reset_page_idxs();
		chunked_c_maps.reset_page_idxs();
		chunked_cs_maps.reset_page_idxs();
	}

	BYTE_COUNT_TYPE get_total_byte_count(bool with_print = false)
	{
		BYTE_COUNT_TYPE bytes_total = chunked_As.get_total_byte_count() + 
			chunked_ps.get_total_byte_count() + chunked_qs.get_total_byte_count() + 
			chunked_bs.get_total_byte_count() + chunked_c_maps.get_total_byte_count() +
			chunked_cs_maps.get_total_byte_count();
		if(with_print)
		{
			printf("-Coalign Element Storage: Memory Usage Breakdown:\n");
			printf("  As use: %.3lf MBs\n", ((double)chunked_As.get_total_byte_count()) / (1024*1024) );
			printf("  ps/qs each use: %.3lf MBs\n", ((double)chunked_ps.get_total_byte_count()) / (1024*1024) );
			printf("  bs use: %.3lf MBs\n", ((double)chunked_bs.get_total_byte_count()) / (1024*1024) );
			printf("  c_maps/cs_maps each use: %.3lf MBs\n", ((double)chunked_c_maps.get_total_byte_count()) / (1024*1024) );
			printf("  *Total: %.3lf MBs\n", ((double)bytes_total) / (1024*1024));
		}
		return bytes_total;
	}

	BYTE_COUNT_TYPE get_total_byte_count(BYTE_COUNT_TYPE* As_bytes, BYTE_COUNT_TYPE* ps_bytes, BYTE_COUNT_TYPE* bs_bytes, BYTE_COUNT_TYPE* c_maps_bytes)
	{
		BYTE_COUNT_TYPE a_bytes = chunked_As.get_total_byte_count();
		BYTE_COUNT_TYPE p_bytes = chunked_ps.get_total_byte_count();
		BYTE_COUNT_TYPE b_bytes = chunked_bs.get_total_byte_count();
		BYTE_COUNT_TYPE cm_bytes = chunked_c_maps.get_total_byte_count();
		*As_bytes += a_bytes;
		*ps_bytes += p_bytes;
		*bs_bytes += b_bytes;
		*c_maps_bytes += cm_bytes;
		return a_bytes + 2*p_bytes + b_bytes + 2*cm_bytes;
	}


	void deinit()
	{
		chunked_As.deinit();
		chunked_ps.deinit();
		chunked_qs.deinit();
		chunked_bs.deinit();
		chunked_c_maps.deinit();
		chunked_cs_maps.deinit();
	}

};

struct ReductionElemStorage
{
	ChunkedPackedElement<double> chunked_As;
	ChunkedPackedElement<double> chunked_ps;
	ChunkedPackedElement<double> chunked_bs;

	void init(const uint pages_at_start, BYTE_COUNT_TYPE page_size_bytes, int mem_init_strategy = 0)
	{
		chunked_As.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_ps.init(pages_at_start, page_size_bytes, mem_init_strategy);
		chunked_bs.init(pages_at_start, page_size_bytes, mem_init_strategy);
	}

	void extend_storage(BYTE_COUNT_TYPE ps_bytes, BYTE_COUNT_TYPE bs_bytes, const int d)
	{
		chunked_As.extend_elems(ps_bytes * d);
		chunked_ps.extend_elems(ps_bytes);
		chunked_bs.extend_elems(bs_bytes);
	}

	void set_term_ptrs(CauchyTerm* term)
	{
		BYTE_COUNT_TYPE m = term->m;
		BYTE_COUNT_TYPE d = term->d;
		chunked_As.copy_then_set_elem_ptr(&(term->A), m*d);
		chunked_ps.copy_then_set_elem_ptr(&(term->p), m);
		chunked_bs.copy_then_set_elem_ptr(&(term->b), d);
	}

	void unallocate_unused_space()
	{
		chunked_As.unallocate_unused_space();
		chunked_ps.unallocate_unused_space();
		chunked_bs.unallocate_unused_space();
	}

	void reset()
	{
		chunked_As.reset();
		chunked_ps.reset();
		chunked_bs.reset();
	}

	void reset_page_idxs()
	{
		chunked_As.reset_page_idxs();
		chunked_ps.reset_page_idxs();
		chunked_bs.reset_page_idxs();
	}

	BYTE_COUNT_TYPE get_total_byte_count(bool with_print = false)
	{
		BYTE_COUNT_TYPE bytes_total = chunked_As.get_total_byte_count() + 
			chunked_ps.get_total_byte_count() + chunked_bs.get_total_byte_count();
		if(with_print)
		{
			printf("-Reduction Element Storage: Memory Usage Breakdown:\n");
			printf("  As use: %.3lf MBs\n", ((double)chunked_As.get_total_byte_count()) / (1024*1024) );
			printf("  ps use: %.3lf MBs\n", ((double)chunked_ps.get_total_byte_count()) / (1024*1024) );
			printf("  bs use: %.3lf MBs\n", ((double)chunked_bs.get_total_byte_count()) / (1024*1024) );
			printf("  *Total: %.3lf MBs\n", ((double)bytes_total) / (1024*1024));
		}
		return bytes_total;
	}

	BYTE_COUNT_TYPE get_total_byte_count(BYTE_COUNT_TYPE* As_bytes, BYTE_COUNT_TYPE* ps_bytes, BYTE_COUNT_TYPE* bs_bytes)
	{
		BYTE_COUNT_TYPE a_bytes = chunked_As.get_total_byte_count();
		BYTE_COUNT_TYPE p_bytes = chunked_ps.get_total_byte_count();
		BYTE_COUNT_TYPE b_bytes = chunked_bs.get_total_byte_count();
		*As_bytes += a_bytes;
		*ps_bytes += p_bytes;
		*bs_bytes += b_bytes;
		return a_bytes + p_bytes + b_bytes;
	}

	void deinit()
	{
		chunked_As.deinit();
		chunked_ps.deinit();
		chunked_bs.deinit();
	}

};

// Cauchy Stats -- Used to determine different statistics as the estimator runs
struct CauchyStats
{

	BYTE_COUNT_TYPE get_table_memory_usage(ChunkedPackedTableStorage* gb_tables, int max_threads, bool with_print)
	{

		BYTE_COUNT_TYPE gtables_bytes = 0;
		BYTE_COUNT_TYPE gtable_ps_bytes = 0;
		BYTE_COUNT_TYPE btables_bytes = 0;
		BYTE_COUNT_TYPE btable_ps_bytes = 0;
		BYTE_COUNT_TYPE page_size_mbs = gb_tables[0].page_size_bytes / (1024*1024);

		uint num_gtable_pages = 0;
		uint num_gtable_p_pages = 0;
		uint num_btable_pages = 0;
		uint num_btable_p_pages = 0;

		for(int tid = 0; tid < max_threads; tid++)
		{
			uint* current_page_idxs = gb_tables[tid].current_page_idxs;
			BYTE_COUNT_TYPE** used_elems_per_page = gb_tables[tid].used_elems_per_page;
			uint* page_limits = gb_tables[tid].page_limits;
			if(page_limits[0] > 0)
				for(uint i = 0; i <= current_page_idxs[0]; i++)
					gtables_bytes += used_elems_per_page[0][i] * sizeof(GTABLE_TYPE);
			if(page_limits[1] > 0)
				for(uint i = 0; i <= current_page_idxs[1]; i++)
					gtable_ps_bytes += used_elems_per_page[1][i] * sizeof(GTABLE_TYPE);
			if(page_limits[2] > 0)
				for(uint i = 0; i <= current_page_idxs[2]; i++)
					btables_bytes += used_elems_per_page[2][i] * sizeof(BKEYS_TYPE);
			if(page_limits[3] > 0)
				for(uint i = 0; i <= current_page_idxs[3]; i++)
					btable_ps_bytes += used_elems_per_page[3][i] * sizeof(BKEYS_TYPE);		
			num_gtable_pages += page_limits[0];
			num_gtable_p_pages += page_limits[1];
			num_btable_pages += page_limits[2];
			num_btable_p_pages += page_limits[3];
		}
		if(with_print)
		{
			printf("-Gtables, Gtable_ps, Btables, Btable_ps Memory Usage Breakdown:\n");
			printf("  Gtables use: %d x %llu MBs/page. In Use: %.3lf MBs\n", num_gtable_pages, page_size_mbs, ((double)gtables_bytes) / (1024*1024));
			printf("  Gtable_ps use: %d x %llu MBs/page. In Use: %.3lf MBs\n", num_gtable_p_pages, page_size_mbs, ((double)gtable_ps_bytes) / (1024*1024));
			printf("  Btables use: %d x %llu MBs/page. In Use: %.3lf MBs\n", num_btable_pages, page_size_mbs, ((double)btables_bytes) / (1024*1024));
			printf("  Btable_ps use: %d x %llu MBs/page. In Use: %.3lf MBs\n", num_btable_p_pages, page_size_mbs, ((double)btable_ps_bytes) / (1024*1024));
			printf("  *Total Table Memory: %.3lf MBs ---\n", ((double)(gtables_bytes + gtable_ps_bytes + btables_bytes + btable_ps_bytes)) / (1024*1024));
		}
		return gtables_bytes + gtable_ps_bytes + btables_bytes + btable_ps_bytes;
	}

	static int kv_sort(const void* _kv1, const void* _kv2)
	{
		KeyValue* kv1 = (KeyValue*) _kv1;
		KeyValue* kv2 = (KeyValue*) _kv2;
		return kv1->key - kv2->key;
	}

	void print_cell_count_histograms(CauchyTerm** terms_dp, const int shape_range, int* terms_per_shape, int* cell_counts_cen )
	{
		printf("<-------------------------------- Table Cell Count Summary: ------------------------------->\n");
		if(HALF_STORAGE)
		{
			printf("Note: HALF_STORAGE method was selected in cauchy_types.hpp\n");
			printf("Note: The cell counts below are therefore doubled!\n");
		}
		#if _WIN32
			KeyValue* table_counts[MAX_HP_SIZE];
		#else 
			KeyValue* table_counts[shape_range];
		#endif

		for(int i = 0; i < shape_range; i++)
		{	
			BYTE_COUNT_TYPE table_size_bytes = 2 * cell_counts_cen[i] * sizeof(KeyValue);
			table_counts[i] = (KeyValue*) malloc( table_size_bytes );
			null_ptr_check(table_counts[i]);
			memset(table_counts[i], kByteEmpty, table_size_bytes);
		}

		for(int shape = 1; shape < shape_range; shape++)
		{
			// Now create the histogram tables of range of cell counts per shape
			KeyValue* kv_query;
			CauchyTerm* terms = terms_dp[shape];
			int Nt_shape = terms_per_shape[shape];
			for(int i = 0; i < Nt_shape; i++)
			{
				CauchyTerm* term = terms + i;
				const int m = term->m;
				const int cells_gtable = term->cells_gtable * (1+HALF_STORAGE);
				const int max_cells = cell_counts_cen[m];
				BYTE_COUNT_TYPE table_size = 2 * max_cells;
				if( hashtable_find(table_counts[m], &kv_query, cells_gtable, table_size) )
				{
					printf(RED "[ERROR Print Table Histograms:] hashtable_find reported failure! Debug Here!" NC "\n");
					exit(1);
				}
				// If there is no entry, add entry 
				if(kv_query == NULL)
				{
					KeyValue kv;
					kv.key = cells_gtable;
					kv.value = 1;
					if( hashtable_insert(table_counts[m], &kv, table_size) )
					{
						printf(RED "[ERROR Print Table Histograms:] hashtable_insert reported failure! Debug Here!" NC "\n");
						exit(1);
					}
				}
				else 
					kv_query->value += 1;
			}
		}
		for(int m = 1; m < shape_range; m++)
		{	
			int Nt_shape = terms_per_shape[m];
			if(Nt_shape > 0)
			{
				const int max_cells = cell_counts_cen[m];
				BYTE_COUNT_TYPE table_size = 2 * max_cells;
				int counts = 0;
				KeyValue* enteries = (KeyValue*) malloc(table_size * sizeof(KeyValue));
				null_ptr_check(enteries);
				printf("Tables of Shape %d: %d Terms Total! Max Cells is: %d\n", m, Nt_shape, max_cells);
				for(uint i = 0; i < table_size; i++)
					if(table_counts[m][i].key != kEmpty)
						enteries[counts++] = table_counts[m][i];
				// Sort based on keys
				qsort(enteries, counts, sizeof(KeyValue), kv_sort);
				for(int i = 0; i < counts; i++)
					printf("  %d/%d terms have %d cells\n", enteries[i].value, Nt_shape, enteries[i].key);
				free(enteries);
			}
		}
		printf(">------------------------------ End of Cell Count Summary: --------------------------------<\n");

		for(int i = 0; i < shape_range; i++)
			free(table_counts[i]);
	}

	BYTE_COUNT_TYPE print_coalign_storage_memory(CoalignmentElemStorage* coalign_storage, const int num_threads_tp_to_muc)
	{
		BYTE_COUNT_TYPE coalign_store_bytes = 0;
		BYTE_COUNT_TYPE coalign_store_As_bytes = 0;
		BYTE_COUNT_TYPE coalign_store_ps_bytes = 0;
		BYTE_COUNT_TYPE coalign_store_bs_bytes = 0;
		BYTE_COUNT_TYPE coalign_store_c_maps_bytes = 0;
		for(int i = 0; i < num_threads_tp_to_muc; i++)
			coalign_store_bytes += coalign_storage[i].get_total_byte_count(&coalign_store_As_bytes, &coalign_store_ps_bytes, &coalign_store_bs_bytes, &coalign_store_c_maps_bytes);
		printf("-Coalign Element Storage: Memory Usage Breakdown: (threaded %d)\n", num_threads_tp_to_muc);
		printf("  As use: %.3lf MBs\n", ((double)coalign_store_As_bytes) / (1024*1024) );
		printf("  ps/qs each use: %.3lf MBs\n", ((double)coalign_store_ps_bytes) / (1024*1024) );
		printf("  bs use: %.3lf MBs\n", ((double)coalign_store_bs_bytes) / (1024*1024) );
		printf("  c_maps/cs_maps each use: %.3lf MBs\n", ((double)coalign_store_c_maps_bytes) / (1024*1024) );
		printf("  *Total: %.3lf MBs\n", ((double)coalign_store_bytes) / (1024*1024));
		return coalign_store_bytes;
	}

	BYTE_COUNT_TYPE print_reduce_storage_memory(ReductionElemStorage* reduce_storage, const int num_threads_ftr_gtables)
	{
		BYTE_COUNT_TYPE reduce_store_bytes = 0;
		BYTE_COUNT_TYPE reduce_store_As_bytes = 0;
		BYTE_COUNT_TYPE reduce_store_ps_bytes = 0;
		BYTE_COUNT_TYPE reduce_store_bs_bytes = 0;

		for(int i = 0; i < num_threads_ftr_gtables; i++)
			reduce_store_bytes += reduce_storage[i].get_total_byte_count(&reduce_store_As_bytes, &reduce_store_ps_bytes, &reduce_store_bs_bytes);
		printf("-Reduction Element Storage: Memory Usage Breakdown: (threaded %d)\n", num_threads_ftr_gtables);
		printf("  As use: %.3lf MBs\n", ((double)reduce_store_As_bytes) / (1024*1024) );
		printf("  ps use: %.3lf MBs\n", ((double)reduce_store_ps_bytes) / (1024*1024) );
		printf("  bs use: %.3lf MBs\n", ((double)reduce_store_bs_bytes) / (1024*1024) );
		printf("  *Total: %.3lf MBs\n", ((double)reduce_store_bytes) / (1024*1024));
		return reduce_store_bytes;
	}

	void print_total_estimator_memory(ChunkedPackedTableStorage* gb_tables, 
		CoalignmentElemStorage* coalign_storage,
		ReductionElemStorage* reduce_storage,
		const BYTE_COUNT_TYPE Nt, 
		bool before_ftr, 
		const int num_threads_tp_to_muc,
		const int num_threads_ftr_gtables)
	{
		BYTE_COUNT_TYPE coalign_store_bytes;
		BYTE_COUNT_TYPE reduce_store_bytes;
		if(before_ftr)
		{
			printf("<------------------------------ TP to MUC CF Memory Breakdown ----------------------------->\n");		
			reduce_store_bytes = print_reduce_storage_memory(reduce_storage, num_threads_ftr_gtables);		
			coalign_store_bytes = print_coalign_storage_memory(coalign_storage, num_threads_tp_to_muc);
		}
		else
		{
			printf("<----------------------------- FTR/Gtable CF Memory Breakdown ----------------------------->\n");
			coalign_store_bytes = print_coalign_storage_memory(coalign_storage, num_threads_tp_to_muc);
			reduce_store_bytes = print_reduce_storage_memory(reduce_storage, num_threads_ftr_gtables);		
		}

		int max_threads = num_threads_ftr_gtables > num_threads_tp_to_muc ? num_threads_ftr_gtables : num_threads_tp_to_muc;
		BYTE_COUNT_TYPE tables_mem = get_table_memory_usage(gb_tables, max_threads, true);
		BYTE_COUNT_TYPE term_mem_usage = Nt * sizeof(CauchyTerm);
		printf("*Term Structure Memory Usage: %.3lf MBs\n", ((double)term_mem_usage) / (1024*1024));
		printf("*Peak CF Memory Usage: %.3lf MBs\n", ((double)(coalign_store_bytes + reduce_store_bytes + tables_mem +  term_mem_usage)) / (1024*1024) );
		printf(">------------------------------ End of CF Memory Breakdown --------------------------------<\n");
	}

};

// Logs a LTI/LTV/Nonlinear simulation
struct SimulationLogger
{
	char* log_dir;
	int num_steps;
	int n;
	int p;
	int pncc;
	int cmcc;
	double* true_state_history; //= {x0, x1, ... , x_{num_steps} }, x_i \in R^n
	double* msmt_history; //= {z0, z1, ... , z_{num_steps} }, z_i \in R^p
	double* msmt_noise_history; //= {v0, v1, ... , v_{num_steps} }, v_i \in R^p
	double* proc_noise_history; //= {w0, w1, ... , w_{num_steps-1} }, w_i \in R^pncc
	double* x0;
	CauchyDynamicsUpdateContainer* cduc;
	KalmanDynamicsUpdateContainer* kduc;
	void (*kduc_transition_model)(KalmanDynamicsUpdateContainer*, double* xk1, double* w);
    void (*kduc_msmt_model)(KalmanDynamicsUpdateContainer*, double* z, double* v);
	void (*cduc_transition_model)(CauchyDynamicsUpdateContainer*, double* xk1, double* w);
    void (*cduc_msmt_model)(CauchyDynamicsUpdateContainer*, double* z, double* v);

	SimulationLogger(char* _log_dir, int _num_steps,
		double* _x0, KalmanDynamicsUpdateContainer* _kduc,
		void (*_transition_model)(KalmanDynamicsUpdateContainer*, double* xk1, double* w),
    	void (*_msmt_model)(KalmanDynamicsUpdateContainer*, double* z, double* v)
	)
	{
		log_dir = _log_dir;
		num_steps = _num_steps;
		if(_kduc == NULL)
		{
			printf(RED "[ERROR SimulationLogger:] _kduc==NULL! Must provide an initialized KalmanDynamicsUpdateContainer! Exiting!" NC "\n");
			exit(1);
		}
		kduc = _kduc;
        assert_correct_kalman_dynamics_update_container_setup(kduc);
		n = kduc->n;
		p = kduc->p;
		pncc = kduc->pncc;
		cmcc = kduc->cmcc;
		kduc_transition_model = _transition_model;
		kduc_msmt_model = _msmt_model;
		cduc = NULL;
		cduc_transition_model = NULL;
		cduc_msmt_model = NULL;
		init_simulation_histories();
		memcpy(x0, _x0, n * sizeof(double));
		if(cmcc != 0)
		{
			printf(RED "[ERROR SimulationLogger:] cmcc != 0. This function does not yet support controls! Think of how to implement, or remove this error message! Exiting!" NC "\n");
			exit(1);
		}
	}

	SimulationLogger(char* _log_dir, int _num_steps,
		double* _x0, CauchyDynamicsUpdateContainer* _cduc,
		void (*_transition_model)(CauchyDynamicsUpdateContainer*, double* xk1, double* w),
    	void (*_msmt_model)(CauchyDynamicsUpdateContainer*, double* z, double* v)
	)
	{
		log_dir = _log_dir;
		num_steps = _num_steps;
		if(_cduc == NULL)
		{
			printf(RED "[ERROR SimulationLogger:] _cduc==NULL! Must provide an initialized CauchyDynamicsUpdateContainer! Exiting!" NC "\n");
			exit(1);
		}
		cduc = _cduc;
        assert_correct_cauchy_dynamics_update_container_setup(cduc);
		n = cduc->n;
		p = cduc->p;
		pncc = cduc->pncc;
		cmcc = cduc->cmcc;
		cduc_transition_model = _transition_model;
		cduc_msmt_model = _msmt_model;
		kduc = NULL;
		kduc_transition_model = NULL;
		kduc_msmt_model = NULL;
		init_simulation_histories();
		memcpy(x0, _x0, n * sizeof(double));
		if(cmcc != 0)
		{
			printf(RED "[ERROR SimulationLogger:] cmcc != 0. This function does not yet support controls! Think of how to implement, or remove this error message! Exiting!" NC "\n");
			exit(1);
		}

	}

	void init_simulation_histories()
	{
		int T = num_steps + 1;
		true_state_history = (double*) malloc(T * n * sizeof(double));
		msmt_history = (double*) malloc(T * p * sizeof(double));
		msmt_noise_history = (double*) malloc(T * p * sizeof(double));
		proc_noise_history = (double*) malloc(num_steps * pncc * sizeof(double));
		x0 = (double*) malloc(n * sizeof(double));
	}

	// Logs a simulation's true_states, process, measurement, and noise histories
	// true_states = {x0, x1, ... , x_{num_steps} }, x_i \in R^n
	// msmt_history = {z0, z1, ... , z_{num_steps} }, z_i \in R^p
	// msmt_noise_history = {v0, v1, ... , v_{num_steps} }, v_i \in R^p
	// proc_noise_history = {w0, w1, ... , w_{num_steps-1} }, w_i \in R^pncc
	void run_simulation_and_log()
	{
        // We probably need to either instruct user these functions could change the cduc/kduc initialization via the callbacks below (or somehow, do it internally)
		if(cduc != NULL)
		{
			int init_step = cduc->step;
            simulate_dynamic_system(num_steps, n, pncc, p, x0,  
				true_state_history, msmt_history,
				proc_noise_history, msmt_noise_history,
				cduc, cduc_transition_model, cduc_msmt_model, true);
            cduc->step = init_step;
		}
		else
		{
            int init_step = kduc->step;
			simulate_dynamic_system(num_steps, n, cmcc, pncc, p, x0, NULL, 
				true_state_history, msmt_history, 
				proc_noise_history, msmt_noise_history,
				kduc, kduc_transition_model, kduc_msmt_model, true);
            kduc->step = init_step;
		}
		if(log_dir == NULL)
		{
			printf("SimulationLogger: No log directory provided, so not logging simulation!\n");
			return;
		}

		check_dir_and_create(log_dir);
		char* tmp_path = (char*) malloc(4096);
		int len_log_dir = strlen(log_dir);
		assert(len_log_dir < 4000);
		FILE* f_true_state_history;
		FILE* f_msmt_history;
		FILE* f_msmt_noise_history;
		FILE* f_proc_noise_history;
		char slash_delimit[2] = "/";
		char no_delimit[2] = "";
		char* delimit = log_dir[len_log_dir-1] == '/' ? no_delimit : slash_delimit;

		sprintf(tmp_path, "%s%s%s", log_dir, delimit, "true_states.txt");
		f_true_state_history = fopen(tmp_path, "w");
		if(f_true_state_history == NULL)
		{
			printf(RED "[ERROR: log_simulation_history] true_states.txt file path has failed! Debug here!" NC "\n");
			exit(1);
		}
		sprintf(tmp_path, "%s%s%s", log_dir, delimit, "msmts.txt");
		f_msmt_history = fopen(tmp_path, "w");
		if(f_msmt_history == NULL)
		{
			printf(RED "[ERROR: log_simulation_history] msmts.txt file path has failed! Debug here!" NC "\n");
			exit(1);
		}
		sprintf(tmp_path, "%s%s%s", log_dir, delimit, "msmt_noises.txt");
		f_msmt_noise_history = fopen(tmp_path, "w");
		if(f_msmt_noise_history == NULL)
		{
			printf(RED "[ERROR: log_simulation_history] msmt_noises.txt file path has failed! Debug here!" NC "\n");
			exit(1);
		}
		sprintf(tmp_path, "%s%s%s", log_dir, delimit, "proc_noises.txt");
		f_proc_noise_history = fopen(tmp_path, "w");
		if(f_proc_noise_history == NULL)
		{
			printf(RED "[ERROR: log_simulation_history] proc_noises.txt file path has failed! Debug here!" NC "\n");
			exit(1);
		}

		for(int i = 0; i <= num_steps; i++)
		{
			log_double_array_to_file(f_true_state_history, true_state_history + i*n, n);
			log_double_array_to_file(f_msmt_history, msmt_history + i*p, p);
			log_double_array_to_file(f_msmt_noise_history, msmt_noise_history + i*p, p);
			if(i != num_steps )
				log_double_array_to_file(f_proc_noise_history, proc_noise_history + i*pncc, pncc);
		}
		fclose(f_true_state_history);
		fclose(f_msmt_history);
		fclose(f_msmt_noise_history);
		fclose(f_proc_noise_history);
		free(tmp_path);
	}

	~SimulationLogger()
	{
		free(true_state_history);
		free(msmt_history);
		free(msmt_noise_history);
		free(proc_noise_history);
		free(x0);
	}
};

struct CauchyPoint3D
{
    double x;
    double y;
    double z;
};

struct CauchyPoint2D
{
    double x;
    double y;
};

// Numerical checks on the covariance computed
int covariance_checker(C_COMPLEX_TYPE* covariance, const int d, const int win_num, const int step, const int total_steps, bool with_warnings)
{
  int cov_error_flags = 0;
  int d_squared = d * d;
  double* cov_real = (double*) malloc(d_squared * sizeof(double));
  null_ptr_check(cov_real);
  double* cov_imag = (double*) malloc(d_squared * sizeof(double));
  null_ptr_check(cov_imag);
  double* cov_work = (double*) malloc(d_squared * sizeof(double));
  null_ptr_check(cov_work);
  double* cov_eigs = (double*) malloc(d * sizeof(double));
  null_ptr_check(cov_eigs);
  // Split covariance into real and imaginary
  for(int i = 0; i < d_squared; i++)
  {
    cov_real[i] = creal(covariance[i]);
    cov_imag[i] = cimag(covariance[i]);
  }

  // The following checks take place to the calculated covariance matrix
  // Check whether the eigenvalues of the covariance matrix are all positive (no small eig consideration here... window manager gives warnings for this)
  // Check whether the covariance matrix is valid (the correlations are less than or equal to +/-1)
  // Check whether the ratio of the imag part to the real part is less than a defined threshold
  // If all checks are OK, then we return 0
  // If checks are not OK, the following bits are set on return
  // If eigs are not positive, flag |= (1 << 0)
  // If correlations are not \leq +/-1, flag |= (1<<1)
  // If ratio of imag part to real part is not less than imag_over_real_threshold, flag |= (1<<2)
  // If any imaginary values of the covariance is larger than the hard limit, flag |= (1<<3)

  // Check positivity of eigenvalues
  
  // LAPACKE CODE
  //memcpy(cov_work, cov_real, d_squared*sizeof(double));
  //memset(cov_eigs, 0, d*sizeof(double));
  //LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', d, cov_work, d, cov_eigs);
  
  // STAND ALONE EVAL/VEC CODE
  sym_eig(cov_real, cov_eigs, cov_work, d);


  for(int i = 0; i < d; i++)
  {
    if(cov_eigs[i] < COV_EIGENVALUE_TOLERANCE)
      cov_error_flags |= (1<<COV_ERROR_FLAGS_INVALID_EIGENVALUES);
  }
  if(with_warnings && (cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_EIGENVALUES) ))
  {
    printf(YEL"[Covariance Checker: Window %d, Step %d/%d] Eigenvalue Check: Failed\n", 
      win_num, step, total_steps);
    print_vec(cov_eigs, d);
    printf(NC"\n");
  }

  // Check Correlations 
  memset(cov_work, 0, d_squared * sizeof(double));
  for(int i = 0; i < d-1; i++)
  {
    cov_work[i*d+i] = 1;
    double sig_ii = sqrt(cov_real[i*d+i]);
    for(int j = i+1; j < d; j++)
    {
      double sig_jj = sqrt(cov_real[j*d+j]);
      double cov_ij = cov_real[i*d+j];
      double corr_ij = cov_ij / (sig_ii * sig_jj);
      cov_work[i*d+j] = corr_ij;
      if( fabs(corr_ij) > 1 )
        cov_error_flags |= (1<<COV_ERROR_FLAGS_INVALID_CORRELATION);
    }
  }
  if(with_warnings && (cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_CORRELATION) ))
  {
    printf(YEL"[Covariance Checker: Window %d, Step %d/%d] Correlation Check: Failed", 
      win_num, step, total_steps);
    printf("Correlation Matrix is: (Upper Part)\n");
    cov_work[d_squared-1] = 1;
    print_mat(cov_work, d, d);
    printf(NC"\n");
  }

  // Check ratio of real to imaginary magnitude
  for(int i = 0; i < d; i++)
  {
    for(int j = i; j < d; j++)
    {
	  double cr = fabs(cov_real[i*d+j]);
	  double ci = fabs(cov_imag[i*d+j]);
	  double ratio = ci / (cr + 1e-15);
      cov_work[i*d + j] = ratio;
      if( cov_work[i*d + j] > THRESHOLD_COVARIANCE_IMAG_TO_REAL)
        cov_error_flags |= (1<<COV_ERROR_FLAGS_INVALID_I2R_RATIO);
    }
  }
  if(with_warnings && (cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_I2R_RATIO) ))
  {
    printf(YEL"[Covariance Checker: Window %d, Step %d/%d] Imag/Real Ratio Check: Failed", 
      win_num, step, total_steps);
    printf("Matrix of Imag/Real ratios (Upper Part) is: \n");
    print_mat(cov_work, d, d);
    printf(NC"\n");
  }

  // Check covariance imaginary part hard limit
  for(int i = 0; i < d*d; i++)
  {
	  if( fabs(cov_imag[i]) > HARD_LIMIT_IMAGINARY_COVARIANCE )
	  	cov_error_flags |= (1<<COV_ERROR_FLAGS_INVALID_IMAGINARY_VALUE);
  }
  if(with_warnings && (cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_IMAGINARY_VALUE) ))
  {
	  printf(YEL"[Covariance Checker: Window %d, Step %d/%d] Imag Hard Limit Check: Failed", 
      win_num, step, total_steps);
      printf("Max (absolute) Imag value of %d x %d covariance is: %lf\n", d, d, max_abs_array(cov_imag, d_squared));
      printf(NC"\n");
  }
  // Free heap 
  free(cov_real);
  free(cov_imag);
  free(cov_work);
  free(cov_eigs);

  return cov_error_flags;
}

void print_numeric_moment_errors()
{
	printf("Estimator Has accumulated the following errors since being initialized:\n");
	printf(RED "FUNCTION print_numeric_moment_errors not implemented yet! Please fill this in! Exiting!\n");
	exit(1);
}

void set_tr_search_idxs_ordering(const int* const ordering, int n)
{
	for(int i = 0; i < n; i++)
		TR_SEARCH_IDXS_ORDERING[i] = ordering[i];
}

// Util Is Nan 
bool CHECK_IS_NAN(double x) 
{
    return x != x;
}

#endif