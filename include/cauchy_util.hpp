#ifndef _CAUCHY_UTIL_HPP_
#define _CAUCHY_UTIL_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "gtable.hpp"

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
		
	bool F[cmcc];
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


void get_contiguous_bs_from_term_list(double*** _bs, int*** _shape_idxs, CauchyTerm* terms, const int terms_total, int* terms_per_shape, const int shape_range, const int d)
{
	// _bs is a trip point to initialize double pointer bs
	// _shape_idxs is a trip point to initialize double pointer shape_idxs
	// bs[i] are the bs list (array) of all terms with i hyperplanes
	// shape_idxs[i][j] is the index of the "bs[i] + d*j" vector in the term list
	// counts[i] specifies how many terms have i hyperplanes

	*_bs = (double**) malloc(shape_range * sizeof(double*));
	null_dptr_check((void**)*_bs);
	*_shape_idxs = (int**) malloc(shape_range * sizeof(int*));
	null_dptr_check((void**)*_shape_idxs);
	int* counts = (int*) malloc(shape_range * sizeof(int));
	null_ptr_check(counts);
	for(int i = 0; i < shape_range; i++)
	{
		if(terms_per_shape[i] > 0)
		{    
			(*_bs)[i] = (double*) malloc((size_t)terms_per_shape[i] * d * sizeof(double));
			null_ptr_check((*_bs)[i]);
			(*_shape_idxs)[i] = (int*) malloc((size_t)terms_per_shape[i] * sizeof(int));
			null_ptr_check((*_shape_idxs)[i]);
		}
		else
		{
			(*_bs)[i] = (double*) malloc(0);
			(*_shape_idxs)[i] = (int*) malloc(0);
		}
		counts[i] = 0;
	}
	double** bs = *_bs;
	int** shape_idxs = *_shape_idxs;
	for(int i = 0; i < terms_total; i++)
	{
		int m = terms[i].m;
		int count = counts[m]++;
		shape_idxs[m][count] = i;
		memcpy( bs[m] + count*d, terms[i].b, d*sizeof(double) );
	}
	free(counts);
}

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
		printf("[WARN ForwardFlagArray Win Num %d, Step %d/%d:] Too many threads allocated (idx_count < num_threads-1)...\n", win_num, step, total_steps);
		is_print = true;
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
		printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Lowering thread count to %d...\n", win_num, step, total_steps, new_num_threads);
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
		  is_print = true;
		  printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Too many threads allocated...\n", win_num, step, total_steps);
		  new_num_threads = idx_count;
		  printf("[WARN ForwardFlagArray: Win Num %d, Step %d/%d] Lowering thread count to %d...\n", win_num, step, total_steps, new_num_threads);
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

template<typename T>
void ptr_swap(T** x, T** y)
{
	T* z = *x;
	*x = *y;
	*y = z;
}

template<typename T>
void val_swap(T* x, T* y)
{
	T z = *x;
	*x = *y;
	*y = z;
}

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
			current_page_idxs[i] = 0;			
		}
		chunked_gtables = (GTABLE*) malloc(pages_at_start * sizeof(GTABLE));
		chunked_gtable_ps = (GTABLE*) malloc(pages_at_start * sizeof(GTABLE));
		chunked_btables = (BKEYS*) malloc(pages_at_start * sizeof(BKEYS));
		chunked_btable_ps = (BKEYS*) malloc(pages_at_start * sizeof(BKEYS));
		for(uint page_idx = 0; page_idx < pages_at_start; page_idx++)
		{
			chunked_gtable_ps[page_idx] = (GTABLE) malloc(page_size_bytes); //page_alloc();	
			chunked_gtables[page_idx] = (GTABLE) malloc(page_size_bytes); //page_alloc();
			chunked_btables[page_idx] = (BKEYS) malloc(page_size_bytes); //page_alloc();
			chunked_btable_ps[page_idx] = (BKEYS) malloc(page_size_bytes); //page_alloc();
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
			chunked_gtables[new_page_idx] = (GTABLE) malloc(page_size_bytes); //page_alloc();
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
			chunked_btables[new_page_idx] = (BKEYS) malloc(page_size_bytes); //page_alloc();
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
			chunked_btable_ps[new_page_idx] = (BKEYS) malloc(page_size_bytes); //page_alloc();
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
		assert(page_idx < page_limits[0]); // REMOVE
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[0][page_idx];
		if( (elems_used_in_page + elems_gtable) > page_size_gtable_num_elements )
		{
			page_idx = ++current_page_idxs[0];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			assert(used_elems_per_page[0][page_idx] == 0); // REMOVE
		}
		*gtable = chunked_gtables[page_idx] + elems_used_in_page;
		if(mem_ptr_incr)
			used_elems_per_page[0][page_idx] += elems_gtable;
	}

	// This function assumes that there is no way to increment over the number of allocated pages
	void set_term_btable_pointer(BKEYS* btable, int cells_gtable, const bool mem_ptr_incr = true)
	{
		uint page_idx = current_page_idxs[2];
		assert(page_idx < page_limits[2]); // REMOVE
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[2][page_idx];
		if( (elems_used_in_page + cells_gtable) > page_size_btable_num_elements )
		{
			page_idx = ++current_page_idxs[2];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			assert(used_elems_per_page[2][page_idx] == 0); // REMOVE
		}
		*btable = chunked_btables[page_idx] + elems_used_in_page;
		if(mem_ptr_incr)
			used_elems_per_page[2][page_idx] += cells_gtable;
	}
	
	// This function assumes that there is no way to increment over the number of allocated pages
	void set_term_bp_table_pointer(BKEYS* bp_table, int cells_gtable, const bool mem_ptr_incr = true)
	{
		uint page_idx = current_page_idxs[3];
		assert(page_idx < page_limits[3]); // REMOVE
		BYTE_COUNT_TYPE elems_used_in_page = used_elems_per_page[3][page_idx];
		if( (elems_used_in_page + cells_gtable) > page_size_btable_num_elements )
		{
			page_idx = ++current_page_idxs[3];
			elems_used_in_page = 0; // should be zero, we are incrementing to next page
			assert(used_elems_per_page[3][page_idx] == 0); // REMOVE
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
		// free chunked gtable_ps
		for(uint page_idx = 0; page_idx < page_limits[1]; page_idx++)
			free(chunked_gtable_ps[page_idx]);
		free(chunked_gtable_ps);
		free(used_elems_per_page[1]);
		// free chunked btables
		for(uint page_idx = 0; page_idx < page_limits[2]; page_idx++)
			free(chunked_btables[page_idx]);
		free(chunked_btables);
		free(used_elems_per_page[2]);
		// free chunked btable_ps
		for(uint page_idx = 0; page_idx < page_limits[3]; page_idx++)
			free(chunked_btable_ps[page_idx]);
		free(chunked_btable_ps);
		free(used_elems_per_page[3]);
	}

};

// Cauchy Stats -- Used to determine different statistics as the estimator runs

struct CauchyStats
{

	BYTE_COUNT_TYPE get_table_memory_usage(ChunkedPackedTableStorage* gb_tables, bool with_print)
	{
		uint* current_page_idxs = gb_tables->current_page_idxs;
		BYTE_COUNT_TYPE** used_elems_per_page = gb_tables->used_elems_per_page;
		BYTE_COUNT_TYPE page_size_bytes = gb_tables->page_size_bytes;
		uint* page_limits = gb_tables->page_limits;
		printf("--- Gtables, Gtable_ps, Btables, Btable_ps Memory Usage: ---\n");
		double gtables_bytes = 0;
		double gtable_ps_bytes = 0;
		double btables_bytes = 0;
		double btable_ps_bytes = 0;
		for(uint i = 0; i <= current_page_idxs[0]; i++)
			gtables_bytes += used_elems_per_page[0][i] * sizeof(GTABLE_TYPE);
		for(uint i = 0; i <= current_page_idxs[1]; i++)
			gtable_ps_bytes += used_elems_per_page[1][i] * sizeof(GTABLE_TYPE);
		for(uint i = 0; i <= current_page_idxs[2]; i++)
			btables_bytes += used_elems_per_page[2][i] * sizeof(BKEYS_TYPE);
		for(uint i = 0; i <= current_page_idxs[3]; i++)
			btable_ps_bytes += used_elems_per_page[3][i] * sizeof(BKEYS_TYPE);
		BYTE_COUNT_TYPE page_size_mbs = page_size_bytes / (1024*1024);
		if(with_print)
		{
			printf("  Gtables: Allocated pages: %d x %llu MBs per page. In Use: %.3lf MBs\n", page_limits[0], page_size_mbs, gtables_bytes / (1024*1024));
			printf("  Gtable_ps: Allocated pages: %d x %llu MBs per page. In Use: %.3lf MBs\n", page_limits[1], page_size_mbs, gtable_ps_bytes / (1024*1024));
			printf("  Btables: Allocated pages: %d x %llu MBs per page. In Use: %.3lf MBs\n", page_limits[2], page_size_mbs, btables_bytes / (1024*1024));
			printf("  Btable_ps: Allocated pages: %d x %llu MBs per page. In Use: %.3lf MBs\n", page_limits[3], page_size_mbs, btable_ps_bytes / (1024*1024));
			printf("--- Total G/Btable Memory Pressure: %.3lf ---\n", (gtables_bytes + gtable_ps_bytes + btables_bytes + btable_ps_bytes) / (1024*1024));
		}
		return gtables_bytes + gtable_ps_bytes + btables_bytes + btable_ps_bytes;
	}

	// Prints the memory usage of the A,p,b elements of all terms
	BYTE_COUNT_TYPE get_elem_mem_usage(int* terms_per_shape, int shape_range, int d, bool with_print)
	{
		BYTE_COUNT_TYPE mem_usage[3];
		memset(mem_usage, 0, 3 * sizeof(BYTE_COUNT_TYPE));
		for(int i = 0; i < shape_range; i++)
		{
			BYTE_COUNT_TYPE Nt_shape = terms_per_shape[i];
			if(Nt_shape > 0)
			{
				mem_usage[0] += Nt_shape * i * d * sizeof(double);// A
				mem_usage[1] += Nt_shape * i * sizeof(double); // p 
				mem_usage[2] += Nt_shape * d * sizeof(double); // b
			}
		}
		if(with_print)
		{
			double A_mbs = ((double)mem_usage[0]) / (1024*1024);
			double p_mbs = ((double)mem_usage[1]) / (1024*1024);
			double b_mbs = ((double)mem_usage[2]) / (1024*1024);
			printf("--- Memory Usage of A/p/q/b Parameters: ---\n");
			printf("  As use %.3lf MBs\n", A_mbs);
			printf("  ps (and qs) each use %.3lf MBs\n", p_mbs);
			printf("  bs use %.3lf MBs\n", b_mbs);
			printf("  Total: %.3lf Mbs---\n", A_mbs + 2*p_mbs + b_mbs);
			printf("--- End of A/p/q/b Summary: ---\n");
		}
		return mem_usage[0] + 2*mem_usage[1] + mem_usage[2];
	}

	static int kv_sort(const void* _kv1, const void* _kv2)
	{
		KeyValue* kv1 = (KeyValue*) _kv1;
		KeyValue* kv2 = (KeyValue*) _kv2;
		return kv1->key - kv2->key;
	}

	void print_cell_count_histograms(CauchyTerm* terms, const int shape_range, int* terms_per_shape, int* cell_counts_cen, const int Nt)
	{
		if(HALF_STORAGE)
			printf("--- Table Cell Count Summary. Half Storage Used. Results are therefore doubled! ---\n");
		else
			printf("--- Table Cell Count Summary ---\n");

		KeyValue* table_counts[shape_range];
		for(int i = 0; i < shape_range; i++)
		{	
			BYTE_COUNT_TYPE table_size_bytes = 2 * cell_counts_cen[i] * sizeof(KeyValue);
			table_counts[i] = (KeyValue*) malloc( table_size_bytes );
			memset(table_counts[i], kByteEmpty, table_size_bytes);
		}

		// Now create the histogram tables of range of cell counts per shape
		KeyValue* kv_query;
		for(int i = 0; i < Nt; i++)
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

		for(int m = 0; m < shape_range; m++)
		{	
			int Nt_shape = terms_per_shape[m];
			if(Nt_shape > 0)
			{
				const int max_cells = cell_counts_cen[m];
				BYTE_COUNT_TYPE table_size = 2 * max_cells;
				int counts = 0;
				KeyValue* enteries = (KeyValue*) malloc(table_size * sizeof(KeyValue));
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
		printf("--- End of Cell Count Summary ---\n");
		for(int i = 0; i < shape_range; i++)
			free(table_counts[i]);
	}

	void print_total_estimator_memory(ChunkedPackedTableStorage* gb_tables, const BYTE_COUNT_TYPE Nt, const int shape_range, int* terms_per_shape, const int d)
	{
		printf("---------- CF Memory Breakdown ---------- \n");
		BYTE_COUNT_TYPE elem_mem_usage = get_elem_mem_usage(terms_per_shape, shape_range, d, true);
		BYTE_COUNT_TYPE tables_mem = get_table_memory_usage(gb_tables, true);
		BYTE_COUNT_TYPE term_mem_usage = Nt * sizeof(CauchyTerm);
		printf("Term Structure Memory Usage: %llu MBs\n", term_mem_usage / (1024*1024));
		printf("Total CF Memory Usage: %llu MBs\n", (elem_mem_usage + tables_mem +  term_mem_usage) / (1024*1024) );
		printf("---------- End of CF Memory Breakdown ---------- \n");
	}

};

#endif