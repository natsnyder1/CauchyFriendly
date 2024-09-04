#ifndef _TERM_REDUCTION_HPP_
#define _TERM_REDUCTION_HPP_

#include "cauchy_types.hpp"
#include "gtable.hpp"
#include "cauchy_term.hpp"

// ----- Begin Fast Term Reduction Method ----- //

struct PointMap
{
    // The coordinate of the j-th axis point
    double p;
    // backward map coordinate, says what index point p is located in the point set
    int pi; 
};

int compare_pointmap(const void* pm1, const void* pm2)
{
    double p1 = ((PointMap*)pm1)->p;
    double p2 = ((PointMap*)pm2)->p;
    if( p1 > p2 )
        return 1;
    else if( p1 < p2 )
        return -1;
    else 
        return 0;
}

void build_ordered_point_maps(CauchyTerm* terms, double** ordered_points, int** forward_map, int** backward_map, const int n, const int d, const bool is_print)
{
    // Helper variables
    int* fm; int* bm; double* op; int pi;
    // Build the PointMap sorting structure 
    PointMap* pm = (PointMap*) malloc(n * sizeof(PointMap));
    null_ptr_check(pm);
    for(int i = 0; i < d; i++)
    {
        // Construct PointMap for the j-th axis
        for(int j = 0; j < n; j++)
        {
            pm[j].p = terms[j].b[i]; //points[j*d + i];
            pm[j].pi = j;
        }
        // Sort the PointMap array
        qsort(pm, n, sizeof(PointMap), &compare_pointmap);
        // Construct the ordered set, the forward_map, and the backward map for the j-th axis
        fm = forward_map[i];
        bm = backward_map[i];
        op = ordered_points[i];
        for(int j = 0; j < n; j++)
        {
            pi = pm[j].pi;
            op[j] = pm[j].p;
            bm[j] = pi;
            fm[pi] = j;
        }
        if(is_print)
        {
            printf("----points on %d-th axis:----\n", i);
            for(int j = 0; j < n; j++)
                printf("%.1lf, ", terms[j].b[i]);
            printf("\n");
            // Sanity check at this point
            printf("ordered_points[%d]: ", i);
            for(int j = 0; j < n; j++)
                printf("%.1lf, ", op[j]);
            printf("\n");
            printf("backward_map[%d]: ", i);
            for(int j = 0; j < n; j++)
                printf("%d, ", bm[j]);
            printf("\n");
            printf("forward_map[%d]: ", i);
            for(int j = 0; j < n; j++)
                printf("%d, ", fm[j]);
            printf("\n----------------------------\n");
        }
    }
    free(pm);
}

void swap_arr_vals(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

int ftr_binary_search(double qp, double* arr, const int n, const bool less_than)
{
    int high = n-1;
    int low = 0;
    int mid;
    while( low <= high )
    {
        mid = (high+low)/2;
        if(qp < arr[mid])
            high = mid-1;
        else 
            low = mid+1;
    }
    if(less_than)
        return low-1;
    else 
        return high+1;
}

int construct_candidate_list(int* candidate_list, double qp, int qpi, double* op, int* bm, int* F, const double ep, const int n)
{
    int candidate_list_count = 0;
    // The extents of the bounds are -1 to n, as the indices can be [0,n-1], the bounds are +/-1 for these, respectively
    // NOTE: It may be faster to walk from qp to the upper extent (qp+ep) and then from qp to the lower extent (qp-ep)
    // .... this will only be so if the range of points in [qp-ep, qp+ep] is consitently smaller than log_2(n)
    int lti = ftr_binary_search(qp - ep, op, n, true); // less then index (index of the greatest lower bound)
    int gti = ftr_binary_search(qp + ep, op, n, false); // greater then index (index of the least upper bound)
    // If there is only 1 point found, (the query point itself), then the range (gti - lti) will be 2. 
    // Only if gti-lti is greater than 2 does it imply a candidate point was found that is not the query point
    int pi;
    int i;
    if( (gti - lti) > 2 )
    {
        // There is at least one candidate (excluding the query point itself)
        // Using the backwards map, now constuct the candidate list of points
        for(i = lti + 1; i < gti; i++)
        {
            pi = bm[i]; // index of the point
            // Only consider candidates that have an index higher then the root point in question             
            if(pi > qpi)
            {
                // Exclude all points already marked as reduced / approximated out via the flag array
                if(F[pi] == pi)
                    candidate_list[candidate_list_count++] = pi;
            }
        }
    }
    return candidate_list_count;
}

int prune_candidate_list(int* candidate_list, int candidate_list_count, double qp, double* op, int* fm, const double ep, const int n)
{
    int new_candidate_list_count = candidate_list_count;
    // NOTE: It may be faster to walk from qp to the upper extent (qp+ep) and then from qp to the lower extent (qp-ep)
    // .... this will only be so if the range of points in [qp-ep, qp+ep] is consitently smaller than log_2(n)
    int lti = ftr_binary_search(qp - ep, op, n, true); // less than index (index of the greatest lower bound)
    int gti = ftr_binary_search(qp + ep, op, n, false); // greater than index (index of the least upper bound)
    // Iterate over the candidate list, checking to see if these points fall within [lti+1,gti-1]
    ++lti;
    --gti;
    for(int i = candidate_list_count-1; i > -1; --i)
    {
        // Forward map the candidate point from its index in the points array to its index in the ordered_points array (for the j-th dimension) 
        int opi = fm[candidate_list[i]];
        if( (opi < lti) || (opi > gti) )
            swap_arr_vals(&(candidate_list[i]), &(candidate_list[--new_candidate_list_count]));
    }
    return new_candidate_list_count;
}

void fast_term_reduction( 
  CauchyTerm* terms,
  int* F, 
  double** ordered_bs, 
  int** forward_map, 
  int** backward_map,
  const double ep, const int n, const int m, const int d, 
  int start = -1, int end = -1)
{
    double* point;
    double* pi;
    double* pj;
    double* Ai;
    double* Aj;
    double* op;
    int* fm;
    int* bm;
    int candidate_list_count;
    int* candidate_list = (int*) malloc(n * sizeof(int));
    null_ptr_check(candidate_list);
    const int* search_idxs = TR_SEARCH_IDXS_ORDERING;
    int i, j, l, k, cl;
    bool reduction_checks_passed;
    double A_root, A_cmp;
    int pos_gate, neg_gate;
    start = start < 0 ? 0 : start;
    end = end < 0 ? n : end;
    for(i = start; i < end; i++)
    {
        // only iterate over this point if the point has not already been reduced / approximated out
        if(F[i] == i)
        {
            // Find the candidate list for point i
            point = terms[i].b;
            // Find the initial candidate list using the first coordinate axis
            op = ordered_bs[search_idxs[0]];
            fm = forward_map[search_idxs[0]];
            bm = backward_map[search_idxs[0]];
            candidate_list_count = construct_candidate_list(candidate_list, point[search_idxs[0]], i, op, bm, F, ep, n);
            //if(candidate_list_count > 100)
            //  printf(RED "[FTR:] Term %d has initial candidate_list_count of %d"
            //  NC "\n", i, candidate_list_count);
            // If the candidate list set is not empty, then we need to prune this list 
            if(candidate_list_count)
            {
                // Prunes the candidate list over each dimension d
                for(j = 1; j < d; j++)
                {
                    op = ordered_bs[search_idxs[j]];
                    fm = forward_map[search_idxs[j]];
                    bm = backward_map[search_idxs[j]];
                    candidate_list_count = prune_candidate_list(candidate_list, candidate_list_count, point[search_idxs[j]], op, fm, ep, n);
                    if(candidate_list_count == 0)
                        break;
                }
                // If there are candidates after pruning, then these candidates fall within the hypercube, and these terms should be reduced together
                if(candidate_list_count)
                {
                  pi = terms[i].p;
                  Ai = terms[i].A;
                  for(j = 0; j < candidate_list_count; j++)
                  {
                    // Although almost always the case, when the bs combined, so do the yeis...
                    // If the reduction epsilon is slighly too large, we need to check yeis as well
                    if(WITH_FTR_P_CHECK)
                    {
                      cl = candidate_list[j];
                      reduction_checks_passed = true;
                      pj = terms[cl].p;
                      for(k = 0; k < m; k++)
                      {
                        if(fabs(pj[k] - pi[k]) > ep)
                        {
                          reduction_checks_passed = false;
                          break;
                        }
                      }
                      if(WITH_FTR_HPA_CHECK)
                      {
                        if(reduction_checks_passed)
                        {
                          Aj = terms[cl].A;
                          for(k = 0; k < m; k++)
                          {
                            pos_gate = true;
                            neg_gate = true;
                            for(l = 0; l < d; l++)
                            {
                              A_root = Ai[k];
                              A_cmp = Aj[k];
                              if(pos_gate)
                                pos_gate &= fabs(A_root - A_cmp) < ep;
                              if(neg_gate)
                                neg_gate &= fabs(A_root + A_cmp) < ep;
                              if( !(pos_gate || neg_gate) )
                                break;
                            }
                            if( !(pos_gate || neg_gate) )
                            {
                              reduction_checks_passed = false;
                              break;
                            }
                          }
                        }
                      }
                      if(reduction_checks_passed)
                        F[cl] = i;
                    }
                    // Assuming reduction epsilon is OK, then no need to check ps/As
                    else
                      F[candidate_list[j]] = i;
                  }
                }
            }
        }
    }
    free(candidate_list);
}

// ----- End Fast Term Reduction Method ----- //


// ----- Begin Threaded Fast Term Reduction Method ----- //

/*
// Tim sort functions
#define MIN_MERGE 32
// Function to merge two sorted subarrays of PointMap
void merge(PointMap* arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    PointMap* L = (PointMap*)malloc(n1 * sizeof(PointMap));
    PointMap* R = (PointMap*)malloc(n2 * sizeof(PointMap));

    // Copy data to temporary arrays
    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[]
    i = 0;
    j = 0;
    k = left;
    while (i < n1 && j < n2) {
        if (L[i].p <= R[j].p) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    // Free temporary arrays
    free(L);
    free(R);
}

// Insertion sort function for PointMap
void insertionSort(PointMap* arr, int left, int right) {
    int i, j;
    PointMap key;
    for (i = left + 1; i <= right; i++) {
        key = arr[i];
        j = i - 1;

        while (j >= left && arr[j].p > key.p) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Function to perform Timsort
void timSort(PointMap* arr, int n) 
{
    int i, size, left, mid, right;

    // Sort individual subarrays of size MIN_MERGE
    for (i = 0; i < n; i += MIN_MERGE)
        insertionSort(arr, i, fmin((i + MIN_MERGE - 1), (n - 1)));

    // Merge subarrays in a bottom-up manner
    for (size = MIN_MERGE; size < n; size = 2 * size) {
        for (left = 0; left < n; left += 2 * size) {
            mid = left + size - 1;
            right = fmin((left + 2 * size - 1), (n - 1));
            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}
*/


struct threadsort_struct
{
    PointMap* point_map;
    int n;
};

struct threadmerge_struct
{
    PointMap** arr1;
    PointMap** arr2;
    int* n_arr1;
    int* n_arr2;
};

void* threaded_pointmap_sort_callback(void* args)
{
    threadsort_struct* ts = (threadsort_struct*) args;
    qsort(ts->point_map, ts->n, sizeof(PointMap), &compare_pointmap);
    //timSort(ts->point_map, ts->n); // could optionally try this on cluster...
    return NULL;
}

void* threaded_merge(void* args)
{
    threadmerge_struct* tms = (threadmerge_struct*) args;
    PointMap* arr1 = *tms->arr1;
    PointMap* arr2 = *tms->arr2;
    const int n_arr1 = *tms->n_arr1;
    const int n_arr2 = *tms->n_arr2;
    const int n = n_arr1 + n_arr2;
    int c_arr1 = 0;
    int c_arr2 = 0;
    int c = 0;
    *tms->arr1 = (PointMap*) malloc( n * sizeof(PointMap) );
    *tms->n_arr1 = n;
    *tms->n_arr2 = 0;
    PointMap* merge_arr = *tms->arr1;
    while( (c_arr1 < n_arr1) && (c_arr2 < n_arr2) )
    {    
        if( arr1[c_arr1].p < arr2[c_arr2].p )
            merge_arr[c++] = arr1[c_arr1++];
        else 
            merge_arr[c++] = arr2[c_arr2++];
    }
    for(int i = c_arr1; i < n_arr1; i++)
        merge_arr[c++] = arr1[i];
    for(int i = c_arr2; i < n_arr2; i++)
        merge_arr[c++] = arr2[i];
    assert(c == n);
    free(arr1);
    free(arr2);
    return NULL;
}

void merge_sorted_arrays(threadsort_struct* ts, const int nts)
{
    threadmerge_struct* tms = (threadmerge_struct*) malloc(nts/2*sizeof(threadmerge_struct));
    pthread_t* tids = (pthread_t*) malloc(nts/2*sizeof(pthread_t));
    int pow_2 = (int) (log2(nts)+0.999);
    for(int j = 0; j < pow_2; j++)
    {
        // Merge Fs[cmp_idx] into Fs[rt_idx]
        int compares = nts / (1 << (j+1));
        for(int k = 0; k < compares; k++)
        {
            int rt_idx = k*(1<<(j+1));
            int cmp_idx = rt_idx + (1<<j);
            tms[k].arr1 = &ts[rt_idx].point_map;
            tms[k].arr2 = &ts[cmp_idx].point_map;
            tms[k].n_arr1 = &ts[rt_idx].n;
            tms[k].n_arr2 = &ts[cmp_idx].n;
            pthread_create(&tids[k], NULL, &threaded_merge, &tms[k]);
        }
        for(int k = 0; k < compares; k++)
            pthread_join(tids[k], NULL);
    }
	free(tms);
	free(tids);
}

void threaded_pointmap_sort(PointMap* point_map_data, const int n, const int num_threads)
{
    assert( (num_threads & (num_threads-1)) == 0);
    assert( n > num_threads);
    int npt = n / num_threads;
    int csn = 0;
    PointMap** point_map_chunks = (PointMap**) malloc(num_threads * sizeof(PointMap*));
	pthread_t* tids = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    threadsort_struct* tss = (threadsort_struct*)malloc(num_threads * sizeof(threadsort_struct));
    for(int i = 0; i < num_threads; i++)
    {
        if( i == (num_threads-1) )
            npt += n % num_threads;
        point_map_chunks[i] = (PointMap*) malloc(npt * sizeof(PointMap));
        memcpy(point_map_chunks[i], point_map_data + csn, npt * sizeof(PointMap));
        csn += npt;
        tss[i].point_map = point_map_chunks[i];
        tss[i].n = npt;
        pthread_create(&tids[i], NULL, threaded_pointmap_sort_callback, &tss[i]);
    }
    for(int i = 0; i < num_threads; i++)
        pthread_join(tids[i], NULL);

    // Merge the results
    merge_sorted_arrays(tss, num_threads);
    memcpy(point_map_data, tss[0].point_map, n*sizeof(PointMap));
    free(tss[0].point_map);
    free(point_map_chunks);
	free(tids);
	free(tss);
}

struct bopm_struct
{
    CauchyTerm* terms;
    double* op;
    int* fm;
    int* bm;
    int n;
    int i;
    int num_threads;
};

void* threaded_bopm(void* args)
{
    bopm_struct* bopm = (bopm_struct*) args;
    CauchyTerm* terms = bopm->terms;
    double* op = bopm->op;
    int* fm = bopm->fm;
    int* bm = bopm->bm;
    int n = bopm->n;
    int i = bopm->i;
    int num_threads = bopm->num_threads;
    PointMap* pm = (PointMap*) malloc(n * sizeof(PointMap));
    int pi;
    null_ptr_check(pm);
    // Construct PointMap for the j-th axis
    for(int j = 0; j < n; j++)
    {
        pm[j].p = terms[j].b[i];//points[j*d + i];
        pm[j].pi = j;
    }

    threaded_pointmap_sort(pm, n, num_threads);
    for(int j = 0; j < n; j++)
    {
        pi = pm[j].pi;
        op[j] = pm[j].p;
        bm[j] = pi;
        fm[pi] = j;
    }
    free(pm);
    return NULL;
}

void threaded_build_ordered_point_maps(CauchyTerm* terms, 
  double** ordered_points, int** forward_map, 
  int** backward_map, const int n, const int d, const int num_threads)
{
    pthread_t* tids = (pthread_t*) malloc(d*sizeof(pthread_t));
    bopm_struct* bopm = (bopm_struct*)malloc(d * sizeof(bopm_struct));
    for(int i = 0; i < d; i++)
    {
        bopm[i].terms = terms;
        bopm[i].op = ordered_points[i];
        bopm[i].fm = forward_map[i];
        bopm[i].bm = backward_map[i];
        bopm[i].n = n;
        bopm[i].i = i;
        bopm[i].num_threads = num_threads;
        pthread_create(&tids[i], NULL, threaded_bopm, &bopm[i]);   
    }
    for(int i = 0; i < d; i++)
        pthread_join(tids[i], NULL);
	free(tids);
	free(bopm);
}



struct ftr_struct
{
  CauchyTerm* terms;
  int* F;
  double** ordered_bs;
  int** forward_map;
  int** backward_map;
  double ep;
  int n; 
  int m; 
  int d;
  int start;
  int end;
};

void* threaded_fast_term_reduction_callback(void* args)
{
    ftr_struct* ftr_args = (ftr_struct*) args;
    fast_term_reduction(
        ftr_args->terms,
        ftr_args->F, 
        ftr_args->ordered_bs, 
        ftr_args->forward_map, 
        ftr_args->backward_map,
        ftr_args->ep, ftr_args->n, ftr_args->m, ftr_args->d, 
        ftr_args->start, ftr_args->end);
    return NULL;
}

struct FlagMergeStruct
{
    int* F1;
    int* F2;
    int n;
    int start;
};

void merge_flag_arrays(int* F1, int* F2, int n, int start)
{
    for(int i = start; i < n; i++)
    {
      if(F2[i] != -1)
      {
        if(F2[i] != i)
        {
          if(F1[i] == i)
          {
            // Merge condition
            if(F1[F2[i]] == F2[i])
                F1[i] = F2[i];
            // Merge With linking condition
            else //if(F1[F2[i]] != F2[i])
                F1[i] = F1[F2[i]]; 
          }
        }
      }
    }
}

void* thread_merge_flag_arrays(void* args)
{
    FlagMergeStruct* fms = (FlagMergeStruct*) args;
    merge_flag_arrays(fms->F1, fms->F2, fms->n, fms->start);
    return NULL;
}

void merge_flag_arrays(ftr_struct* ftr_args, const int num_threads)
{
    assert(num_threads > 1);

    FlagMergeStruct* fms = (FlagMergeStruct*) malloc(num_threads/2*sizeof(FlagMergeStruct));
    pthread_t* tids = (pthread_t*)malloc(num_threads / 2 * sizeof(pthread_t));
    
	assert((num_threads & (num_threads-1)) == 0); // must be a power of two
    int pow_2 = (int) (log2(num_threads)+0.999);
    for(int j = 0; j < pow_2; j++)
    {
        // Merge Fs[cmp_idx] into Fs[rt_idx]
        int compares = num_threads / (1 << (j+1));
        for(int k = 0; k < compares; k++)
        {
            int rt_idx = k*(1<<(j+1));
            int cmp_idx = rt_idx + (1<<j);
            fms[k].F1 = ftr_args[rt_idx].F;
            fms[k].F2 = ftr_args[cmp_idx].F;
            fms[k].start = ftr_args[cmp_idx].start;
            fms[k].n = ftr_args[cmp_idx].n;
            pthread_create(&tids[k], NULL, &thread_merge_flag_arrays, &fms[k]);
        }
        for(int k = 0; k < compares; k++)
            pthread_join(tids[k], NULL);
    }
	free(fms);
	free(tids);
}

void threaded_fast_term_reduction(
  CauchyTerm* terms,
  int* F, 
  double** ordered_bs, 
  int** forward_map, 
  int** backward_map,
  const double ep, const int n, 
  const int m, const int d,
  const int num_threads, const int chunk_size)
{
    pthread_t* tids = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
    ftr_struct* ftr_args = (ftr_struct*)malloc(num_threads * sizeof(ftr_struct));
    int** Fs = (int**) malloc(num_threads * sizeof(int*));
    // Divide fast term reduction into several chunks
    for(int i = 0; i < num_threads; i++)
    {
        ftr_args[i].terms = terms;
        Fs[i] = (int*) malloc(n * sizeof(int));
        memcpy(Fs[i] + i*chunk_size, F + i*chunk_size, (n - i*chunk_size) * sizeof(int));
        //memcpy(Fs[i], F, n * sizeof(int));
        ftr_args[i].F = Fs[i];
        ftr_args[i].ordered_bs = ordered_bs;
        ftr_args[i].forward_map = forward_map;
        ftr_args[i].backward_map = backward_map;
        ftr_args[i].ep = ep;
        ftr_args[i].n = n;
        ftr_args[i].m = m;
        ftr_args[i].d = d;
        ftr_args[i].start = i*chunk_size;
        ftr_args[i].end = (i == (num_threads - 1) ) ? n : (i+1)*chunk_size;
        pthread_create(&tids[i], NULL, threaded_fast_term_reduction_callback, &ftr_args[i]);
    }
    for(int i = 0; i < num_threads; i++)
      pthread_join(tids[i], NULL);

    // Flag Array Merge -- merging the partial results of FTR for each chunk
    merge_flag_arrays(ftr_args, num_threads);
    memcpy(F, ftr_args[0].F, n * sizeof(int));

    for(int i = 0; i < num_threads; i++)
        free(Fs[i]);
    free(Fs);
	free(tids);
	free(ftr_args);
}

// ----- End Threaded Fast Term Reduction Method ----- //

#endif // _TERM_REDUCTION_HPP_