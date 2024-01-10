#ifndef _CELL_ENUMERATION_HPP_
#define _CELL_ENUMERATION_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "gtable.hpp"
#include "random_variables.hpp"
#include <algorithm>

BYTE_COUNT_TYPE binomialCoeff(int n, int k)
{
    BYTE_COUNT_TYPE res = 1;

    // Since C(n, k) = C(n, n-k)
    if (k > n - k)
        k = n - k;
 
    // Calculate value of
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int i = 0; i < k; ++i) {
      res *= (n - i);
      res /= (i + 1);
    }
 
    return res;
}

int nchoosek(int n, int k)
{
    return binomialCoeff(n, k);
}

int cell_count_central(int hyp, int dim)
{
  if(hyp < dim)
    return 1 << hyp;
  BYTE_COUNT_TYPE fc = 0;
  for( int i = 0; i < dim; i++)
  {
    fc += binomialCoeff(hyp-1, i);
  }
  return (int) (2*fc);
}

int cell_count_general(int hyp, int dim)
{
  if(hyp < dim)
    return 1 << hyp;
  BYTE_COUNT_TYPE fc = 0;
  for( int i = 0; i < dim+1; i++)
  {
    fc += binomialCoeff(hyp, i);
  }
  return (int) fc;
}

// Returns encoded sign sequences for encircling a vertex, with -1 encoded as 1 and 1 encoded as 0
// The array returned is sized (2^d x d)
int* init_encoded_sign_sequences_around_vertex(int d)
{
    int num_cells_around_vetex = (1 << d);
    int* SSav = (int*) malloc( num_cells_around_vetex * d * sizeof(int) ); // Indicator Sequences around vertex
    null_ptr_check(SSav);
    for(int i = 0; i < num_cells_around_vetex; i++)
        for(int j = 0; j < d; j++)
            SSav[i*d + j] = (i & (1 << j) ) >> j;
    return SSav;
}

int* combinations(int n, int k)
{
    int* combos = (int*) malloc( nchoosek(n,k)*k*sizeof(int));
    null_ptr_check(combos);
    std::string bitmask(k, 1); // K leading 1's
    bitmask.resize(n, 0); // N-K trailing 0's
    int position = 0;
    // print integers and permute bitmask
    do {
        for (int i = 0; i < n; ++i) // [0..N-1] integers
        {
            if (bitmask[i])
            {
                //std::cout << " " << i;
                combos[position] = i;
                position += 1;
            }
        }
        //std::cout << std::endl;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return combos;
}

// Returns the integers of m that are not in combos
// anti_combos array is sized num_combos x (m-n) on return
int* init_anti_combos(int* combos, int num_combos, int m, int n)
{
    assert(m>n);
    int num_anti_combo = m-n;
    int* anti_combos = (int*) malloc(num_anti_combo*num_combos*sizeof(int));
    null_ptr_check(anti_combos);
    for(int i = 0; i < num_combos; i++)
    {
        int* anti_combo = anti_combos + i*num_anti_combo;
        int* combo = combos + i*n;
        int count = 0;
        // test if j is in combo
        for(int j = 0; j < m; j++)
        {
            bool not_in_combo = true;
            for(int k = 0; k < n; k++)
                if(j == combo[k])
                    not_in_combo = false;
            if(not_in_combo)
                anti_combo[count++] = j;
        }
    }
    return anti_combos;
}

BKEYS_TYPE remove_bit(BKEYS_TYPE b, BKEYS_TYPE bit)
{
    BKEYS_TYPE high = b >> (bit+1);
    BKEYS_TYPE low = b & ((1<<bit)-1);
    BKEYS_TYPE b_bit_removed = (high << bit) | low;
    return b_bit_removed;
}

int sort_func_B_enc(const void* p1, const void* p2)
{
  return *((int*)p1) - *((int*)p2);
}

void sort_encoded_B(int* B_enc, int cell_count)
{
    qsort(B_enc, cell_count, sizeof(int), &sort_func_B_enc);
}

int is_Bs_different(int* B1, int* B2, int cells)
{
    for(int i = 0; i < cells; i++)
    {
        if(B1[i] != B2[i])
            return true;
    }
    return false;
}

void print_B_encoded(int *B_enc, int cell_count, int m, bool with_sort)
{
    if(with_sort)
    {    
        sort_encoded_B(B_enc, cell_count);
        printf("B_enc (with %d sorted sign-vectors) is:\n", cell_count);
    }
    else
        printf("B_enc is:\n");
    for(int i = 0; i < cell_count; i++)
        printf("%d, ", B_enc[i]);
    printf("\n");
    if(HALF_STORAGE)
    {
        printf("B_enc for opposite halfspace is:\n");
        int* B_enc_opp = (int*) malloc(cell_count * sizeof(int));
        null_ptr_check(B_enc_opp);
        memcpy(B_enc_opp, B_enc, cell_count * sizeof(int));
        int rev_mask = (1<<m) - 1;
        for(int i = 0; i < cell_count; i++)
            B_enc_opp[i] ^= rev_mask;
        if(with_sort)
            qsort(B_enc_opp, cell_count, sizeof(int), &sort_func_B_enc);
        for(int i = 0; i < cell_count; i++)
            printf("%d, ", B_enc_opp[i]);
        printf("\n");
    }
}

void print_B_unencoded(int* B_enc, int cell_count, int m, bool with_sort = false)
{
    if(with_sort)
    {    
        sort_encoded_B(B_enc, cell_count);
        printf("B_enc (with sorted sign-vectors) is:\n");
    }
    else
        printf("B_enc is:\n");
    for(int i = 0; i < cell_count; i++)
    {
        for(int j = 0; j < m; j++)
        {
            if( ((B_enc[i] >> j) & 0x01) == 1)
                printf("-1, ");
            else
                printf("1, ");
        }
        printf("\n");
    }
    printf("\n");
}


// DCE_MU: Differential Cell Enumeration for the new child terms generated by MU
// term must be a new child
// new children point to their parent term
// their will be a seg fault / invalid memory access if an old term in placed here
void make_new_child_btable(CauchyTerm* term,
    BKEYS B_mu, int cells_parent,
    KeyValue* B_mu_hash, int size_B_mu_hash, 
    KeyValue* B_coal_hash, int size_B_coal_hash, 
    BKEYS B_uncoal, bool* F)
{
    int m = term->m;
    int d = term->d;
    if(m <= d)
    {
        // Elementary Btable
        int child_cells;
        if(FULL_STORAGE)
            child_cells = 1 << m;
        else
            child_cells = 1 << (m-1);
        BKEYS enc_B = term->enc_B;
        term->cells_gtable = child_cells;
        for(int i = 0; i < child_cells; i++)
            enc_B[i] = i;
        return;
    }
    
    int pbc = term->pbc;
    int bit_mask[pbc];
    
    uint8_t* c_map = term->c_map;
    memset(B_mu_hash, kByteEmpty, size_B_mu_hash * sizeof(KeyValue));
    memset(F, 1, cells_parent * sizeof(bool));

    // Step 1: Hash all Sign-Vectors of B_mu
    BKEYS B = term->enc_B;
    int count_B = 0;
    KeyValue* kv_query;
    KeyValue kv;
    int z = term->z;
    // If child is not coaligned, then we can directly fill enc_B with the result of the alg
    bool is_child_coaligned = m < pbc;
    BKEYS Buc = is_child_coaligned ? B_uncoal : B;
    // Insert all sign vectors in B_mu into the temporary hashtable
    for(int j = 0; j < cells_parent; j++)
    {
        kv.key = B_mu[j];
        kv.value = j;
        if( hashtable_insert(B_mu_hash, &kv, size_B_mu_hash) )
        {
            printf(RED"[ERROR #1 DCE_MU:] Error when inserting value into B_mu_hash hashtable. Debug here! Exiting!" NC"\n");
            exit(1);
        }
    }

    // Take the parent sign vector (psv) of length m, and manipulates the integer to create the child sign vectors (csv1, csv2) for the z-th child
    int shift_high = z + 1;
    int shift_z = pbc - 1;
    int mask_low = (1 << z) - 1;
    int mask_z = (1 << z);
    int mask_hbit = (1 << shift_z);
    int rev_pbc_sv = (1<<pbc) - 1;
    int z_bit, csv1, csv2;
    for(int j = 0; j < cells_parent; j++)
    {
        if(F[j])
        {
            int b = B_mu[j];
            int b_query = b ^ mask_z;
            // Make sure b_query is reversed if using half storage
            if(HALF_STORAGE)
            {
                if(b_query & mask_hbit)
                    b_query ^= rev_pbc_sv;
            }
            if( hashtable_find(B_mu_hash, &kv_query, b_query, size_B_mu_hash) )
            {
                printf(RED"[ERROR #2 DCE_MU:] Error when finding value in B_mu_hash hashtable. Debug here! Exiting!" NC"\n");
                exit(1);
            }
            if(kv_query != NULL)
            {
                F[j] = 0;
                F[kv_query->value] = 0;
                z_bit = (b & mask_z) >> z; // extract z-th bit
                csv1 = ((b >> shift_high) << z) | (b & mask_low) | (z_bit << shift_z);
                csv2 = csv1 ^ mask_hbit;
                // If storage method is FULL, store both
                if(FULL_STORAGE)
                {
                    Buc[count_B++] = csv1;
                    Buc[count_B++] = csv2;
                }
                // For half storage, only store HPs in the positive halfspace of the last HP
                // Here this is convienient, since the hyperplane -mu_z is at the last position 
                // We simply just need to check if the last bit of csv1 is positive and we can store it
                // If the last bit is not positive, then we can store csv2
                else
                {   
                    // Store csv1 if csv1's pbc-th bit is not set
                    // Otherwise, store its opposite
                    if(csv1 & mask_hbit)
                        Buc[count_B++] = csv1 ^ rev_pbc_sv;
                    else 
                        Buc[count_B++] = csv1;
                    // Store csv2 if csv2's pbc-th bit is not set
                    // Otherwise, store its opposite
                    if(csv2 & mask_hbit)
                        Buc[count_B++] = csv2 ^ rev_pbc_sv;
                    else 
                        Buc[count_B++] = csv2;
                }
            }
        }
    }

    // Only enter coalignment section if there is coalignment
    if(is_child_coaligned)
    {
        // Step 2: Use c_map and F to make a mask of bits to select non-coaligned signs from B_p
        memset(F, 1, pbc * sizeof(bool));
        memset(B_coal_hash, kByteEmpty, size_B_coal_hash * sizeof(KeyValue));
        int count_coal = 0;
        for(int j = 0; j < pbc; j++)
        {
            int c_idx = c_map[j];
            if(F[c_idx])
            {
                F[c_idx] = 0;
                bit_mask[count_coal] = (1<<j);
                count_coal++;
            }
        }             
        // Step 3: Coalign Buc into B_coal
        int bc;
        int count_Bc = 0;
        kv.value = 0;
        for(int j = 0; j < count_B; j++)
        {
            bc = 0;
            int b = Buc[j];
            for(int l = 0; l < count_coal; l++)
                if( b & bit_mask[l] )
                    bc |= (1 << l);
            // If bc is not in the coaligned hash table
            if( hashtable_find(B_coal_hash, &kv_query, bc, size_B_coal_hash) )
            {
                printf(RED"[ERROR #3 DCE_MU:] Error when finding value in B_coal_hash hashtable. Debug here! Exiting!" NC"\n");
                exit(1);
            }
            if( kv_query == NULL )
            {
                kv.key = bc;
                if( hashtable_insert(B_coal_hash, &kv, size_B_coal_hash) )
                {
                    printf(RED"[ERROR #4 DCE_MU:] Error when inserting value into B_coal_hash hashtable. Debug here! Exiting!" NC"\n");
                    exit(1);
                }
                B[count_Bc++] = bc;
            }
        }
        term->cells_gtable = count_Bc;
    }
    else
        term->cells_gtable = count_B;
}


struct DiffCellEnumHelper
{
    // Variables for MU DCE
    KeyValue *B_mu_hash;
    int size_B_mu_hash;
    KeyValue *B_coal_hash; 
    int size_B_coal_hash;
    int* B_uncoal;
    int* cell_counts_cen;
    int max_shape;
    int d;
    int storage_multiplier;

    // Variables for TP DCE
    int** combos;
    int* combo_counts;
    int** anti_combos;

    int** combos_fast;
    int* combo_counts_fast;
    int** anti_combos_fast;

    int* cell_counts_gen;
    int* SSav;
    int** SSnav; // Sign Sequences not around vertex

    double* b_pert;

    bool* F; // used for masking in both TP and MU sections

    void init(int _max_shape, int _d, int _storage_multiplier)
    {   
        if(_max_shape > 28)
            printf(YEL "[WARNING DCE HELPER:] F is a mask defined as 2^max_shape for speed during lookups (to avoid a hashtable).\n" 
                   YEL "Consider adding a better masking data structure. (2^32 == 4GB)" NC "\n");

        max_shape = _max_shape;
        d = _d;
        // These storage places are used in both MU and TP 
        // TP requires at most cells_gen[m_max], so they are initialized to maximum size
        storage_multiplier = _storage_multiplier;
        int cells_max = cell_count_general(max_shape, _d);
        size_B_mu_hash = cells_max * storage_multiplier;
        size_B_coal_hash = cells_max * storage_multiplier;
        B_mu_hash = (KeyValue*) malloc(size_B_mu_hash * sizeof(KeyValue));
        null_ptr_check(B_mu_hash);
        B_coal_hash = (KeyValue*) malloc(size_B_coal_hash * sizeof(KeyValue));
        null_ptr_check(B_coal_hash);
        B_uncoal = (int*) malloc(cells_max * sizeof(int) );
        null_ptr_check(B_uncoal);
        F = (bool*) malloc(( 1<<max_shape) * sizeof(bool) );
        null_ptr_check(F);
        cell_counts_cen = (int*) malloc((max_shape+1)*sizeof(int));
        null_ptr_check(cell_counts_cen);
        cell_counts_gen = (int*) malloc((max_shape+1)*sizeof(int));
        null_ptr_check(cell_counts_gen);

        for(int i = 0; i < max_shape+1; i++)
        {
            cell_counts_cen[i] = cell_count_central(i, d);
            cell_counts_gen[i] = cell_count_general(i, d);
        }
        make_tp_dce_helpers();
    }

    void make_tp_dce_helpers()
    {
        int shape_range = max_shape + 1;
        SSav = init_encoded_sign_sequences_around_vertex(d);
        if(!FAST_TP_DCE)
        {
            combos = (int**) malloc(shape_range * sizeof(int*));
            null_dptr_check((void**)combos);
            anti_combos = (int**) malloc(shape_range * sizeof(int*));
            null_dptr_check((void**)anti_combos);
            combo_counts = (int*) malloc(shape_range * sizeof(int));
            null_ptr_check(combo_counts);
            
            // Helpers for regular TP DCE
            // TP DCE method is not used when the count of HPs are less than d
            for(int i = 0; i <= d; i++)
            {
                combos[i] = (int*) malloc(0);
                anti_combos[i] = (int*) malloc(0);
                combo_counts[i] = 0;
            }
            for(int m = d+1; m < shape_range; m++)
            {
                combos[m] = combinations(m, d);
                combo_counts[m] = nchoosek(m,d);
                anti_combos[m] = init_anti_combos(combos[m], combo_counts[m], m, d);
            }
            // Define perturbations for Gamma HP and other HPS
            b_pert = (double*) malloc( max_shape * sizeof(double));
            null_ptr_check(b_pert);
            for(int i = 0; i < max_shape; i++)
                b_pert[i] = 2*random_uniform() - 1;
        }
        else 
        {
            init_encoded_sign_sequences_not_around_vertex();
            combos_fast = (int**) malloc(shape_range * sizeof(int*));
            null_dptr_check((void**)combos_fast);
            anti_combos_fast = (int**) malloc(shape_range * sizeof(int*));
            null_dptr_check((void**)anti_combos_fast);
            combo_counts_fast = (int*) malloc(shape_range * sizeof(int));
            null_ptr_check(combo_counts_fast);
            for(int i = 0; i < d; i++)
            {
                combos_fast[i] = (int*) malloc(0);
                anti_combos_fast[i] = (int*) malloc(0);
                combo_counts_fast[i] = 0;
            }
            for(int m = d; m < shape_range; m++)
            {
                combos_fast[m] = combinations(m, d-1);
                combo_counts_fast[m] = nchoosek(m,d-1);
                anti_combos_fast[m] = init_anti_combos(combos_fast[m], combo_counts_fast[m], m, d-1);
            }
        }
    }

    // Initialized encoded sign sequences for encircling a vertex, with -1 encoded as 1 and 1 encoded as 0
    // The arrays are returned sized (2^mi x mi), for mi in [1,...,max_shape-d]
    // Since d HPs form the vertex, we only need these helpers for max_shape - d HPs
    void init_encoded_sign_sequences_not_around_vertex()
    {
        int shape_range = max_shape + 1 - d;
        SSnav = (int**) malloc(shape_range * sizeof(int*));
        null_dptr_check((void**)SSnav);
        SSnav[0] = (int*) malloc(0);
        for(int i = 1; i < shape_range; i++)
        {
            int two_to_i = (1<<i);
            SSnav[i] = (int*) malloc( two_to_i * i * sizeof(int) );
            null_ptr_check(SSnav[i]);
            
            for(int j = 0; j < two_to_i; j++)
            {
                for(int k = 0; k < i; k++)
                    SSnav[i][j*i + k] = (j & (1 << k)) ? 1 : 0;
            }

        }
    }
    
    void print_tp_info()
    {
        int shape_range = max_shape + 1;
        for(int m = d+1-FAST_TP_DCE; m < shape_range; m++)
        {
            int* combos_m = combos[m];
            int combo_count = combo_counts[m];
            int* anti_combos_m = anti_combos[m];
            printf("Combos for m=%d has shape (%d x %d):\n", m, combo_count, d-FAST_TP_DCE);
            print_mat(combos_m, combo_count, d-FAST_TP_DCE);
            printf("Anti Combos for m=%d, has shape (%d x %d):\n", m, combo_count, m-d+FAST_TP_DCE);
            print_mat(anti_combos_m, combo_count, m-d+FAST_TP_DCE);
        }
    }
    
    
    void deinit()
    {
        free(B_mu_hash);
        free(B_coal_hash);
        free(B_uncoal);
        free(cell_counts_cen);

        free(cell_counts_gen);
        free(F);
        free(SSav);

        // De-init TP helpers
        int shape_range = max_shape + 1;
        if(!FAST_TP_DCE)
        {
            for(int m = 0; m < shape_range; m++)
            {
                free(combos[m]);
                free(anti_combos[m]);
            }
            free(combos);
            free(anti_combos);
            free(combo_counts);
            free(b_pert);
        }
        else 
        {
            for(int m = 0; m < shape_range; m++)
            {
                free(combos_fast[m]);
                free(anti_combos_fast[m]);
            }
            free(combos_fast);
            free(anti_combos_fast);
            free(combo_counts_fast);
            for(int m = 0; m < shape_range-d; m++)
                free(SSnav[m]);
            free(SSnav);
        }
    }

};

// For two terms HPAs A_i and A_j that are equal,
// but possibly with different orientation of their HP normals,
// this routine updates A_i's btable_i to be in A_j's orientation
// if btable_j is NULL, btable_i on return is compatible with A_j
// if btable_j != NULL, btable_j is formed on return 
void update_btable(double* A_i, BKEYS btable_i, double* A_j, BKEYS btable_j, const int num_cells, const int m, const int d)
{
  int sigma_enc = 0;
  int kd = 0;
  int k = 0;
  int l;
  while(k < m)
  {
    l = 0;
    while( (fabs(A_i[kd + l]) < REDUCTION_EPS) || (fabs(A_j[kd + l]) < REDUCTION_EPS) )
      l++;
    if( (A_i[kd + l] * A_j[kd + l]) < 0)
      sigma_enc |= (1<<k);
    k += 1;
    kd += d;
  }
  // Adding: If sigma_enc has '-1' in m-th bit, reverse sigma_enc
  if(HALF_STORAGE)
  {
      int rev_mask = (1<<m) - 1;
      int two_to_m_minus1 = (1<<(m-1));
      if(sigma_enc & two_to_m_minus1)
        sigma_enc ^= rev_mask;
  }
  // Only update if a HP orientation of A_j is fliiped negative w.r.t A_i
  
  if(btable_j == NULL)
  {
    if(sigma_enc)
      for(k = 0; k < num_cells; k++)
        btable_i[k] ^= sigma_enc;
  }
  else 
  {
    if(sigma_enc)
        for(k = 0; k < num_cells; k++)
            btable_j[k] = btable_i[k] ^ sigma_enc;
    else 
        memcpy(btable_j, btable_i, num_cells * sizeof(BKEYS_TYPE));
  }

}

void tp_enu_warnings_check(double* A, double* b, double* v, 
    int* combo, int* anti_combo,
    const int m, const int n, const int term_idx, const int combo_num)
{
    double warn_work[m];
    matvecmul(A, v, warn_work, m, n);
    sub_vecs(warn_work, b, m);

    double max_resid_combo = 0;
    for(int i = 0; i < n; i++)
    {
        double w = fabs(warn_work[combo[i]]);
        if(w > max_resid_combo)
            max_resid_combo = w;
    }
    double min_resid_anti_combo = 1e100;
    int mn = m-n;
    for(int i = 0; i < mn; i++)
    {
        double w = fabs(warn_work[anti_combo[i]]);
        if(w < min_resid_anti_combo)
            min_resid_anti_combo = w;
    }
    if(min_resid_anti_combo < max_resid_combo)
    {
        printf(YEL "--------------- [FAST TP ENUMERATION WARN:] ---------------\n");
        printf("Shape: %d, Term Idx: %d\n", m, term_idx);
        printf("Combo #%d= ", combo_num);
        for(int i = 0; i < n; i++)
            printf("%d, ", combo[i]);
        printf("\n");
        printf("Anti Combo %d= ", combo_num);
        for(int i = 0; i < mn; i++)
            printf("%d, ", anti_combo[i]);
        printf("\n");      
        printf("min_resid_anti_combo is not greater then max_resid_combo\n");
        printf("min_resid_anti_combo = max(abs(A[anti_combo,:] @ v - b[anti_combo))=%.4E\n", min_resid_anti_combo);
        printf("max_resid_combo = max(abs(A[combo,:] @ v - b[combo))=%.4E\n", max_resid_combo);
        printf("This error indicates the resultant sign-vectors are likely undefined!\n");
        printf("-----------------------------------------------------------" NC "\n");
        if(EXIT_ON_FAILURE)
            exit(1);
    }
}

void make_time_prop_btable(CauchyTerm* term, DiffCellEnumHelper* dce_helper)
{
    // If shape < d, return the trivial set
    const int m = term->m;
    const int d = term->d;
    if(m<d)
    {
        int two_to_m = FULL_STORAGE ? 1<<m : 1<<(m-1);
        for(int i = 0; i < two_to_m; i++)
            term->enc_B[i] = i;
        return;
    }
    // Otherwise, begin routine
    int phc = term->phc;
    int* combos = dce_helper->combos[m];
    int combo_counts = dce_helper->combo_counts[m];
    int* anti_combos = dce_helper->anti_combos[m];
    int num_anti_combo = m - d;
    double Ac[d*d];
    double work[d*d]; // workspace for Ac when solving
    int P[d]; // Perm. Matrix for PLU solving
    double bc[d];
    double vertex[d];
    double b_pert[m];
    memcpy(b_pert, dce_helper->b_pert, m * sizeof(double));
    double* A = term->A;
    int two_to_d = 1 << d;
    // We need to find all sign vectors of sgn(A_tp* \nu - b), where \Gamma are the last (cmcc) HPs in A_tp
    // Since we have free reign to choose the offset b, we can choose it as [0^m_i^{k-1|k-1} ; -vec(eps)^cmcc], where eps >0
    // Doing so, B_parent can be immediately added to our hash set
    // the gtable_ps also already contain a hashtable over the parent_Bs

    // insert the SVs into Btp_hash, mark as visited
    const int cells_parent = term->cells_gtable_p;
    const int cells_gen = dce_helper->cell_counts_gen[m];
    const int Btp_hash_table_size = cells_gen * dce_helper->storage_multiplier;
    BTABLE Btp_hash = dce_helper->B_mu_hash; // renaming temp space
    BKEYS Btp_intermediate = dce_helper->B_uncoal; // renaming temp space
    bool* F = dce_helper->F;
    int gtable_p_table_size = cells_parent * GTABLE_SIZE_MULTIPLIER;
    GTABLE gtable_p = term->gtable_p;
    const int two_to_phc_minus1 = 1<<(phc-1);
    const int two_to_m_minus1 = 1<<(m-1);
    const int rev_phc_mask = (1<<phc)-1; // reverses sv of length phc with xor operator
    const int rev_m_mask = (1<<m)-1; // reverses sv of length m with xor operator
    int gamma_mask = rev_phc_mask; // clears gamma bits with and operator

    memset(Btp_hash, kByteEmpty, Btp_hash_table_size * sizeof(BTABLE_TYPE));
    memset(F, 1, (1<<m) * sizeof(bool) );

    int count_Btp_set = 0;
    KeyValue kv;
    // We need to loop through the combinations of Gamma, find vertices, and encircle them
    KeyCValue* kcv_query;
    for(int i = 0; i < combo_counts; i++)
    {
        // Form a vertex point corresponding to Gamma 
        int* combo = combos + i*d;
        for(int j = 0; j < d; j++)
        {
            memcpy(Ac + j * d, A + combo[j] * d, d*sizeof(double));
            bc[j] = b_pert[combo[j]];
        }
        // Solve for vertex:
        if(STABLE_SOLVE)
        {
            double cond_num = cond('1', Ac, work, P, d, PLU_EPS);
            if(cond_num > COND_EPS)
                continue;
            solve_trf(Ac, P, bc, vertex, d);
        }
        else
        {
            if( PLU(Ac, P, d, PLU_EPS) )
                continue;
            if( (fabs(Ac[0]) / (fabs(Ac[d*d-1]) + 1e-25)) > COND_EPS)
                continue;
            solve_trf(Ac, P, bc, vertex, d);
        }
        // Find signs of HPs not in vertex
        int* anti_combo = anti_combos + i * num_anti_combo;
        if(WITH_WARNINGS)
            tp_enu_warnings_check(A, b_pert, vertex, combo, anti_combo, m, d, -1, i);
        int enc_sv_niv = 0;
        for(int j = 0; j < num_anti_combo; j++)
        {
            int ac = anti_combo[j];
            if( (dot_prod(A + ac*d, vertex, d) - b_pert[ac]) < 0 )
                enc_sv_niv |= (1 << ac);
        }
        // Use sign sequences around vertex array to fill in all signs around  
        for(int j = 0; j < two_to_d; j++)
        {
            int* ssav = dce_helper->SSav + j*d;
            int enc_sv = enc_sv_niv;
            for(int k = 0; k < d; k++)
                if(ssav[k])
                    enc_sv |= (1 << combo[k]);
            // Only process this sign vector if it has not yet been processed
            if(F[enc_sv])
            {
                F[enc_sv] = 0;
                // Obtain the partial sign sequence without the sign of Gamma
                int enc_psv = enc_sv & gamma_mask; //int enc_psv
                if(HALF_STORAGE)
                {
                    if(enc_psv & two_to_phc_minus1)
                        enc_psv ^= rev_phc_mask;
                }
                // Check 1. Must have an enc_psv which is a member of the parent term set Bpar_set
                gtable_p_find(gtable_p, &kcv_query, enc_psv, gtable_p_table_size, cells_parent);
                if(kcv_query != NULL)
                {
                    // Add this sign vector since enc_psv is a member of the parent set
                    kv.key = enc_sv;
                    kv.value = count_Btp_set;
                    if( hashtable_insert(Btp_hash, &kv, Btp_hash_table_size) )
                    {
                        printf(RED "[Error TP DCE #4]: hashtable find has failed! Debug here!" NC "\n");
                        exit(1);
                    }
                    Btp_intermediate[count_Btp_set++] = enc_sv;
                    assert(count_Btp_set <= cells_gen);
                }
            }
        }
        if(count_Btp_set == cells_gen)
            break;
    }
    // Begin Check 2.) All sign vectors must have an opposite sign vector
    memset(F, 1, count_Btp_set);
    KeyValue* kv_query;
    int count_Btp = 0;
    int* Btp = term->enc_B;
    for(int i = 0; i < count_Btp_set; i++)
    {
        if(F[i])
        {
            int b = Btp_intermediate[i];
            int b_rev = b ^ rev_m_mask;
            if( hashtable_find(Btp_hash, &kv_query, b_rev, Btp_hash_table_size) )
            {
                printf(RED "[Error TP DCE #5]: hashtable find has failed! Debug here!" NC "\n");
                exit(1);
            }

            // If its opposite has been found, add them to Bprop_pred. Mark both in F
            if(kv_query != NULL)
            {
                F[i] = 0; // dont really need to mark this, I think
                F[kv_query->value] = 0;
                if(FULL_STORAGE)
                {
                    Btp[count_Btp++] = b;
                    Btp[count_Btp++] = b_rev;
                }
                else
                {
                    if(b & two_to_m_minus1)
                        Btp[count_Btp++] = b_rev;
                    else 
                        Btp[count_Btp++] = b;
                }
            }
        }
    }
    assert(count_Btp <= (dce_helper->cell_counts_cen[m] / (1 + HALF_STORAGE)) );
    term->cells_gtable = count_Btp;
}

// Moshes method of TP DCE by only shifting Gamma
// Called if the FAST_TP_DCE setting is set true
// If there are multiple Gammas, the algorithm is applied sequentially to each Gamma column
void make_time_prop_btable_fast(CauchyTerm* term, DiffCellEnumHelper* dce_helper)
{
    // If shape < d, return the trivial set
    if(term->m < term->d)
    {
        int two_to_m = FULL_STORAGE ? 1 << term->m : 1<<(term->m-1);
        for(int i = 0; i < two_to_m; i++)
            term->enc_B[i] = i;
        return;
    }
    // Otherwise, begin routine
    int d = term->d;
    int dm1 = d-1;
    double Ac[d*d];
    double work[d*d]; // workspace for Ac when solving
    int P[d]; // Perm. Matrix for PLU solving
    double bc[d];
    double vertex[d];
    double* A = term->A;
    memset(bc, 0, dm1*sizeof(double));
    bc[dm1] = -GAMMA_PERTURB_EPS;
    int two_to_d_minus1 = 1 << dm1;
    int cells_parent = term->cells_gtable_p;
    int* Btp = term->enc_B;
    BTABLE Btp_hash = dce_helper->B_mu_hash; // renaming temp space
    BKEYS Btp_intermediate = dce_helper->B_uncoal; // renaming temp space
    bool* F = dce_helper->F; // Flags for unvisited sign vectors

    // Loop until all columns of Gamma added to the HPA have been dealt with
    // Start with the parent hyperplanes, adding columns of Gamma one at a time
    int count_Gam = 0;
    for(int phc = term->phc; phc < term->m; phc++)
    {
        const int m = phc+1; // m is always 1 more than the number of "parent hyperplanes" (due to the addition of the new column of Gamma)
        const int* combos = dce_helper->combos_fast[phc];
        const int combo_counts = dce_helper->combo_counts_fast[phc];
        const int* anti_combos = dce_helper->anti_combos_fast[phc];
        const int num_anti_combo = m - d;
        const int cells_gen = dce_helper->cell_counts_gen[m];
        const int Btp_hash_table_size = cells_gen * dce_helper->storage_multiplier;
        const int parent_table_size = cells_parent * GTABLE_SIZE_MULTIPLIER;
        const int two_to_m_minus1 = 1<<(m-1);
        const int rev_phc_mask = (1<<phc)-1; // reverses sv of length phc with xor operator
        const int rev_m_mask = (1<<m)-1; // reverses sv of length m with xor operator
        const int gamma_mask = rev_phc_mask; // clears gamma bits with and operator
        int zero_resids[m];
        memcpy(Ac + dm1*d, A + phc*d, d*sizeof(double)); // last HP in Ac is always constant (Gamma)
        memset(Btp_hash, kByteEmpty, Btp_hash_table_size * sizeof(BTABLE_TYPE));
        memset(F, 1, (1<<m) * sizeof(bool) );

        int count_Btp_set = 0;
        KeyValue kv;
        // The parent table of SVs is immediately added to our hash set
        // This is because we shift Gamma to have a positive halfspace w.r.t the origin
        // Therefore, all parent SVs are inserted into this hashtable
        // If count_Gam == 0, extract the keys of gtable_p, otherwise, the keys are already in Btp
        if(count_Gam == 0)
            gtable_p_get_keys(term->gtable_p, parent_table_size, Btp); // returns cells_parent, which is already loaded
        
        for(int i = 0; i < cells_parent; i++)
        {        
            int b_enc = Btp[i];
            F[b_enc] = 0;
            kv.key = b_enc;
            kv.value = count_Btp_set;
            Btp_intermediate[count_Btp_set++] = b_enc;
            if( hashtable_insert(Btp_hash, &kv, Btp_hash_table_size) )
            {
                printf(RED "[Error TP DCE #1]: hashtable insert has failed! Debug here!" NC "\n");
                exit(1);
            }
            // If we used half storage, store the opposite temporarily 
            if(HALF_STORAGE)
            {
                b_enc ^= rev_phc_mask;
                F[b_enc] = 0;
                kv.key = b_enc;
                kv.value = count_Btp_set;
                Btp_intermediate[count_Btp_set++] = b_enc;
                if( hashtable_insert(Btp_hash, &kv, Btp_hash_table_size) )
                {
                    printf(RED "[Error TP DCE #2]: hashtable insert has failed! Debug here!" NC "\n");
                    exit(1);
                }
            }
        }

        // Now we need to loop through all combinations of (m-1) choose (d-1), find vertices, encircle them
        KeyValue* kv_query;
        for(int i = 0; i < combo_counts; i++)
        {
            // Form a vertex point corresponding to Gamma 
            const int* combo = combos + i*dm1;
            for(int j = 0; j < dm1; j++)
                memcpy(Ac + j * d, A + combo[j] * d, d*sizeof(double));
            memcpy(Ac + dm1*d, A + phc*d, d*sizeof(double));
            // Solve for vertex:
            if(STABLE_SOLVE)
            {
                double cond_num = cond('1', Ac, work, P, d, PLU_EPS);
                if(cond_num > COND_EPS)
                    continue;
                solve_trf(Ac, P, bc, vertex, d);
            }
            else
            {
                if( PLU(Ac, P, d, PLU_EPS) )
                    continue;
                if( (fabs(Ac[0]) / (fabs(Ac[d*d-1]) + 1e-25)) > COND_EPS)
                    continue;
                solve_trf(Ac, P, bc, vertex, d);
            }
            // Find signs of HPs not in vertex
            const int* anti_combo = anti_combos + i * num_anti_combo;
            //if(WITH_WARNINGS)
            //    fast_tp_enu_warnings_check(A, vertex, combo, anti_combo, m, d, -1, i);
            int enc_sv_niv = 1 << phc;
            int zero_resid_count = 0;
            for(int j = 0; j < num_anti_combo; j++)
            {
                int ac = anti_combo[j];
                double resid = dot_prod(A + ac*d, vertex, d);
                if( fabs(resid) < 1e-10 )
                    zero_resids[zero_resid_count++] = ac;
                else if( resid < 0 )
                    enc_sv_niv |= (1 << ac);
            }
            int two_to_zero_resid_count = (1 << zero_resid_count);
            int* SSnav = dce_helper->SSnav[zero_resid_count];
            // Use sign sequences around vertex array to fill in all signs around  
            for(int j = 0; j < two_to_d_minus1; j++)
            {
                int* ssav = dce_helper->SSav + j*d;
                int enc_sv = enc_sv_niv;
                for(int k = 0; k < dm1; k++)
                    if(ssav[k])
                        enc_sv |= (1 << combo[k]);
                
                // Add \pm1 to the encirclement if the residuals for 'niv' HPS are zero
                if(zero_resid_count > 0)
                {
                    for(int k = 0; k < two_to_zero_resid_count; k++)
                    {   
                        int enc_sv_r = enc_sv;
                        int* ssnav = SSnav + k*zero_resid_count;
                        for(int l = 0; l < zero_resid_count; l++)
                            if(ssnav[l])
                                enc_sv_r |= (1 << zero_resids[l]);

                        if(F[enc_sv_r])
                        {
                            F[enc_sv_r] = 0;
                            // Obtain the partial sign sequence without the sign of Gamma
                            int enc_psv = enc_sv_r & gamma_mask; //int enc_psv
                            if(hashtable_find(Btp_hash, &kv_query, enc_psv, Btp_hash_table_size))
                            {
                                printf(RED "[Error TP DCE #3]: hashtable find has failed! Debug here!" NC "\n");
                                exit(1);
                            }
                            if(kv_query != NULL)
                            {
                                // Add this sign vector since enc_psv is a member of the parent set
                                kv.key = enc_sv_r;
                                kv.value = count_Btp_set;
                                if( hashtable_insert(Btp_hash, &kv, Btp_hash_table_size) )
                                {
                                    printf(RED "[Error TP DCE #4]: hashtable insert has failed! Debug here!" NC "\n");
                                    exit(1);
                                }
                                Btp_intermediate[count_Btp_set++] = enc_sv_r;
                                assert(count_Btp_set <= cells_gen);
                            }
                        }
                    }
                }
                // Only process this sign vector if it has not yet been processed
                else if(F[enc_sv])
                {
                    F[enc_sv] = 0;
                    // Obtain the partial sign sequence without the sign of Gamma
                    int enc_psv = enc_sv & gamma_mask; //int enc_psv
                    if(hashtable_find(Btp_hash, &kv_query, enc_psv, Btp_hash_table_size))
                    {
                        printf(RED "[Error TP DCE #5]: hashtable find has failed! Debug here!" NC "\n");
                        exit(1);
                    }
                    if(kv_query != NULL)
                    {
                        // Add this sign vector since enc_psv is a member of the parent set
                        kv.key = enc_sv;
                        kv.value = count_Btp_set;
                        if( hashtable_insert(Btp_hash, &kv, Btp_hash_table_size) )
                        {
                            printf(RED "[Error TP DCE #6]: hashtable insert has failed! Debug here!" NC "\n");
                            exit(1);
                        }
                        Btp_intermediate[count_Btp_set++] = enc_sv;
                        assert(count_Btp_set <= cells_gen);
                    }
                }
            }
            if(count_Btp_set == cells_gen)
                break;
        }
        // Begin Check 2.) All sign vectors must have an opposite sign vector
        memset(F, 1, count_Btp_set);
        int count_Btp = 0;
        for(int i = 0; i < count_Btp_set; i++)
        {
            if(F[i])
            {
                int b = Btp_intermediate[i];
                int b_rev = b ^ rev_m_mask;
                if( hashtable_find(Btp_hash, &kv_query, b_rev, Btp_hash_table_size) )
                {
                    printf(RED "[Error TP DCE #5]: hashtable find has failed! Debug here!" NC "\n");
                    exit(1);
                }

                // If its opposite has been found, add them to Bprop_pred. Mark both in F
                if(kv_query != NULL)
                {
                    F[i] = 0; // dont really need to mark this, I think
                    F[kv_query->value] = 0;
                    if(FULL_STORAGE)
                    {
                        Btp[count_Btp++] = b;
                        Btp[count_Btp++] = b_rev;
                    }
                    else
                    {
                        if(b & two_to_m_minus1)
                            Btp[count_Btp++] = b_rev;
                        else 
                            Btp[count_Btp++] = b;
                    }
                }
            }
        }
        cells_parent = count_Btp;
        assert(count_Btp <= (dce_helper->cell_counts_cen[m] / (1 + HALF_STORAGE)) );
        count_Gam++;
        term->cells_gtable = count_Btp;
    }
}

// Function to propagate the parent enumeration matrix to the child's enumeration matrix 
void make_lowered_child_btable(CauchyTerm* term,
    BKEYS B_mu, int cells_parent,
    KeyValue* B_mu_hash, int size_B_mu_hash, 
    KeyValue* B_coal_hash, int size_B_coal_hash, 
    BKEYS B_uncoal, bool* F)
{
    // Make sure m > d...otherwise the lower child should just be removed
    assert(term->m >= term->d);
    // If m == d, all we need to do is assign the trivial set of sign vectors to it
    if(term->m == term->d)
    {
        int num_cells = (1<<term->d) / (1 + HALF_STORAGE);
        for(int i = 0; i < num_cells; i++)
            term->enc_B[i] = i;
        term->cells_gtable = num_cells;
        return;
    }
    // Otherwise, run algorithm
    int phc = term->phc;
    BKEYS B = term->enc_B;
    uint8_t* c_map = term->c_map;
    int z = term->z;
    int enc_lhp = term->enc_lhp;
    int mask_phc = 1<<(phc-1);
    int rev_phc_sv = (1<<phc) - 1;
    // If enc_lhp has a '-1' w.r.t the last HP, need to reverse, so that B_mu[j] ^ enc_lhp remains in positive halfspace
    if(HALF_STORAGE)
    {
        if(enc_lhp & mask_phc)
            enc_lhp ^= rev_phc_sv;
    }
    
    memset(B_mu_hash, kByteEmpty, size_B_mu_hash * sizeof(KeyValue));
    memset(F, 1, cells_parent * sizeof(bool));
    int bit_mask[phc];
    bool is_child_coaligned = c_map != NULL;

    // Step 1: Hash all Sign-Vectors of B_mu
    // If child is not coaligned, then we can directly fill enc_B with the result of the alg
    KeyValue* kv_query;
    KeyValue kv;
    BKEYS Buc = is_child_coaligned ? B_uncoal : B;
    // Insert all sign vectors in B_mu into the temporary hashtable
    for(int j = 0; j < cells_parent; j++)
    {
        kv.key = B_mu[j] ^ enc_lhp;
        kv.value = j;
        if( hashtable_insert(B_mu_hash, &kv, size_B_mu_hash) )
        {
            printf(RED"[ERROR #1 DCE_MU:] Error when inserting value into B_mu_hash hashtable. Debug here! Exiting!" NC"\n");
            exit(1);
        }
    }

    // Take the parent sign vector (psv) of length m, and manipulates the integer to create the child sign vectors (csv1, csv2) for the z-th child
    int count_B = 0;
    bool is_last_child = (z == (phc-1));
    int mask_z = (1 << z);
    int csv1, csv2;
    for(int j = 0; j < cells_parent; j++)
    {
        if(F[j])
        {
            int b = B_mu[j] ^ enc_lhp;
            int b_query = b ^ mask_z;
            // Make sure b_query is reversed if using half storage
            if(HALF_STORAGE)
            {
                // The last child will look over and flip the last indices sign vectors, 
                // Therefore, all queried sign vectors (b_query) will be in negative halfspace w.r.t parent arrangement
                // Reverse the queried sign vector, look for it instead
                if(is_last_child)
                    b_query ^= rev_phc_sv;
            }
            if( hashtable_find(B_mu_hash, &kv_query, b_query, size_B_mu_hash) )
            {
                printf(RED"[ERROR #2 DCE_MU:] Error when finding value in B_mu_hash hashtable. Debug here! Exiting!" NC"\n");
                exit(1);
            }
            if(kv_query != NULL)
            {
                F[j] = 0;
                F[kv_query->value] = 0;
                csv1 = remove_bit(b, z);
                if(FULL_STORAGE)
                {
                    Buc[count_B++] = csv1;
                }
                else 
                {
                    // For last child, b and b_query (after removing the z-th bit) will now be opposites
                    // One of these opposites will be in the negative halfspace of the child's last hyperplane
                    // Only store the sign vector in the positive halfspace w.r.t the last hyperplane
                    // This will be the sign vector with lower encoded magnitude, store it
                    if(is_last_child)
                    {
                        csv2 = remove_bit(b_query, z);
                        assert( (csv1 & csv2) == 0); // must be opposites, or I have wrong thinking...
                        Buc[count_B++] = csv1 < csv2 ? csv1 : csv2;
                    }
                    else
                        Buc[count_B++] = csv1;
                }
            }
        }
    }

    // Only enter coalignment section if there is coalignment
    if(is_child_coaligned)
    {
        // Step 2: Use c_map and F to make a mask of bits to select non-coaligned signs from B_p
        memset(F, 1, phc * sizeof(bool));
        memset(B_coal_hash, kByteEmpty, size_B_coal_hash * sizeof(KeyValue));
        int count_coal = 0;
        for(int j = 0; j < phc; j++)
        {
            int c_idx = c_map[j];
            if(F[c_idx])
            {
                F[c_idx] = 0;
                bit_mask[count_coal] = (1<<j);
                count_coal++;
            }
        }             
        // Step 3: Coalign Buc into B_coal
        int bc;
        int count_Bc = 0;
        kv.value = 0;
        for(int j = 0; j < count_B; j++)
        {
            bc = 0;
            int b = Buc[j];
            for(int l = 0; l < count_coal; l++)
                if( b & bit_mask[l] )
                    bc |= (1 << l);
            // Check if bc is not in the coaligned hash table
            if( hashtable_find(B_coal_hash, &kv_query, bc, size_B_coal_hash) )
            {
                printf(RED"[ERROR #3 DCE_MU:] Error when finding value in B_coal_hash hashtable. Debug here! Exiting!" NC"\n");
                exit(1);
            }
            // If bc is not in the coaligned hash table, add it, and add to final set
            if( kv_query == NULL )
            {
                kv.key = bc;
                if( hashtable_insert(B_coal_hash, &kv, size_B_coal_hash) )
                {
                    printf(RED"[ERROR #4 DCE_MU:] Error when inserting value into B_coal_hash hashtable. Debug here! Exiting!" NC"\n");
                    exit(1);
                }
                B[count_Bc++] = bc;
            }
        }
        term->cells_gtable = count_Bc;
    }
    else
        term->cells_gtable = count_B;
}

// General purposed 'inc-enu' like algorithm to solve a cell enumeration for a central arrangement
// Input: an m x d central hyperplane arrangement (stored row-wise), with m=#hyperplanes and d=dimension
// Output: returns a pointer to the enumeration array, which has *num_cells encoded sign-vectors
// Note: If you wish to view the non-encoded representation of the sign vectors, call print_B_unencoded() function
// If dce_helper == NULL, helpers (which are members of this structure) are created then cleaned up internally
// If dce_helper != NULL. helpers are already initialized, which saves some time
// If half_enum == false, the full encoded arrangement is returned
// If half_enum == true, only the encoded sign vectors in the positive halspace of the last hyperplane are returned (half the SVs)
// If warn_checks
int* make_enumeration_matrix(double* A, const int m, const int d, int* num_cells, DiffCellEnumHelper* dce_helper = NULL, const bool half_enum = false, const bool warn_checks = false)
{
    assert(m <= 32);
    if(m<=d)
    {
        int two_to_m = (1 << m) / (1 + half_enum);
        int* enc_B = (int*) malloc(two_to_m * sizeof(int));
        for(int i = 0; i < two_to_m; i++)
            enc_B[i] = i;
        
        return enc_B;
    }
    int cells_gen;
    int B_hash_table_size;
    BTABLE B_hash;
    BKEYS B_intermediate;
    bool* F;
    int* combos;
    int num_combos;
    int* anti_combos;
    int m_minus_d;
    int* SSav;
    bool is_dce_null;
    double* b_pert;
    if(dce_helper == NULL)
    {
        is_dce_null = true;
        cells_gen = cell_count_general(m, d);
        B_hash_table_size = cells_gen * 4;
        B_hash = (BTABLE) malloc( B_hash_table_size * sizeof(BTABLE_TYPE) );
        null_ptr_check(B_hash);
        memset(B_hash, kByteEmpty, B_hash_table_size * sizeof(BTABLE_TYPE));
        B_intermediate = (BKEYS) malloc(cells_gen * sizeof(BKEYS_TYPE));
        null_ptr_check(B_intermediate);
        F = (bool*) malloc( (1<<m) * sizeof(bool) );
        null_ptr_check(F);
        memset(F, 1, (1<<m) * sizeof(bool) );
        combos = combinations(m, d);
        num_combos = nchoosek(m, d);
        anti_combos = init_anti_combos(combos, num_combos, m, d);
        m_minus_d = m - d;
        SSav = init_encoded_sign_sequences_around_vertex(d);
        b_pert = (double*) malloc( m * sizeof(double) );
        null_ptr_check(b_pert);
        for(int i = 0; i < m; i++)
            b_pert[i] = 2*random_uniform() - 1;
    }
    else 
    {
        is_dce_null = false;
        cells_gen = dce_helper->cell_counts_gen[m];
        B_hash_table_size = cells_gen * dce_helper->storage_multiplier;
        B_hash = dce_helper->B_mu_hash; // renaming temp space
        B_intermediate = dce_helper->B_uncoal; // renaming temp space
        memset(B_hash, kByteEmpty, B_hash_table_size * sizeof(BTABLE_TYPE));
        F = dce_helper->F;
        memset(F, 1, (1<<m) * sizeof(bool) );
        combos = dce_helper->combos[m];
        num_combos = dce_helper->combo_counts[m];
        anti_combos = dce_helper->anti_combos[m];
        m_minus_d = m - d;
        SSav = dce_helper->SSav;
        b_pert = dce_helper->b_pert;
    }

    double Ac[d*d];
    double work[d*d];
    double bc[d];
    double vertex[d];
    int P[d];
    int count_B_set = 0;
    int two_to_d = 1<<d;
    KeyValue kv;
    // We need to loop through the combinations of Gamma, find vertices, and encircle them
    for(int i = 0; i < num_combos; i++)
    {
        // Form a vertex point corresponding to Gamma 
        int* combo = combos + i*d;
        for(int j = 0; j < d; j++)
        {
            memcpy(Ac + j * d, A + combo[j] * d, d*sizeof(double));
            bc[j] = b_pert[combo[j]];
        }
        // Solve for vertex:
        if(STABLE_SOLVE)
        {
            double cond_num = cond('1', Ac, work, P, d, PLU_EPS);
            if(cond_num > COND_EPS)
                continue;
            solve_trf(Ac, P, bc, vertex, d);
        }
        else
        {
            if( PLU(Ac, P, d, PLU_EPS) )
                continue;
            if( (fabs(Ac[0]) / (fabs(Ac[d*d-1]) + 1e-25)) > COND_EPS)
                continue;
            solve_trf(Ac, P, bc, vertex, d);
        }
        // Find signs of HPs not in vertex
        int* anti_combo = anti_combos + i * m_minus_d;
        if(warn_checks)
            tp_enu_warnings_check(A, b_pert, vertex, combo, anti_combo, m, d, -1, i);
        int enc_sv_niv = 0;
        for(int j = 0; j < m_minus_d; j++)
        {
            int ac = anti_combo[j];
            if( (dot_prod(A + ac*d, vertex, d) - b_pert[ac]) < 0 )
                enc_sv_niv |= (1 << ac);
        }
        // Use sign sequences around vertex array to fill in all signs around  
        for(int j = 0; j < two_to_d; j++)
        {
            int* ssav = SSav + j*d;
            int enc_sv = enc_sv_niv;
            for(int k = 0; k < d; k++)
                if(ssav[k])
                    enc_sv |= (1 << combo[k]);
            // Only process this sign vector if it has not yet been processed
            if(F[enc_sv])
            {
                F[enc_sv] = 0;
                kv.key = enc_sv;
                kv.value = count_B_set;
                if( hashtable_insert(B_hash, &kv, B_hash_table_size) )
                {
                    printf(RED "[Error Make Btable #1]: hashtable find has failed! Debug here!" NC "\n");
                    exit(1);
                }

                B_intermediate[count_B_set++] = enc_sv;
                assert(count_B_set <= cells_gen);
            }
        }
        if(count_B_set == cells_gen)
            break;
    }
    // All sign vectors of a central arrangement must have an opposite sign vector
    memset(F, 1, count_B_set);
    KeyValue* kv_query;
    int count_B = 0;
    int rev_m_mask = (1<<m) - 1;
    int mask_m = (1<<(m-1));
    int* enc_B = (int*) malloc( cells_gen * sizeof(int) );
    null_ptr_check(enc_B);
    for(int i = 0; i < count_B_set; i++)
    {
        if(F[i])
        {
            int b = B_intermediate[i];
            int b_rev = b ^ rev_m_mask;
            if( hashtable_find(B_hash, &kv_query, b_rev, B_hash_table_size) )
            {
                printf(RED "[Error Make Btable #2]: hashtable find has failed! Debug here!" NC "\n");
                exit(1);
            }

            // If its opposite has been found, add them to Bprop_pred. Mark both in F
            if(kv_query != NULL)
            {
                F[i] = 0; // dont really need to mark this, I think
                F[kv_query->value] = 0;
                if(half_enum)
                {
                    if(b & mask_m)
                        enc_B[count_B++] = b_rev;
                    else
                        enc_B[count_B++] = b;
                }
                else
                {
                    enc_B[count_B++] = b;
                    enc_B[count_B++] = b_rev;
                }
            }
        }
    }
    enc_B = (int*) realloc(enc_B, count_B * sizeof(int));
    null_ptr_check(enc_B);
    *num_cells = count_B;
    if(is_dce_null)
    {
        free(B_hash);
        free(B_intermediate);
        free(F);
        free(combos);
        free(anti_combos);
        free(SSav);
        free(b_pert);
    }
    return enc_B;
}


#endif