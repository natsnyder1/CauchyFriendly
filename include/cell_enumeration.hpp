#ifndef _CELL_ENUMERATION_HPP_
#define _CELL_ENUMERATION_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
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

int sort_func_B_enc(const void* p1, const void* p2)
{
  return *((int*)p1) - *((int*)p2);
}

void print_B_encoded(int *B_enc, int cell_count, int m, bool with_sort)
{
    if(with_sort)
    {    
        qsort(B_enc, cell_count, sizeof(int), &sort_func_B_enc);
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
        qsort(B_enc, cell_count, sizeof(int), &sort_func_B_enc);
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
    
    int pbc = term->parent->m;
    int bit_mask[pbc];
    int cells_parent = term->parent->cells_gtable;
    
    uint8_t* c_map = term->c_map;
    memset(B_mu_hash, kByteEmpty, size_B_mu_hash * sizeof(KeyValue));
    memset(F, 1, cells_parent * sizeof(bool));

    // Step 1: Hash all Sign-Vectors of B_mu
    BKEYS B = term->enc_B;
    BKEYS B_mu = term->parent->enc_B;
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
    int cmcc;
    int*** combos_of_Gam;
    int** counts_of_Gam;
    int*** anti_combos_of_Gam;
    int* cell_counts_gen;
    int* SSav;
    double* b_pert;

    bool* F; // used for masking in both TP and MU sections

    void init(int _max_shape, int _d, int _storage_multiplier, int _cmcc)
    {   
        if(_max_shape > 28)
            printf(YEL "[WARNING DCE HELPER:] F is a mask defined as 2^max_shape for speed during lookups (to avoid a hashtable).\n" 
                   YEL "Consider adding a better masking data structure. (2^32 == 4GB)" NC "\n");

        max_shape = _max_shape;
        d = _d;
        cmcc = _cmcc;
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
        init_encoded_sign_sequences_around_vertex();
        int cmcc_range = cmcc+1;
        combos_of_Gam = (int***) malloc(cmcc_range * sizeof(int**));
        null_tptr_check((void***)combos_of_Gam);
        anti_combos_of_Gam = (int***) malloc(cmcc_range * sizeof(int**));
        null_tptr_check((void***)anti_combos_of_Gam);
        counts_of_Gam = (int**) malloc(cmcc_range * sizeof(int*));
        null_dptr_check((void**)counts_of_Gam);
        // if cmcc = 0, nothing to do in TP routine
        combos_of_Gam[0] = (int**)malloc(0);
        anti_combos_of_Gam[0] = (int**)malloc(0);
        counts_of_Gam[0] = (int*)malloc(0);
        const int shape_range = max_shape+1;
        for(int c = 1; c < cmcc_range; c++)
        {
            combos_of_Gam[c] = (int**)malloc(shape_range * sizeof(int*));
            anti_combos_of_Gam[c] = (int**)malloc(shape_range * sizeof(int*));
            counts_of_Gam[c] = (int*) malloc(shape_range * sizeof(int));
            for(int i = 0; i < d; i++)
            {
                combos_of_Gam[c][i] = (int*) malloc(0);
                anti_combos_of_Gam[c][i] = (int*) malloc(0);
                counts_of_Gam[c][i] = 0;
            }
            for(int m = d; m < shape_range; m++)
            {
                int* combos = combinations(m, d);
                // If m=d or m <= c there are no anti combos
                if( ( m==d ) || ( m <= c ) )
                {
                    combos_of_Gam[c][m] = combos;
                    counts_of_Gam[c][m] = nchoosek(m,d);
                    anti_combos_of_Gam[c][m] = (int*) malloc(0);
                }
                else
                {
                    //int select_c[c]; // select last c indices
                    //for(int i = 0; i < c; i++)
                    //    select_c[i] = m-c+i;
                    int combo_counts = nchoosek(m,d);
                    //int counts_of_Gam_cm;
                    combos_of_Gam[c][m] = combos; //selective_combinations(combos, select_c, combo_counts, c, d, &counts_of_Gam_cm);
                    counts_of_Gam[c][m] = combo_counts; //counts_of_Gam_cm;
                    anti_combos_of_Gam[c][m] = init_anti_combos(combos_of_Gam[c][m], counts_of_Gam[c][m], m, d);
                    //free(combos);
                }
            }
        }
        // Define perturbations for Gamma HP and other HPS
        b_pert = (double*) malloc( max_shape * sizeof(double));
        for(int i = 0; i < max_shape; i++)
            b_pert[i] = 2*random_uniform() - 1;
        
    }

    // Returns encoded sign sequences for encircling a vertex, with -1 encoded as 1 and 1 encoded as 0
    // The array returned is sized (2^d x d)
    void init_encoded_sign_sequences_around_vertex()
    {
        int num_cells_around_vetex = (1 << d);
        SSav = (int*) malloc( num_cells_around_vetex * d * sizeof(int) ); // Indicator Sequences around vertex
        null_ptr_check(SSav);
        for(int i = 0; i < num_cells_around_vetex; i++)
            for(int j = 0; j < d; j++)
                SSav[i*d + j] = (i & (1 << j) ) >> j;
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

    // Returns only the combinations which contain the select indices
    int* selective_combinations(int* combos, int* select, int len_combos, int len_select, int d, int* len_selective_combos)
    {
        int select_counts = 0;
        int* select_combos = (int*) malloc(len_combos * d * sizeof(int));
        null_ptr_check(select_combos);
        for(int i = 0; i < len_combos; i++)
        {
            int* combo = combos + i*d;
            bool is_select_in_combo = false;
            for(int j = 0; j < d; j++)
            {
                int c = combo[j];
                for(int k = 0; k < len_select; k++)
                {
                    if(c == select[k])
                    {
                        is_select_in_combo = true;
                        break;
                    }
                }
                if(is_select_in_combo)
                    break;
            }
            if(is_select_in_combo)
            {
                memcpy(select_combos + select_counts*d, combo, d * sizeof(int));
                select_counts++;
            }
        }
        select_combos = (int*) realloc(select_combos, select_counts * d * sizeof(int));
        null_ptr_check(select_combos);
        *len_selective_combos = select_counts;
        return select_combos;
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
    
    void print_tp_info()
    {
        int cmcc_range = cmcc + 1;
        int shape_range = max_shape + 1;
        for(int c = 1; c < cmcc_range; c++)
        {
            for(int m = d; m < shape_range; m++)
            {
                int* cG = combos_of_Gam[c][m];
                int num_cG = counts_of_Gam[c][m];
                int* acG = anti_combos_of_Gam[c][m];
                int num_acG = m - d;
                printf("Combos of Gamma for m=%d, cmcc=%d, has shape (%d x %d):\n", m, c, num_cG, d);
                print_mat(cG, num_cG, d);
                if( (num_acG > 0) && (m > c ) )
                {
                    printf("Combos of Anti Gamma for m=%d, cmcc=%d, has shape (%d x %d):\n", m, c, num_cG, m-d);
                    print_mat(acG, num_cG, m-d);
                }
                else
                {
                    printf("No anti combos for m=%d, cmcc=%d, d=%d\n", m, c, d);
                }
            }
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

        // De-init TP helpers
        free(SSav);
        int cmcc_range = cmcc + 1;
        for(int c = 1; c < cmcc_range; c++)
        {
            int shape_range = max_shape + 1;
            for(int m = 0; m < shape_range; m++)
            {
                free(combos_of_Gam[c][m]);
                free(anti_combos_of_Gam[c][m]);
            }
            free(combos_of_Gam[c]);
            free(anti_combos_of_Gam[c]);
            free(counts_of_Gam[c]);
        }
        free(combos_of_Gam[0]);
        free(anti_combos_of_Gam[0]);
        free(counts_of_Gam[0]);
        free(combos_of_Gam);
        free(anti_combos_of_Gam);
        free(counts_of_Gam);
        free(b_pert);
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

void make_time_prop_btable(BKEYS B_parent, CauchyTerm* term, DiffCellEnumHelper* dce_helper)
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
    int num_cmcc = m - phc;
    int* combos_of_Gam = dce_helper->combos_of_Gam[num_cmcc][m];
    int counts_of_Gam = dce_helper->counts_of_Gam[num_cmcc][m];
    int* anti_combos_of_Gam = dce_helper->anti_combos_of_Gam[num_cmcc][m];
    int num_anti_combo = m - d;
    double Ac[d*d];
    double work[d*d]; // workspace for Ac when solving
    int P[d]; // Perm. Matrix for PLU solving
    double bc[d];
    double vertex[d];
    double b_pert[m];
    //memset(b_pert, 0, m*sizeof(double));
    memcpy(b_pert, dce_helper->b_pert, m * sizeof(double));
    //for(int i = 0; i < num_cmcc; i++)
        //b_pert[m-num_cmcc+i] = -GAMMA_PERTURB_EPS;
        //b_pert[m-num_cmcc+i] = -fabs(b_pert[m-num_cmcc+i]);

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
    /* Moshes way not working...reverting to old way
    for(int i = 0; i < cells_parent; i++)
    {
        int b_enc = B_parent[i];
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
    */
    // We need to loop through the combinations of Gamma, find vertices, and encircle them
    KeyCValue* kcv_query;
    for(int i = 0; i < counts_of_Gam; i++)
    {
        // Form a vertex point corresponding to Gamma 
        int* combo = combos_of_Gam + i*d;
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
        int* anti_combo = anti_combos_of_Gam + i * num_anti_combo;
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

#endif