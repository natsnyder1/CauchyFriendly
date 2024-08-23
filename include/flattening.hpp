#ifndef _FLATTENING_HPP_
#define _FLATTENING_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cell_enumeration.hpp"
#include "cpu_timer.hpp"
#include "eval_gs.hpp"

//// HASHTABLE
// Returns G values per face of each term.
void make_gtable_first(CauchyTerm* term, const double G_SCALE_FACTOR)
{
  const int m = term->d;
  int rev_mask = (1<<m)-1;
  double sign_b[m];
  double ygi; 
  uint Horthog_flag = term->Horthog_flag;
  double* p = term->p;
  GTABLE gtable = term->gtable;
  int size_gtable = term->cells_gtable * GTABLE_SIZE_MULTIPLIER;
  int cells_gtable = term->cells_gtable / (1 + FULL_STORAGE);
  KeyCValue g_kv;
  memset(gtable, 0xff, size_gtable * sizeof(GTABLE_TYPE));
  int binsearch_count = 0;
  if(!DENSE_STORAGE)
    for(int j = 0; j < term->cells_gtable; j++)
      term->enc_B[j] = j;
  for(int j = 0; j < cells_gtable; j++)
  {
    ygi = 0;
    // If we have an arrangement where H is orthog to a hyperplane (during the msmt update)
    for(int k = 0; k < m; k++)
    {
      if( !(Horthog_flag & (1<<k)) )
      {
        sign_b[k] = ((j >> k) & 0x01) == 0 ? 1 : -1;
        ygi += p[k] * sign_b[k];
      }
    }
    g_kv.key = j;
    //C_COMPLEX_TYPE denom_left = CMPLX(ygi + term->d_val, term->c_val);
    //C_COMPLEX_TYPE denom_right = CMPLX(ygi - term->d_val, term->c_val);
    //g_kv.value = 1.0 / denom_left - 1.0 / denom_right;
    g_kv.value = 1.0 / (term->d_val + ygi + I*term->c_val) - 1.0 / (-term->d_val + ygi + I*term->c_val);
    g_kv.value *= G_SCALE_FACTOR;

    if(HASHTABLE_STORAGE)
      gtable_insert(gtable, &g_kv, size_gtable, rev_mask);
    if(BINSEARCH_STORAGE)
    {
      gtable_insert(gtable, &g_kv, binsearch_count++, rev_mask);
      if(FULL_STORAGE)
        binsearch_count++;
    }
    if(DENSE_STORAGE)
      gtable_insert(gtable, &g_kv, size_gtable, rev_mask);
  }
  // After creating the gtable, sort the keys in ascending order if using binsearch storage
  // To do: Make this use timsort
  if(BINSEARCH_STORAGE)
    qsort(gtable, term->cells_gtable, sizeof(GTABLE_TYPE), sort_key_cvalues);
}

//// HASHTABLE
// B_tp points to the B_mu array
// work_Btable is temporary space to construct the hashtable that build enc_B
// work2_Btable is temporary space to construct the hashtable that build enc_B
// On entry, term->enc_B must point to a valid memory array 
// On entry, term->gtable must point to a valid memory array
// If KEEP_BTABLES is set to true, B_enc points to memory in the chunked_Btable_ps array
// If KEEP_BTABLES is set to false, B_enc points to temporary memory
bool make_gtable(
  CauchyTerm* term, 
  const double G_SCALE_FACTOR)
{
  bool is_term_negligable = false;
  double p_sum_squared, ygi;
  int b_enc, enc_lp, enc_lm;
  int bval_idx; int coal_sign; int b_val;
  int j, k, l; 
  const int m = term->m;
  const int phc = term->phc;
  const int two_to_m_minus1 = (1<<(m-1));
  const int rev_mask = (1<<m)-1;
  const int two_to_phc_minus1 = (1<<(phc-1));
  const int rev_phc_mask = (1<<phc) - 1; // mask to reverse enc_lp or enc_lm if they have SV in negative halfspace w.r.t last HP 
  int sign_b[m];
  KeyCValue g_kv;
  C_COMPLEX_TYPE g_num_p, g_num_m;
  double c_val = term->c_val;
  double d_val = term->d_val;

  // experimental 
  if(WITH_TERM_APPROXIMATION)
  {
    is_term_negligable = true;
    p_sum_squared = sum_vec(term->p, m);
    p_sum_squared *= p_sum_squared;
  }

  const int num_cells = term->cells_gtable;
  int* enc_B = term->enc_B;
  int Horthog_flag = term->Horthog_flag;
  double* q = term->q;
  uint8_t* c_map = term->c_map;
  int8_t* cs_map = term->cs_map;
  GTABLE gtable_p = term->gtable_p;
  int size_gtable_p = term->cells_gtable_p * GTABLE_SIZE_MULTIPLIER;
  GTABLE gtable = term->gtable;
  int size_gtable = term->cells_gtable * GTABLE_SIZE_MULTIPLIER;
  if(!DENSE_STORAGE)
    memset(gtable, kByteEmpty, size_gtable * sizeof(GTABLE_TYPE));
  int enc_lhp = term->enc_lhp;
  int z_idx = term->z;
  
  //For Binsearch
  int binsearch_count=0;

  // Whether we store half the arrangement's sign-vectors or all them, only evaluate gtable over half the arrangement
  for(j = 0; j < num_cells; j++)
  {
    // only consider half of the B's, since the Gs of the other half are the complex conjugate of these
    b_enc = enc_B[j];
    if(FULL_STORAGE)
      if( b_enc & two_to_m_minus1)
        continue;

    ygi = 0;
    if(Horthog_flag)
    {
      for(k = 0; k < m; k++)
      {
        sign_b[k] = ((b_enc >> k) & 0x01) ? -1 : 1;
        if( !(Horthog_flag & (1<<k)) )
          ygi += q[k] * sign_b[k];
      }
    }
    else 
    {
      for(k = 0; k < m; k++)
      {
        sign_b[k] = ((b_enc >> k) & 0x01) ? -1 : 1;
        ygi += q[k] * sign_b[k];
      }
    }
    
    // enc_lp and enc_lm will be the same if this is an old term
    if( !(term->is_new_child) )
    {
      enc_lp = b_enc & rev_phc_mask; // clear all Gamma bits
      enc_lm = enc_lp;
    }
    // enc_lp and enc_lm dont need coalign maps if no coalignment happens
    else if(term->c_map == NULL)
    {
      // Create lambda_p and lambda_m
      k = 0;
      l = 0;
      enc_lp = 0;
      enc_lm = 0;
      while(k < phc)
      {
        if(k == z_idx)
        {
          enc_lm |= (1 << k);
          k++;
          if(k == phc)
            break;
        }
        if(sign_b[l] < 0)
        {
          enc_lp |= (1 << k);
          enc_lm |= (1 << k);
        }
        k++;
        l++;
      }
    }
    // using the coalignment maps to create enc_lp and enc_lp if coalignment occurs
    else 
    {
      // Create lambda_p and lambda_m by expanding out the sign vector sign_b found by inc_enu
      // This is done by using the coalignment map and the coalignment sign map to place the sign-values of the vector sign_b into their respective positions
      //for(int k = 0; k < phc; k++)
      k = 0;
      l = 0;
      enc_lp = 0;
      enc_lm = 0;
      while(k < phc)
      {
        if(k == z_idx)
        {
          enc_lm |= (1 << k);
          k++;
          if(k == phc)
            break;
        }
        bval_idx = c_map[l]; // index of the sign value sign_b[k] in the parent sign vector lambda_p and lambda_m (due to coalignment)
        coal_sign = cs_map[l]; // sign flip of the sign value sign_b[k] in the parent sign vector lambda_p and lambda_m (due to coalignment)
        b_val = sign_b[bval_idx] * coal_sign; // flip (potentially) 
        if(b_val < 0)
        {
          enc_lp |= (1 << k);
          enc_lm |= (1 << k);
        }
        k++;
        l++;
      }
    }

    // Create encoded versions of lambda_p \circ lambda_hat and lambda_m \circ lambda_hat to access parent gs
    g_num_p = lookup_g_numerator(enc_lp ^ enc_lhp, two_to_phc_minus1, rev_phc_mask, gtable_p, size_gtable_p, true);
    g_num_m = lookup_g_numerator(enc_lm ^ enc_lhp, two_to_phc_minus1, rev_phc_mask, gtable_p, size_gtable_p, false);

    g_kv.key = b_enc;
    g_kv.value = g_num_p / (d_val + ygi + I*c_val) - g_num_m / (-d_val + ygi + I*c_val);
    //g_kv.value = g_num_p / CMPLX(ygi+d_val, c_val) - g_num_m / CMPLX(ygi-d_val, c_val);
    g_kv.value *= G_SCALE_FACTOR;

    // --- Store depending on method --- //
    if(HASHTABLE_STORAGE)
      gtable_insert(gtable, &g_kv, size_gtable, rev_mask);
    if(BINSEARCH_STORAGE)
    {
      gtable_insert(gtable, &g_kv, binsearch_count++, rev_mask);
      if(FULL_STORAGE)
        binsearch_count++;
    }
    if(DENSE_STORAGE)
      gtable_insert(gtable, &g_kv, size_gtable, rev_mask);

    // If we are using the term approximation, check if |g|*p_sum_squared is under eps
    if(WITH_TERM_APPROXIMATION)
    {
      if(is_term_negligable)
        if( (p_sum_squared*cabs(g_kv.value)) > TERM_APPROXIMATION_EPS)
          is_term_negligable = false;
    }
  } 
  // After creating the gtable, sort the keys in ascending order if using binsearch storage
  // To do: Make this use timsort
  if(BINSEARCH_STORAGE)
    qsort(gtable, num_cells, sizeof(GTABLE_TYPE), sort_key_cvalues);

  return is_term_negligable;
}

////HASHTABLE
// Adds the g-table of term i to the g-table of term j
void add_gtables(CauchyTerm* term_i, CauchyTerm* term_j)
{
  GTABLE gtable_i = term_i->gtable;
  GTABLE gtable_j = term_j->gtable;
  int size_gtable = term_i->cells_gtable * GTABLE_SIZE_MULTIPLIER; // gtables are same size
  BKEYS enc_Bi = term_i->enc_B;
  int cells_Bi = term_i->cells_gtable;
  double* Ai = term_i->A;
  double* Aj = term_j->A;
  const int m = term_i->m;
  const int d = term_i->d;
  assert(term_i->m == term_j->m); // program error if this is not the case

  int enc_bi;
  int enc_bj;
  const int two_to_m_minus1 = (1<<(m-1));
  const int rev_b = (1<<m) - 1;
  int kd = 0;
  int k = 0;
  int l;
  int sigma_enc = 0;
  while(k < m)
  {
    l = 0;
    while( (fabs(Ai[kd + l]) < REDUCTION_EPS) || (fabs(Aj[kd + l]) < REDUCTION_EPS) )
      l++;
    if( (Ai[kd + l] * Aj[kd + l]) < 0)
      sigma_enc |= (1<<k);
    k += 1;
    kd += d;
  }

  for(k = 0; k < cells_Bi; k++)
  {
    if(FULL_STORAGE)
    {
      enc_bi = enc_Bi[k];
      enc_bj = enc_bi ^ sigma_enc;
      gtable_add(enc_bi, enc_bj, gtable_i, gtable_j, size_gtable, false);
    }
    else
    {
      // Only keys that are in the positive halfspace of gtable_i and gtable_j are stored
      enc_bi = enc_Bi[k]; // will be in positive halfspace of last HP
      enc_bj = enc_bi ^ sigma_enc; // may not be in positive halfspace of last HP
      // If enc_bj's sv is in the negative halfspace of its last HP,
      // reverse enc_bj's sv and then add conj(gtable_j["reversed enc_bj"]) g-value to gtable_i["enc_bi"]
      bool use_conj = false;
      if(enc_bj & two_to_m_minus1) 
      {
        use_conj = true;
        enc_bj ^= rev_b;
      }
      gtable_add(enc_bi, enc_bj, gtable_i, gtable_j, size_gtable, use_conj);
    }
  }
}

// Makes all gtables for shape m of the CF
void make_gtables(
         int* Nt_reduced, int* Nt_removed,
         CauchyTerm* terms,
         CauchyTerm* ftr_terms,
         ForwardFlagArray* ffa,
         DiffCellEnumHelper* dce_helper, 
         ChunkedPackedTableStorage* gb_tables,
         ReductionElemStorage* reduce_store,
         int* backward_F,
         int* B_dense, const double G_SCALE_FACTOR,
         const int Nt_shape, const int max_Nt_reduced_shape, 
         const int m, const int d,
         int start_idx = -1, int end_idx = -1, 
         bool is_unlowered_child = true)
{
  // Only in the threaded case are these useful
  start_idx = (start_idx == -1) ? 0 : start_idx;
  end_idx = (end_idx == -1) ? Nt_shape : end_idx;

  // Begin routine
  int Nt_reduced_shape = 0;
  int Nt_removed_shape = 0;
  int max_cells_shape = !DENSE_STORAGE ? dce_helper->cell_counts_cen[m] / (1 + HALF_STORAGE) : (1<<m) / (1+HALF_STORAGE);
  int dce_temp_hashtable_size = max_cells_shape * dce_helper->storage_multiplier;
  gb_tables->extend_gtables(max_cells_shape, max_Nt_reduced_shape);
  BYTE_COUNT_TYPE ps_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * m * sizeof(double);
  BYTE_COUNT_TYPE bs_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * d * sizeof(double);
  reduce_store->extend_storage(ps_bytes, bs_bytes, d);
  if(!DENSE_STORAGE)
      gb_tables->extend_btables(max_cells_shape, max_Nt_reduced_shape);
  // Now make B-Tables, G-Tables, for each reduction group
  int** forward_F = ffa->Fs;
  int* forward_F_counts = ffa->F_counts;
  for(int j = start_idx; j < end_idx; j++)
  {
      // Check whether we need to process term j (if it has reductions or is unique)
      if(backward_F[j] == j)
      {
          int rt_idx = j;
          CauchyTerm* child_j = terms + rt_idx;
          // Make the Btable if not an old term

          if(DENSE_STORAGE)
          {
              child_j->enc_B = B_dense;
              child_j->cells_gtable = max_cells_shape;
          }
          else
          {
              if( (child_j->is_new_child) )
              {
                  // Set the child btable memory position
                  BKEYS parent_B = child_j->enc_B;
                  int num_cells_parent = child_j->cells_gtable;
                  gb_tables->set_term_btable_pointer(&(child_j->enc_B), max_cells_shape, false);
                  // Regular functionality of make gtables
                  if(is_unlowered_child)
                  {
                    make_new_child_btable(child_j, 
                        parent_B, num_cells_parent,
                        dce_helper->B_mu_hash, num_cells_parent * dce_helper->storage_multiplier,
                        dce_helper->B_coal_hash, dce_temp_hashtable_size,
                        dce_helper->B_uncoal, dce_helper->F);
                  }
                  else 
                  {
                    make_lowered_child_btable(child_j, 
                        parent_B, num_cells_parent,
                        dce_helper->B_mu_hash, num_cells_parent * dce_helper->storage_multiplier,
                        dce_helper->B_coal_hash, dce_temp_hashtable_size,
                        dce_helper->B_uncoal, dce_helper->F);
                  }
              }
          }
          //printf("B%d is:\n", Nt_reduced_shape);
          //print_B_encoded(child_j->enc_B, child_j->cells_gtable, child_j->m, true);
          
          // set memory position of the child gtable
          // Make the g-table of the root term
          gb_tables->set_term_gtable_pointer(&(child_j->gtable), child_j->cells_gtable, false);
          // If the term is negligable (is approximated out), we need to search for a new "root"
          if( make_gtable(child_j, G_SCALE_FACTOR) )
              rt_idx = -1;
          else
          {
              gb_tables->incr_chunked_gtable_ptr(child_j->cells_gtable);
              if(!DENSE_STORAGE)
                  if(child_j->is_new_child)
                      gb_tables->incr_chunked_btable_ptr(child_j->cells_gtable);
          }
          
          int num_term_combos = forward_F_counts[j];
          int k = 0;
          // If the root term has been approximated out, we need to search through its term combinations to find a new term to take the place as root
          if(rt_idx == -1)
          {
              int num_cells_of_red_group = child_j->cells_gtable;
              BKEYS btable_for_red_group = child_j->enc_B;
              double* A_lfr = child_j->A; // HPA of the last failed root
              while(k < num_term_combos)
              {
                  int cp_idx = forward_F[j][k++];
                  CauchyTerm* child_k = terms + cp_idx;
                  // The btable of all terms in this reduction group are similar
                  // The only difference is the orientation of their hyperplanes
                  // Update the Btable of the last potential root for child_k
                  // Use the memory space currently pointed to by child_j only if child_k is not a parent 
                  if(DENSE_STORAGE)
                  {
                      child_k->enc_B = btable_for_red_group;
                      child_k->cells_gtable = num_cells_of_red_group;
                      child_k->gtable = child_j->gtable;
                  }
                  else 
                  {
                      if(child_k->is_new_child)
                      {
                          child_k->enc_B = btable_for_red_group;
                          child_k->cells_gtable = num_cells_of_red_group;
                          update_btable(A_lfr, child_k->enc_B, child_k->A, NULL, child_k->cells_gtable, m, d);
                          // Set memory position of child gtable k here
                          // This child can use the gtable memory position of child_j (since it was approximated out)
                          child_k->gtable = child_j->gtable;
                      }
                      // If child_k is a parent, its B is already in memory, no need to use new space
                      else 
                      {   
                          btable_for_red_group = child_k->enc_B;
                          if(child_k->cells_gtable == num_cells_of_red_group)
                          {
                              // Set memory position of child gtable k here
                              // This child can use the gtable memory position of child_j (since it was approximated out)
                              child_k->gtable = child_j->gtable;
                          }
                          // Only in the case of numerical round off error can two reducing terms
                          // have the same HPA (up to +/- direction of their normals)
                          // but a different numbers of cells. 
                          // So in the case where the two have different cell counts, do the following:
                          
                          // if the cells are less, 
                          // update cell_count of red group
                          // set gtable to child_j memory range (it will fit with certainty)
                          else if(child_k->cells_gtable < num_cells_of_red_group)
                          {
                              num_cells_of_red_group = child_k->cells_gtable;
                              child_k->gtable = child_j->gtable;
                          }
                          // if the cells are greater
                          // update cell_count of red group
                          // recheck gtable pointer memory address
                          else
                          {
                              num_cells_of_red_group = child_k->cells_gtable; 
                              gb_tables->set_term_gtable_pointer(&(child_k->gtable), num_cells_of_red_group, false); 
                          }
                      }
                  }
                  // If child term k is not approximated out, it becomes root
                  if( !make_gtable(child_k, G_SCALE_FACTOR) )
                  {
                      rt_idx = cp_idx;
                      if(!DENSE_STORAGE)
                          if(child_k->is_new_child)
                              gb_tables->incr_chunked_btable_ptr(child_k->cells_gtable);
                      gb_tables->incr_chunked_gtable_ptr(child_k->cells_gtable);
                      child_j = child_k;
                      break;
                  }
                  else
                      A_lfr = child_k->A;
              }
          }
          // For all terms combinations left, create thier g-table. If it is not approximated out, add it to the root g-table
          while(k < num_term_combos)
          {
              int cp_idx = forward_F[j][k++];
              CauchyTerm* child_k = terms + cp_idx;
              // Set memory position of child gtable k here
              // Make the Btable if not an old term

              if(DENSE_STORAGE)
              {
                  child_k->cells_gtable = child_j->cells_gtable;
                  child_k->enc_B = B_dense;
              }
              else
              {
                  if(child_k->is_new_child)
                  {
                      // Set the child btable memory position
                      child_k->cells_gtable = child_j->cells_gtable;
                      gb_tables->set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                      update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                  }
                  else
                  {
                      // To deal with the case where numerical instability causes cell counts to be different, 
                      // If the cell counts are different (due to instability), update child_k's btable to be compatible with the root
                      if(child_k->cells_gtable != child_j->cells_gtable)
                      {
                          if(WITH_WARNINGS)
                            printf(RED"[BIG WARN FTR/Make Gtables:] child_k->cells_gtable != child_j->cells_gtable. We have code below to fix this! But EXITING now until this is commented out!" NC "\n");
                          //exit(1);
                          // If child_k has more than child_j's cells,
                          // Downgrade child_k to be equal to child_j
                          if(child_k->cells_gtable > child_j->cells_gtable)
                          {
                              child_k->cells_gtable = child_j->cells_gtable;
                              update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                          }
                          // If child_k has less than child_j's cells, 
                          // Upgrade child_k to be equal to child_j
                          // Here we need to abandon the old btable memory location of child_k and acquire a new one 
                          // This is to keep the btables consistent, 
                          // child_k's btable then be set to the (re-oriented) child_j's btable.
                          else
                          {
                              child_k->cells_gtable = child_j->cells_gtable;
                              gb_tables->set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                              update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                          }
                      }
                  }
              }
              gb_tables->set_term_gtable_pointer(&(child_k->gtable), child_k->cells_gtable, false);
              // If child term k is not approximated out, we can add the gtables together
              if( !make_gtable(child_k, G_SCALE_FACTOR) )
              { 
                  add_gtables(child_j, child_k);
                  // No need to re-increment gb_table pointers
                  // This is because for the remainder of the terms to be combined, we can use this memory space again
              }
          }
          // If we found a root term, then increase the (reduced) term count
          if(rt_idx != -1)
          {
              child_j->become_parent();
              reduce_store->set_term_ptrs(child_j);
              ftr_terms[Nt_reduced_shape++] = *child_j;
          }
          else
              Nt_removed_shape++;

          //terms[j].become_null();
          //for(int l = 0; l < num_term_combos; l++)
          //    terms[forward_F[j][l]].become_null();
      }
  }
  *Nt_reduced = Nt_reduced_shape;
  *Nt_removed = Nt_removed_shape;
}


struct DIST_MAKE_GTABLES_STRUCT
{
  int Nt_reduced; 
  int Nt_removed;
  CauchyTerm* terms;
  CauchyTerm* ftr_terms;
  ForwardFlagArray* ffa;
  DiffCellEnumHelper* dce_helper;
  ChunkedPackedTableStorage* gb_tables;
  ReductionElemStorage* reduce_store;
  int* backward_F;
  int* B_dense; 
  double G_SCALE_FACTOR;
  int Nt_shape; 
  int max_Nt_reduced_shape;
  int m;
  int d;
  int start_idx;
  int end_idx;
};

void* callback_make_gtables(void* args)
{
  DIST_MAKE_GTABLES_STRUCT* gtable_args = (DIST_MAKE_GTABLES_STRUCT*) args;
  make_gtables(
    &(gtable_args->Nt_reduced), 
    &(gtable_args->Nt_removed),
    gtable_args->terms,
    gtable_args->ftr_terms,
    gtable_args->ffa,
    gtable_args->dce_helper, 
    gtable_args->gb_tables,
    gtable_args->reduce_store,
    gtable_args->backward_F,
    gtable_args->B_dense, 
    gtable_args->G_SCALE_FACTOR,
    gtable_args->Nt_shape, 
    gtable_args->max_Nt_reduced_shape,
    gtable_args->m, gtable_args->d,
    gtable_args->start_idx, gtable_args->end_idx);
  return NULL;
}


int threaded_make_gtables(int* Nt_reduced, int* Nt_removed,
         CauchyTerm* terms,
         CauchyTerm* ftr_terms,
         ForwardFlagArray* ffa,
         DiffCellEnumHelper* dce_helper, 
         ChunkedPackedTableStorage* gb_tables,
         ReductionElemStorage* reduce_store,
         int* backward_F,
         int* B_dense, const double G_SCALE_FACTOR,
         const int Nt_shape, const int m, const int d,
         const int win_num, const int step, const int total_steps)
{
  const int num_chunks = (Nt_shape + MIN_TERMS_PER_THREAD_GTABLE -1) / MIN_TERMS_PER_THREAD_GTABLE;
  int num_tids = num_chunks > NUM_CPUS ? NUM_CPUS : num_chunks;

  // Get start and end indices for each thread's gtable evaluation chunk
  int start_idxs[num_tids];
  int end_idxs[num_tids];
  num_tids = ffa->get_balanced_threaded_flattening_indices(num_tids, start_idxs, end_idxs, win_num, step, total_steps);
  // Number of threads can decrease if there are a very large number of combinations for a certain term, or certain terms
  pthread_t tids[num_tids];
  DIST_MAKE_GTABLES_STRUCT gtable_args[num_tids];
  int cumsum_reduced_terms = 0;
  for(int i = 0; i < num_tids; i++)
  {
    gtable_args[i].terms = terms;
    if(WITH_TERM_APPROXIMATION)
    {
      gtable_args[i].ftr_terms = (CauchyTerm*) malloc(ffa->reduced_terms_per_chunk[i] * sizeof(CauchyTerm));
      null_ptr_check(gtable_args[i].ftr_terms);
    }
    else
      gtable_args[i].ftr_terms = ftr_terms + cumsum_reduced_terms;
    gtable_args[i].ffa = ffa;
    gtable_args[i].dce_helper = dce_helper + i;
    gtable_args[i].gb_tables = gb_tables + i;
    gtable_args[i].reduce_store = reduce_store + i;
    gtable_args[i].backward_F = backward_F;
    gtable_args[i].B_dense = B_dense; 
    gtable_args[i].G_SCALE_FACTOR = G_SCALE_FACTOR;
    gtable_args[i].Nt_shape = Nt_shape; 
    gtable_args[i].max_Nt_reduced_shape = ffa->reduced_terms_per_chunk[i];
    gtable_args[i].m = m;
    gtable_args[i].d = d;
    gtable_args[i].start_idx = start_idxs[i];
    gtable_args[i].end_idx = end_idxs[i];
    cumsum_reduced_terms += ffa->reduced_terms_per_chunk[i];
    pthread_create(tids + i, NULL, callback_make_gtables, gtable_args + i);
  }
  assert(ffa->num_terms_after_reduction == cumsum_reduced_terms);
  for(int i = 0; i < num_tids; i++)
    pthread_join(tids[i], NULL);

  if(WITH_TERM_APPROXIMATION)
  {
    int Nt_reduced_shape = 0;
    int Nt_removed_shape = 0;
    for(int i = 0; i < num_tids; i++)
    {
      memcpy(ftr_terms + Nt_reduced_shape, gtable_args[i].ftr_terms, gtable_args[i].Nt_reduced * sizeof(CauchyTerm));
      Nt_reduced_shape += gtable_args[i].Nt_reduced;
      Nt_removed_shape += gtable_args[i].Nt_removed;
      free(gtable_args[i].ftr_terms);
    }
    *Nt_reduced = Nt_reduced_shape;
    *Nt_removed = Nt_removed_shape;
  }
  else 
  {
    *Nt_reduced = ffa->num_terms_after_reduction;
    *Nt_removed = 0;
  }

  return num_tids;
}

#endif //_FLATTENING_HPP_