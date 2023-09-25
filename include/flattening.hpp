#ifndef _FLATTENING_HPP_
#define _FLATTENING_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cell_enumeration.hpp"
#include "eval_gs.hpp"
#include <assert.h>
#include <cstring>
//#include <sys/types.h>

//// HASHTABLE
// Returns G values per face of each term.
void make_gtable_first(CauchyTerm* term, const double G_SCALE_FACTOR)
{
  const int m = term->d;
  double sign_b[m];
  double ygi; 
  uint Horthog_flag = term->Horthog_flag;
  double* p = term->p;
  GTABLE gtable = term->gtable;
  int size_gtable = term->cells_gtable * GTABLE_SIZE_MULTIPLIER;
  KeyCValue g_kv;
  memset(gtable, 0xff, size_gtable * sizeof(GTABLE_TYPE));
  for(int j = 0; j < term->cells_gtable; j++)
  {
    term->enc_B[j] = j;
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
    if( hashtable_insert(gtable, &g_kv, size_gtable) )
    {
      printf(RED"[ERROR #1: Make Gtable First] hashtable_insert(...) for table returns failure=1. Debug here further! Exiting!" NC"\n");
      exit(1);
    }
  }
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
  memset(gtable, kByteEmpty, size_gtable * sizeof(GTABLE_TYPE));
  int enc_lhp = term->enc_lhp;
  int z_idx = term->z;
  
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
    if(term->parent == NULL)
    {
      enc_lp = b_enc & rev_phc_mask; // clear all Gamma bits
      enc_lm = enc_lp;
    }
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
        bval_idx = c_map[l]; // index of the sign value sign_b[k] in the parent sign vector lambda_p and lambda_m (due to coalignment)
        coal_sign = cs_map[l]; // sign flip of the sign value sign_b[k] in the parent sign vector lambda_p and lambda_m (due to coalignment)
        b_val = sign_b[bval_idx] * coal_sign; // flip (potentially) 
        if(k == z_idx)
        {
          enc_lm |= (1 << k);
          k++;
          if(k == phc)
            break;
        }
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
    if( hashtable_insert(gtable, &g_kv, size_gtable) )
    {
      printf(RED"[ERROR #1: Make Gtable] hashtable_insert(...) for table returns failure=1. Debug here further! Exiting!" NC"\n");
      exit(1);
    }
    if(FULL_STORAGE)
    {
      g_kv.key = b_enc ^ rev_mask;
      g_kv.value = conj(g_kv.value);
      if( hashtable_insert(gtable, &g_kv, size_gtable) )
      {
        printf(RED"[ERROR #2: Make Gtable] hashtable_insert(...) for table returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
      }
    }
    // If we are using the term approximation, check if |g|*p_sum_squared is under eps
    if(WITH_TERM_APPROXIMATION)
    {
      if(is_term_negligable)
        if( (p_sum_squared*cabs(g_kv.value)) > TERM_APPROXIMATION_EPS)
          is_term_negligable = false;
    }
  } 
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
  GTABLE gtable_iter_i;
  GTABLE gtable_iter_j; 
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
      // Place iterators to table positions for keys enc_bi and enc_bj
      // First check for failure signal in hashtable_find(...)
      if( hashtable_find(gtable_i, &gtable_iter_i, enc_bi, size_gtable) )
      {
        printf(RED"[ERROR #1: Add GTables] hashtable_find(...) for table_i returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
      }
      if( hashtable_find(gtable_j, &gtable_iter_j, enc_bj, size_gtable) )
      {
        printf(RED"[ERROR #2: Add GTables] hashtable_find(...) for table_j returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
      }
      // If no errors are tiggered add the g-values at the two positions together
      if( (gtable_iter_i != NULL) && (gtable_iter_j != NULL) )
      {
        gtable_iter_i->value += gtable_iter_j->value;
      }
      // Now check for error signal that a key could not be found (but no failure)
      // Since we are querying gtable_i from a key in gtable_j, an error should originate from gtable_i
      else
      {
        if(gtable_iter_i == NULL)
          printf(YEL"[WARN #1 Add Gtables]: g_table_i does not contain the key %d queried by gtable_iter_i. Debug here further!" NC"\n", enc_bi);
        if(gtable_iter_j == NULL)
          printf(YEL"[WARN #2 Add Gtables]: g_table_j does not contain the key %d queried by gtable_iter_j. Debug here further!" NC"\n", enc_bj);
        if(EXIT_ON_FAILURE)
        {
          printf(RED"[SIG_ABORT #1 in Add Gtables]: EXIT_ON_FAILURE is set true. Exiting!" NC "\n");
          exit(1);
        }
      }
    }
    else
    {
      // Only keys that are in the positive halfspace of gtable_i and gtable_j are stored
      enc_bi = enc_Bi[k]; // will be in positive halfspace of last HP
      if( hashtable_find(gtable_i, &gtable_iter_i, enc_bi, size_gtable) )
      {
        printf(RED"[ERROR #3: Add GTables] hashtable_find(...) for table_j returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
      }
      enc_bj = enc_bi ^ sigma_enc; // may not be in positive halfspace of last HP
      // If enc_bj's sv is in the negative halfspace of its last HP,
      // reverse enc_bj's sv and then add conj(gtable_j["reversed enc_bj"]) g-value to gtable_i["enc_bi"]
      bool use_conj = false;
      if(enc_bj & two_to_m_minus1) 
      {
        use_conj = true;
        enc_bj ^= rev_b;
      }
      if( hashtable_find(gtable_j, &gtable_iter_j, enc_bj, size_gtable) )
      {
        printf(RED"[ERROR #4: Add GTables] hashtable_find(...) for table_i returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
      }
      // If no errors are triggered add the g-values at the two positions together
      if( (gtable_iter_i != NULL) && (gtable_iter_j != NULL) )
      {
        if(use_conj)
          gtable_iter_i->value += conj(gtable_iter_j->value);
        else
          gtable_iter_i->value += gtable_iter_j->value;
      }
      else
      {
        if(gtable_iter_i == NULL)
          printf(YEL"[WARN #3 Add Gtables]: g_table_i does not contain the key queried by gtable_iter_i. original sv enc_bi=%d, reversed=%d (rev_b is %d), queried sv = %d. Debug here further!" NC"\n", enc_bi, enc_bi & two_to_m_minus1, rev_b, ( (enc_bi & two_to_m_minus1) ? enc_bi ^ rev_b : enc_bi) );
        if(gtable_iter_j == NULL)
          printf(YEL"[WARN #4 Add Gtables]: g_table_j does not contain the key %d queried by gtable_iter_j. Debug here further!" NC"\n", enc_bj);
        if(EXIT_ON_FAILURE)
        {
          printf(RED"[SIG_ABORT #2 in Add Gtables]: EXIT_ON_FAILURE is set true. Exiting!" NC "\n");
          exit(1);
        }
      }
    }
  }
}


/*
bool make_gtable_BINSEARCH(CauchyTerm* term, 
  KeyValue* B_mu_hash, int size_B_mu_hash, 
  KeyValue* B_coal_hash, int size_B_coal_hash, 
  BKEYS B_uncoal, bool* F,
  const double G_SCALE_FACTOR)
{

  return false;
}
*/

#endif //_FLATTENING_HPP_