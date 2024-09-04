#ifndef _EVAL_GS_HPP_
#define _EVAL_GS_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_types.hpp"
#include "gtable.hpp"

// -------------- METHODS FOR LOOKING UP NUMERATOR OF G (GTABLE_P) --------------------- // 

// gtable_p is size c_c(A_i^{k-1|k-1}) for FULL_STORAGE AND c_c(A_i^{k-1|k-1})/2 for half store
C_COMPLEX_TYPE g_num_hashtable(int enc_l, int two_to_phc_minus1, int rev_phc_mask, KeyCValue* gtable_p, int gtable_p_size, const bool is_pos_numerator)
{
    if(FULL_STORAGE)
    {
        KeyCValue g_kv;
        g_kv.key = enc_l;
        bool failed = hashtable_lookup(gtable_p, &g_kv, gtable_p_size);
        if( failed || (g_kv.key == kEmpty) )
        {
            if(WITH_WARNINGS)
                printf(YEL "[WARNING #1 LOOKUP G (FULL METHOD):]\n" 
                    YEL "enc_lambda = %d was not found in gtable_p\n"
                    YEL "Positive Numerator: %d, failed: %d, kEmpty: %d"
                    NC "\n", enc_l, is_pos_numerator, failed, g_kv.key == kEmpty);
            if(EXIT_ON_FAILURE)
                exit(1);
            else if(failed)
            {
                printf(RED "Failure boolean was returned. Exiting!" NC "\n");
                exit(1);
            }
            else
                return MAKE_CMPLX(0,0);
        }
        else
            return g_kv.value;
        }
    else
    {
        KeyCValue g_kv;
        g_kv.value = 0;
        if(enc_l & two_to_phc_minus1)
        {
            g_kv.key = rev_phc_mask ^ enc_l;
            bool failed = hashtable_lookup(gtable_p, &g_kv, gtable_p_size);
            if( failed || (g_kv.key == kEmpty) )
            {
                if(WITH_WARNINGS)
                    printf(YEL "[WARNING #2 LOOKUP G (HALFMETHOD):]\n" 
                        YEL "enc_lambda = %d was not found in gtable_p (looked up via its opposite %d)\n"
                        YEL "Positive Numerator: %d, failed: %d, kEmpty: %d"
                        NC "\n", enc_l, rev_phc_mask ^ enc_l, is_pos_numerator, failed, g_kv.key == kEmpty);
                if(EXIT_ON_FAILURE)
                    exit(1);
                else if(failed)
                {
                    printf(RED "Failure boolean was returned. Exiting!" NC "\n");
                    exit(1);
                }
                else
					return MAKE_CMPLX(0, 0);
            }
            else
                return conj(g_kv.value);
        }
        else
        {
            g_kv.key = enc_l;
            bool failed = hashtable_lookup(gtable_p, &g_kv, gtable_p_size);
            if( failed || (g_kv.key == kEmpty) )
            {
                if(WITH_WARNINGS)
                    printf(YEL "[WARNING #3 LOOKUP G (HALFMETHOD):]\n" 
                        YEL "enc_lambda = %d was not found in gtable_p\n"
                        YEL "Positive Numerator: %d, failed: %d, kEmpty: %d"
                        NC "\n", enc_l, is_pos_numerator, failed, g_kv.key == kEmpty);
                if(EXIT_ON_FAILURE)
                    exit(1);
                else if(failed)
                {
                    printf(RED "Failure boolean was returned. Exiting!" NC "\n");
                    exit(1);
                }
                else
					return MAKE_CMPLX(0, 0);
            }
            else
                return g_kv.value;
        }
    }
}

// gtable_p is size c_c(A_i^{k-1|k-1}) for FULL_STORAGE AND c_c(A_i^{k-1|k-1})/2 for half store
C_COMPLEX_TYPE g_num_binsearch(int enc_l, int two_to_phc_minus1, int rev_phc_mask, KeyCValue* gtable_p, int gtable_p_size, const bool is_pos_numerator)
{
    if(FULL_STORAGE)
    {
        int idx = binsearch(gtable_p, enc_l, gtable_p_size);
        if( idx == -1 )
        {
            if(WITH_WARNINGS)
                printf(YEL "[WARNING #1 LOOKUP G (Binsearch FULL METHOD):]\n" 
                    YEL "enc_lambda = %d was not found in gtable_p\n"
                    YEL "Positive Numerator: %d"
                    NC "\n", enc_l, is_pos_numerator);
            if(EXIT_ON_FAILURE)
                exit(1);
            else
                return 0;
        }
        else
            return gtable_p[idx].value;
        }
    else
    {
        if(enc_l & two_to_phc_minus1)
        {            
            int idx = binsearch(gtable_p, rev_phc_mask ^ enc_l, gtable_p_size);
            if( idx == -1 )
            {
                if(WITH_WARNINGS)
                    printf(YEL "[WARNING #2 LOOKUP G (Binsearch HALFMETHOD):]\n" 
                        YEL "enc_lambda = %d was not found in gtable_p (looked up via its opposite %d)\n"
                        YEL "Positive Numerator: %d"
                        NC "\n", enc_l, rev_phc_mask ^ enc_l, is_pos_numerator);
                if(EXIT_ON_FAILURE)
                    exit(1);
                else
                    return 0;
            }
            else
                return conj(gtable_p[idx].value);
        }
        else
        {
            int idx = binsearch(gtable_p, enc_l, gtable_p_size);
            if( idx == -1 )
            {
                if(WITH_WARNINGS)
                    printf(YEL "[WARNING #3 LOOKUP G (binsearch HALFMETHOD):]\n" 
                        YEL "enc_lambda = %d was not found in gtable_p\n"
                        YEL "Positive Numerator: %d"
                        NC "\n", enc_l, is_pos_numerator);
                if(EXIT_ON_FAILURE)
                    exit(1);
                else
                    return 0;
            }
            else
                return gtable_p[idx].value;
        }
    }
}

// gtable_p is size 2^{m_i^{k-1|k-1}} for FULL_STORAGE AND 2^{m_i^{k-1|k-1}-1} for half store
C_COMPLEX_TYPE g_num_dense(int enc_l, int two_to_phc_minus1, int rev_phc_mask, C_COMPLEX_TYPE* gtable_p, int gtable_p_size, const bool is_pos_numerator)
{
    if(FULL_STORAGE)
        return gtable_p[enc_l];
    else
        return  (enc_l & two_to_phc_minus1) ? conj(gtable_p[enc_l ^ rev_phc_mask]) : gtable_p[enc_l]; 
}

// -------------- END OF METHODS FOR LOOKING UP NUMERATOR OF G (GTABLE_P) ------------- // 


// -------------- METHODS FOR INSERTING VALUE INTO GTABLE --------------------- // 

// Insert element to the gtables via a hashtable structure
void g_insert_hashtable(KeyCValue* gtable, KeyCValue* g_kv, uint32_t size_gtable, uint32_t rev_mask)
{
    if( hashtable_insert(gtable, g_kv, size_gtable) )
    {
        printf(RED"[ERROR #1: Make Gtable] hashtable_insert(...) for table returns failure=1. Debug here further! Exiting!" NC"\n");
        exit(1);
    }
    if(FULL_STORAGE)
    {
        g_kv->key ^= rev_mask;
        g_kv->value = conj(g_kv->value);
        if( hashtable_insert(gtable, g_kv, size_gtable) )
        {
            printf(RED"[ERROR #2: Make Gtable] hashtable_insert(...) for table returns failure=1. Debug here further! Exiting!" NC"\n");
            exit(1);
        }
    }
}

// Insert elements to the gtables for binsearch look-up
// Note that these key values must be sorted to make binsearch work
// This is handled through either the qsort or timsort method
void g_insert_binsearch(KeyCValue* gtable, KeyCValue* g_kv, uint32_t idx, uint32_t rev_mask)
{
      gtable[idx++] = *g_kv;
      if(FULL_STORAGE)
      {
        g_kv->key ^= rev_mask;
        g_kv->value = conj(g_kv->value);
        gtable[idx] = *g_kv;
      }
}

// dense insertion 
void g_insert_dense(C_COMPLEX_TYPE* gtable, KeyCValue* g_kv, size_t size_gtable, uint32_t rev_mask)
{
    gtable[g_kv->key] = g_kv->value;
    if(FULL_STORAGE)
    {
        g_kv->key ^= rev_mask;
        g_kv->value = conj(g_kv->value);
        if(g_kv->key >= size_gtable)
        {
            printf(RED "[ERROR G_DENSE_INSERT #2:] Break Here and Debug: Exiting!" NC "\n");
            exit(1);
        }
        gtable[g_kv->key] = g_kv->value;
    }
}

// -------------- END OF METHODS FOR INSERTING VALUE INTO GTABLE -------------- // 


// -------------- METHODS FOR ADDING GTABLES TOGETHER ----------------------------- // 
void gs_add_hashtable(int enc_bi, int enc_bj, KeyCValue* gtable_i, KeyCValue* gtable_j, int size_gtable, bool use_conj)
{
    KeyCValue* gtable_iter_i;
    KeyCValue* gtable_iter_j;
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
            if(WITH_WARNINGS)
                printf(YEL"[WARN #1 Add Gtables]: g_table_i does not contain the key %d queried by gtable_iter_i. Debug here further!" NC"\n", enc_bi);
        if(gtable_iter_j == NULL)
            if(WITH_WARNINGS)
                printf(YEL"[WARN #2 Add Gtables]: g_table_j does not contain the key %d queried by gtable_iter_j. Debug here further!" NC"\n", enc_bj);
        if(EXIT_ON_FAILURE)
        {
            printf(RED"[SIG_ABORT in Add Gtables]: EXIT_ON_FAILURE is set true. Exiting!" NC "\n");
            exit(1);
        }
    }
}

void gs_add_binsearch(int enc_bi, int enc_bj, KeyCValue* gtable_i, KeyCValue* gtable_j, int size_gtable, bool use_conj)
{
    // Place iterators to table positions for keys enc_bi and enc_bj
    // First check for failure signal in binsearch
    int idx_i = binsearch(gtable_i, enc_bi, size_gtable);
    int idx_j = binsearch(gtable_j, enc_bj, size_gtable);
    // If no errors are triggered add the g-values at the two positions together
    if( (idx_i > -1) && (idx_j > -1) )
    {
        if(use_conj)
            gtable_i[idx_i].value += conj(gtable_j[idx_j].value);
        else
            gtable_i[idx_i].value += gtable_j[idx_j].value;
    }
    else
    {
        if(idx_i == -1)
            if(WITH_WARNINGS)
                printf(YEL"[WARN #1 Add Gtables]: g_table_i does not contain the key %d queried by gtable_iter_i. Debug here further!" NC"\n", enc_bi);
        if(idx_j == -1)
            if(WITH_WARNINGS)
                printf(YEL"[WARN #2 Add Gtables]: g_table_j does not contain the key %d queried by gtable_iter_j. Debug here further!" NC"\n", enc_bj);
        if(EXIT_ON_FAILURE)
        {
            printf(RED"[SIG_ABORT in Add Gtables]: EXIT_ON_FAILURE is set true. Exiting!" NC "\n");
            exit(1);
        }
    }
}

void gs_add_dense(int enc_bi, int enc_bj, C_COMPLEX_TYPE* gtable_i, C_COMPLEX_TYPE* gtable_j, int size_gtable, bool use_conj)
{
    assert(enc_bi < size_gtable);
    if(use_conj)
        gtable_i[enc_bi] += conj(gtable_j[enc_bj]);
    else
        gtable_i[enc_bi] += gtable_j[enc_bj];
}
// -------------- END OF  METHODS FOR ADDING GTABLES TOGETHER --------------------- // 


// -------------- METHODS FOR FINDING PARENT GTABLE VALUE ----------------------------- // 
void gp_find_hashtable(KeyCValue* gtable_p, KeyCValue** kcv_query, int enc_psv, int gtable_p_table_size, int cells_parent)
{
    if(hashtable_find(gtable_p, kcv_query, enc_psv, gtable_p_table_size))
    {
        // hashtable_find only has an error if the table size is larger than minimal
        if( gtable_p_table_size > cells_parent)
        {
            printf(RED "[Error gp_find_hashtable]: hashtable find has failed! Debug here!" NC "\n");
            exit(1);
        }
    }
}

void gp_find_binsearch(KeyCValue* gtable_p, KeyCValue** kcv_query, int enc_psv, int gtable_p_table_size, int cells_parent)
{
    int idx = binsearch(gtable_p, enc_psv, gtable_p_table_size);
    if(idx == -1)
        *kcv_query = NULL;
    else
        *kcv_query = gtable_p + idx;
}
// -------------- END OF METHODS FOR FINDING PARENT GTABLE VALUE ---------------------- // 

// -------------- METHODS FOR RETRIVING PARENT GTABLE KEYS ----------------------------- // 
int gtable_p_get_keys_hashtable(KeyCValue* gtable_p, int gtable_p_table_size, int* keys)
{
    int key_count = 0;
    for(int i = 0; i < gtable_p_table_size; i++)
        if(gtable_p[i].key != kEmpty)
            keys[key_count++] = gtable_p[i].key;
    return key_count;
}       
int gtable_p_get_keys_binsearch(KeyCValue* gtable_p, int gtable_p_table_size, int* keys)
{
    int key_count = 0;
    for(int i = 0; i < gtable_p_table_size; i++)
        keys[key_count++] = gtable_p[i].key;
    return key_count;
}       

#endif //_EVAL_GS_HPP_