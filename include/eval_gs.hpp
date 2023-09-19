#ifndef _EVAL_GS_HPP_
#define _EVAL_GS_HPP_

#include "cauchy_types.hpp"

// gtable_p is size c_c(A_i^{k-1|k-1}) for FULL_STORAGE AND c_c(A_i^{k-1|k-1})/2 for half store
C_COMPLEX_TYPE g_num_hashtable(int enc_l, int phc, int two_to_phc_minus1, int rev_phc_mask, KeyCValue* gtable_p, int gtable_p_size, const bool is_pos_numerator)
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
                return 0;
        }
        else
            return g_kv.value;
        }
    else
    {
        KeyCValue g_kv;
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
                    return 0;
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
                    return 0;
            }
            else
                return g_kv.value;
        }
    }
}

/*
// gtable_p is size c_c(A_i^{k-1|k-1}) for FULL_STORAGE AND c_c(A_i^{k-1|k-1})/2 for half store
C_COMPLEX_TYPE g_num_binsearch(int enc_l, int phc, int two_to_phc_minus1, int rev_phc_mask, C_COMPLEX_TYPE* gtable_p, int gtable_p_size, const bool is_pos_numerator)
{
    printf("Binary Search G-lookup not implemented yet!");
    exit(1);
}

// gtable_p is size 2^{m_i^{k-1|k-1}} for FULL_STORAGE AND 2^{m_i^{k-1|k-1}-1} for half store
C_COMPLEX_TYPE g_num_dense(int enc_l, int phc, int two_to_phc_minus1, int rev_phc_mask, C_COMPLEX_TYPE* gtable_p, int gtable_p_size, const bool is_pos_numerator)
{
    if(FULL_STORAGE)
        return gtable_p[enc_l];
    else
        return  (enc_l & two_to_phc_minus1) ? conj(gtable_p[enc_l ^ rev_phc_mask]) : gtable_p[enc_l]; 
}
*/

#endif //_EVAL_GS_HPP_