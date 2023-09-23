#ifndef _CAUCHY_TYPES_HPP_
#define _CAUCHY_TYPES_HPP_

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "gtable.hpp"


// Chosing a Gtable Storage Methods dictates how the program will operate
// Note only one of these three options can be true
#define HASHTABLE_STORAGE true // Sets the gtables to be stored in hashtables
#define BINSEARCH_STORAGE false // Sets the gtables to be stored in arrays using binary search 
#define DENSE_STORAGE false // Sets the gtables to be stored in arrays with all possible 2^m sign vectors evaluated
// Sets whether only half of the gtable sign-vectors are stored in the gtable
// This has no effect when the dense storage method is selected
#define FULL_STORAGE false
// Sets whether to keep the B tables in memory, or simply temporary memory 
// This has no effect for both the dense gtable method and the binsearch method
#define KEEP_BTABLES false 
// Helper declarations that are automatically set
#define HALF_STORAGE (!FULL_STORAGE)
#define DISCARD_BTABLES (!KEEP_BTABLES)

#if HASHTABLE_STORAGE
    typedef KeyCValue GTABLE_TYPE;
    typedef GTABLE_TYPE* GTABLE;
#else 
    typedef C_COMPLEX_TYPE GTABLE_TYPE;
    typedef C_COMPLEX_TYPE* GTABLE;
#endif

enum GTABLE_STORAGE_TYPE{
GTABLE_HASHTABLE_STORAGE,
GTABLE_BINSEARCH_STORAGE,
GTABLE_DENSE_STORAGE,
};

#if HASHTABLE_STORAGE && (!BINSEARCH_STORAGE) && (!DENSE_STORAGE)
    const GTABLE_STORAGE_TYPE GTABLE_STORAGE_METHOD = GTABLE_HASHTABLE_STORAGE;
#elif (!HASHTABLE_STORAGE) && (BINSEARCH_STORAGE) && (!DENSE_STORAGE)
    const GTABLE_STORAGE_TYPE GTABLE_STORAGE_METHOD = GTABLE_BINSEARCH_STORAGE;
#elif (!HASHTABLE_STORAGE) && (!BINSEARCH_STORAGE) && (DENSE_STORAGE)
    const GTABLE_STORAGE_TYPE GTABLE_STORAGE_METHOD = GTABLE_DENSE_STORAGE;
#else
    #error "[COMPILATION ERROR:] ONLY ONE OUT OF HASHTABLE_STORAGE, BINSEARCH_STORAGE, DENSE_STORAGE SETTINGS can be set to true!"
#endif

typedef unsigned long long int BYTE_COUNT_TYPE;
typedef KeyValue BTABLE_TYPE;
typedef BTABLE_TYPE* BTABLE;
typedef int BKEYS_TYPE;
typedef BKEYS_TYPE* BKEYS;

// Function Pointer Definitions
// THIS NEEDS TO BE RE-IMPLEMENTED IN
typedef C_COMPLEX_TYPE (*LOOKUP_G_NUMERATOR_TYPE)(int enc_l, int two_to_phc_minus1, int rev_phc_mask, GTABLE gtable_p, int gtable_p_size, const bool is_pos_numerator);
LOOKUP_G_NUMERATOR_TYPE lookup_g_numerator; // Used to lookup gtable_p values
//typedef bool (*GTABLE_FIND_TYPE)(GTABLE* hashtable, GTABLE** kv, uint32_t key, uint32_t gtable_size);
//GTABLE_FIND_TYPE gtable_find;

//typedef void (*MAKE_GTABLE_TYPE)(int enc_l, int two_to_phc_minus1, int rev_phc_mask, GTABLE gtable_p, int gtable_p_size, const bool is_pos_numerator);



// NULL Pointer checker
void null_ptr_check(void* ptr)
{
    if(ptr == NULL)
    {
        printf("Pointer allocated by malloc has returned with NULL, indicating FAILURE! Please Debug Further!\n");
        exit(1);
    }
}
// NULL Double Pointer checker
void null_dptr_check(void** ptr)
{
    if(ptr == NULL)
    {
        printf("Double Pointer allocated by malloc has returned with NULL, indicating FAILURE! Please Debug Further!\n");
        exit(1);
    }
}
// NULL Triple Pointer checker
void null_tptr_check(void*** ptr)
{
    if(ptr == NULL)
    {
        printf("Double Pointer allocated by malloc has returned with NULL, indicating FAILURE! Please Debug Further!\n");
        exit(1);
    }
}

#endif //_CAUCHY_TYPES_HPP_