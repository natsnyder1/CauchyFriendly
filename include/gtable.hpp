#ifndef _GTABLE_HPP_
#define _GTABLE_HPP_

#include "cauchy_constants.hpp"

// This is the fix for mac, in linux, this ownt trigger -- possibly change cpu_linalg.hpp lines ~1776

#if __APPLE__
    double creal(C_COMPLEX_TYPE val)
    {
        return *( ((double*)(&val)) );
    }

    double cimag(C_COMPLEX_TYPE val)
    {
        return *( ((double*)(&val)) + 1);
    }

    double cabs(C_COMPLEX_TYPE val)
    {
        double real = *((double*)(&val));
        double imag = *( ((double*)(&val)) + 1);
        return sqrt(real*real + imag*imag);
    }

    C_COMPLEX_TYPE MAKE_CMPLX(double real, double imag)
    {
        C_COMPLEX_TYPE val;
        *((double*)&val) = real;
        *((double*)(&val)+1) = imag;
        return val;
    }   

    C_COMPLEX_TYPE conj(C_COMPLEX_TYPE val)
    {
        double* imag_ptr = ((double*)&val)+1;
        *imag_ptr = -(*imag_ptr);
        return val;
    }
#endif


#if ( __linux__ || _WIN32 )
    C_COMPLEX_TYPE MAKE_CMPLX(double real, double imag)
    {
        C_COMPLEX_TYPE val;
        *((double*)&val) = real;
        *((double*)(&val)+1) = imag;
        return val;
    }
#endif 


struct KeyValue
{
    uint32_t key; 
    uint32_t value;
};

struct KeyCValue
{
    uint32_t key;
    C_COMPLEX_TYPE value;
};

// 32 bit Murmur3 hash
uint32_t hash(uint32_t k, uint32_t kHashTableCapacity)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % kHashTableCapacity; //& (kHashTableCapacity-1);
}

// Insert the key/values into the hashtable
bool hashtable_insert(KeyCValue* hashtable, KeyCValue* kv, uint32_t kHashTableCapacity)
{

    uint32_t key = kv->key;
    uint32_t slot = hash(key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        uint32_t prev = hashtable[slot].key;
        if (prev == kEmpty || prev == key)
        {
            hashtable[slot].key = kv->key;
            hashtable[slot].value = kv->value;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity-1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    return true;
}

// returns true on error and false on (un)successful lookup
bool hashtable_lookup(const KeyCValue* hashtable, KeyCValue* kv, uint32_t kHashTableCapacity)
{
    uint32_t slot = hash(kv->key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        if (hashtable[slot].key == kv->key)
        {
            kv->value = hashtable[slot].value;
            return false;
        }
        if (hashtable[slot].key == kEmpty)
        {
            kv->key = kEmpty;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity - 1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    return true;
}

bool hashtable_find(KeyCValue* hashtable, KeyCValue** kv, uint32_t key, uint32_t kHashTableCapacity)
{
    uint32_t slot = hash(key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        if (hashtable[slot].key == key)
        {
            *kv = hashtable + slot;
            return false;
        }
        if (hashtable[slot].key == kEmpty)
        {
            *kv = NULL;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity - 1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    *kv = NULL;
    return true;
}

void hashtable_print(KeyCValue* hashtable, uint32_t kHashTableCapacity)
{
    for(uint32_t i = 0; i < kHashTableCapacity; i++)
    {
        if(hashtable[i].key != kEmpty)
            printf("Key: %u, Value: %lf + %lfj\n", hashtable[i].key, creal(hashtable[i].value), cimag(hashtable[i].value));
    }
}

// Insert the key/values into the hashtable
bool hashtable_insert(KeyValue* hashtable, KeyValue* kv, uint32_t kHashTableCapacity)
{

    uint32_t key = kv->key;
    uint32_t slot = hash(key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        uint32_t prev = hashtable[slot].key;
        if (prev == kEmpty || prev == key)
        {
            hashtable[slot].key = kv->key;
            hashtable[slot].value = kv->value;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity-1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    return true;
}

// returns true on error and false on (un)successful lookup
bool hashtable_lookup(const KeyValue* hashtable, KeyValue* kv, uint32_t kHashTableCapacity)
{
    uint32_t slot = hash(kv->key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        if (hashtable[slot].key == kv->key)
        {
            kv->value = hashtable[slot].value;
            return false;
        }
        if (hashtable[slot].key == kEmpty)
        {
            kv->key = kEmpty;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity - 1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    return true;
}

bool hashtable_find(KeyValue* hashtable, KeyValue** kv, uint32_t key, uint32_t kHashTableCapacity)
{
    uint32_t slot = hash(key, kHashTableCapacity);
    uint32_t iters = 0;
    while (true)
    {
        if (hashtable[slot].key == key)
        {
            *kv = hashtable + slot;
            return false;
        }
        if (hashtable[slot].key == kEmpty)
        {
            *kv = NULL;
            return false;
        }
        slot = (slot + 1) % kHashTableCapacity; //& (kHashTableCapacity - 1);
        iters++;
        if(iters == kHashTableCapacity)
            break;
    }
    *kv = NULL;
    return true;
}

void hashtable_print(KeyValue* hashtable, uint32_t kHashTableCapacity)
{
    for(uint32_t i = 0; i < kHashTableCapacity; i++)
    {
        if(hashtable[i].key != kEmpty)
            printf("Key: %u, Value: %lf + %lfj\n", hashtable[i].key, creal(hashtable[i].value), cimag(hashtable[i].value));
    }
}

// enter binary search methods here
int binsearch(KeyCValue* gtable, uint32_t target_key, uint32_t gtable_size)
{
    int low = 0;
    int high = gtable_size - 1;
    int mid;
    uint32_t mid_key;
    while(low <= high)
    {
        mid = (low + high ) / 2;
        mid_key = gtable[mid].key;
        if(mid_key == target_key)
            return mid;
        else if(mid_key > target_key)
            high = mid - 1;
        else  
            low = mid + 1;
    }
    return -1;
}

int sort_key_cvalues(const void* p1, const void* p2)
{
    return ((KeyCValue*)p1)->key - ((KeyCValue*)p2)->key;
}




#endif // _GTABLE_HPP_