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

    C_COMPLEX_TYPE CMPLX(double real, double imag)
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

#if __linux__
    C_COMPLEX_TYPE CMPLX(double real, double imag)
    {
        return real + I*imag;
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
bool hashtable_insert(KeyCValue* hashtable, const KeyCValue* kv, uint32_t kHashTableCapacity)
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
bool hashtable_insert(KeyValue* hashtable, const KeyValue* kv, uint32_t kHashTableCapacity)
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






#endif // _GTABLE_HPP_