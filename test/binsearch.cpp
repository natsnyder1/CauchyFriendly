#include <stdio.h>
#include "../include/gtable.hpp"

int main()
{
    const int N = 11;
    KeyCValue kvs[N];
    kvs[0].key = 0; kvs[0].value = 0; 
    kvs[1].key = 4; kvs[1].value = 0; 
    kvs[2].key = 7; kvs[2].value = 0; 
    kvs[3].key = 11; kvs[3].value = 0; 
    kvs[4].key = 13; kvs[4].value = 0; 
    kvs[5].key = 18; kvs[5].value = 0; 
    kvs[6].key = 23; kvs[6].value = 0; 
    kvs[7].key = 27; kvs[7].value = 0; 
    kvs[8].key = 31; kvs[8].value = 0; 
    kvs[9].key = 37; kvs[9].value = 0; 
    kvs[10].key = 45; kvs[10].value = 0; 

    for(int i = 0; i < N+1; i++)
    {
        uint32_t target = (i == N) ? 101 : kvs[i].key;
        int idx = binsearch(kvs, target, N);
        if(idx == -1)
            printf("Target Key %d not found!\n", idx);
        else
            printf("Target Key %d was found at index %d\n", target, idx);
    }

    return 0;
}