#ifndef _CPU_TIMER_H
#define _CPU_TIMER_H

#include<stdio.h>
#include<time.h>
#include<unistd.h>
#include<sys/time.h>

/*
struct CPUTimer
{
  clock_t start, end; //clock_t
  double cpu_time_used;

  void tic(void)
  {
    start = clock();
  }

  void toc(bool is_print = true)
  {
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    if(is_print)
      printf("CPU Ellapsed Time: %.8f ms\n", 1000*cpu_time_used);
  }

};
*/

struct CPUTimer
{
  long start, end;
  struct timeval timecheck;
  int cpu_time_used; // milliseconds
  void tic(void)
  {
    gettimeofday(&timecheck, NULL);
    start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
  }

  void toc(bool is_print = true)
  {
    gettimeofday(&timecheck, NULL);
    end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    cpu_time_used = (end - start);
    if(is_print)
      printf("CPU Ellapsed Time: %d ms\n", cpu_time_used);
  }

};


#endif //_CPU_TIMER_H
