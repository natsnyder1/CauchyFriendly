#ifndef _CPU_TIMER_H
#define _CPU_TIMER_H

#include<stdio.h>
//#include<time.h>
#if !(_WIN32)
	#include<unistd.h>
	#include<sys/time.h>
#else 
	//#include <Windows.h>
	#include"../scripts/windows/systime-win/sys/time.h"
#endif

struct CPUTimer
{
#if _WIN32
	long start, end;
	struct timeval timecheck;
	int cpu_time_used; // milliseconds
	void tic(void)
	{
		my_gettimeofday(&timecheck, NULL);
		start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
	}

	void toc(bool is_print = true)
	{
		my_gettimeofday(&timecheck, NULL);
		end = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
		cpu_time_used = (end - start);
		if (is_print)
			printf("CPU Ellapsed Time: %d ms\n", cpu_time_used);
	}
#else
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
		if (is_print)
			printf("CPU Ellapsed Time: %d ms\n", cpu_time_used);
	}
#endif

};


#endif //_CPU_TIMER_H
