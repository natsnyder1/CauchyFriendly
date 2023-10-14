#ifndef _ARRAY_LOGGING_HPP_
#define _ARRAY_LOGGING_HPP_

#include <assert.h>
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

void check_dir_and_create(char* dir_path)
{
   // If this directory does not exist, create this directory
  // check if window folder exists
  DIR* dir = opendir(dir_path);
  if(dir)
  {
    // The directory exists
  }
  else if(ENOENT == errno)
  {
    // Directory doesnt exist, create the directory
    int success = mkdir(dir_path, 0777);
    if(success == -1)
    {
      printf("Failure making the directory %s. mkdir returns %d. Exiting!\n", dir_path, success);
      assert(false);
    }
  }
  else
  {
    // Directory opening failed for some reason
    printf("Directory opening failed!\n");
    assert(false);
  }
  closedir(dir);
}

// For all window data
void log_double_array_to_file(FILE* f_name, double* x, const int len_x)
{
	if(len_x == 1)
		fprintf(f_name, "%.16lf\n", x[0]);
	else
	{
		for(int i = 0; i < len_x-1; i++)
			fprintf(f_name, "%.16lf ", x[i]);
		fprintf(f_name, "%.16lf\n", x[len_x-1]);
	}
}

// For full/best window data
void log_double_array_to_file(FILE* f_name, int win_idx, double* x, const int len_x)
{
	fprintf(f_name, "%d:", win_idx);
	log_double_array_to_file(f_name, x, len_x);
}

#endif