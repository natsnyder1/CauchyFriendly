#ifndef _ARRAY_LOGGING_HPP_
#define _ARRAY_LOGGING_HPP_

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

// Return Codes
// 0: Directory does not exist
// 1: Directory exists
void check_dir_and_create(char* dir_path)
{
   // If this directory does not exist, create this directory
  // check if window folder exists
  DIR* dir = opendir(dir_path);
  if(dir)
  {
    // The directory exists
    //return 1;
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
    //return 0;
  }
  else
  {
    // Directory opening failed for some reason
    printf("Directory opening failed!\n");
    assert(false);
  }
  if( dir != NULL)
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

// For all window data
void log_int_array_to_file(FILE* f_name, int* x, const int len_x)
{
	if(len_x == 1)
		fprintf(f_name, "%d\n", x[0]);
	else
	{
		for(int i = 0; i < len_x-1; i++)
			fprintf(f_name, "%d ", x[i]);
		fprintf(f_name, "%d\n", x[len_x-1]);
	}
}

// For full/best window data
void log_int_array_to_file(FILE* f_name, int win_idx, int* x, const int len_x)
{
	fprintf(f_name, "%d:", win_idx);
	log_int_array_to_file(f_name, x, len_x);
}

// Takes in a path to a file and if it exists, logs the M x N double data their
void log_double_array_to_file(char* fpath, double* data, const int m, const int n)
{
    FILE* f_data = fopen(fpath, "w");
    if(fpath == NULL)
    {     
      printf("[ERROR log_double_array_to_file: Path %s likely does not exist (fpath == NULL). Please inspect path carefully! Exiting!]\n", fpath);
      exit(1);
    }

    for(int i = 0; i < m; i++)
      log_double_array_to_file(f_data, data + i*n, n);
    fclose(f_data);
}

void log_kf_data(char* log_dir, double* kf_state_history, double* kf_covar_history, double* kf_residual_history, const int total_steps, const int state_dim, const int p)
{
  const int len_log_dir = strlen(log_dir);
  assert(len_log_dir < 4000);
  char* temp_path = (char*) malloc(4096);

  char slash_delim[2] = "/";
  char no_delim[2] = "";
  char* delim = log_dir[len_log_dir-1] == '/' ? no_delim : slash_delim;

  sprintf(temp_path, "%s%s%s", log_dir, delim, "kf_cond_means.txt");
  log_double_array_to_file(temp_path, kf_state_history, total_steps, state_dim);
  sprintf(temp_path, "%s%s%s", log_dir, delim, "kf_cond_covars.txt");
  log_double_array_to_file(temp_path, kf_covar_history, total_steps, state_dim*state_dim);
  sprintf(temp_path, "%s%s%s", log_dir, delim, "kf_residuals.txt");
  log_double_array_to_file(temp_path, kf_residual_history, total_steps-1, p);  
  free(temp_path);
}

#endif