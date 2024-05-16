#ifndef _CAUCHY_CONSTANTS_HPP_
#define _CAUCHY_CONSTANTS_HPP_

#include<stdint.h>
#include<assert.h>
#include<complex.h>
#include<math.h>

#define PI M_PI
const double RECIPRICAL_TWO_PI = 1.0 / (2.0 * PI);
typedef double __complex__ C_COMPLEX_TYPE; // This is actually buggy, and cannot change for the moment


// TP Settings
const double COALIGN_TP_EPS = 1e-8;
// MU Settings
const double MU_EPS = 1e-10;
const double COALIGN_MU_EPS = COALIGN_TP_EPS;
const bool SKIP_LAST_STEP = true; // SHOULD BE SET TO TRUE UNLESS OTHERWISE NEEDED
const bool WITH_MSMT_UPDATE_ORTHOG_WARNING = false;

// Shared TP and MU Coalignment Settings
const uint8_t COALIGN_MAP_NOVAL = 255;
const bool WITH_COALIGN_REALLOC = false;

// Chunked Packed Storage Settings
const unsigned long long CP_STORAGE_PAGE_SIZE = 10 * 1024 * 1024; // Should be around 50 MB or larger (can be like 250MB or 1G too)
const int CP_STORAGE_ALLOC_METHOD = 0; // 0: malloc, 1: calloc 2: page touching (after a malloc)

// Gtable-Settings
const uint8_t kByteEmpty = 0xff;
const uint32_t kEmpty = 0xffffffff;
const bool WITH_TERM_APPROXIMATION = false;
const double TERM_APPROXIMATION_EPS = 1e-16;
const bool WITH_WARNINGS = false; // Used to flag when something may be fishy
const bool WITH_GB_TABLE_REALLOC = true;
const bool EXIT_ON_FAILURE = false; // Used to flag when something doesnt go right
bool INTEGRABLE_FLAG = true;

// Differential Cell Enumeration Settings
const int DCE_STORAGE_MULT = 4;
const bool FAST_TP_DCE = false; // If true, we only perturb Gamma, and use moshe's new method to do so.
const bool STABLE_SOLVE = true; // setting to false increases speed but uses a very weak condition number approximation
const double GAMMA_PERTURB_EPS = 1; // Must be positive
const double PLU_EPS = 1e-15; // LEO: 1e-15
const double COND_EPS = 1e12; // LEO: 1e15

// Fast term reduction settings
const double REDUCTION_EPS = 1e-8; // 1e-8
const bool WITH_FTR_P_CHECK = true;
const bool WITH_FTR_HPA_CHECK = true;
int TR_SEARCH_IDXS_ORDERING[12] = {0,1,2,3,4,5,6,7,8,9,10,11}; //{3,2,4,1,0}; //{5,4,3,6,2,1,0}; //{3,2,4,1,0}; //{0,1,2};

// Distributed Computation Settings
const int NUM_CPUS = 8; // Should be 128 on the cluster
const int MIN_TERMS_PER_THREAD_TP_TO_MUC = 200;
const int NUM_CPUS_FTR = NUM_CPUS; // Set this to 32 on the cluster (NUM_CPUS / FTR_CPU_DIVISOR = CPUS USED)
const int MIN_TERMS_PER_THREAD_FTR = 10000;
const int MIN_TERMS_PER_THREAD_GTABLE = 1000;


// Log Error Warnings and Colors
#define RED "\e[0;31m"
#define NC "\e[0m"
#define YEL  "\e[0;33m" 

// Numeric Moment Error Tolerances
const double THRESHOLD_FZ_IMAG_TO_REAL = 1e-3;
const double HARD_LIMIT_IMAGINARY_MEAN = 0.001; //0.1
const double THRESHOLD_MEAN_IMAG_TO_REAL = 1e-1;
const double HARD_LIMIT_IMAGINARY_COVARIANCE = 2000; // 0.75
const double THRESHOLD_COVARIANCE_IMAG_TO_REAL = 10; // Should be less than 1

// Numeric Covariance Error Flag Bits
const int COV_ERROR_FLAGS_INVALID_EIGENVALUES = 0;
const int COV_ERROR_FLAGS_INVALID_CORRELATION = 1;
const int COV_ERROR_FLAGS_INVALID_I2R_RATIO = 2;
const int COV_ERROR_FLAGS_INVALID_IMAGINARY_VALUE = 3;

// Numeric Covariance eigenvalue smallness tolerance
const double COV_EIGENVALUE_TOLERANCE = -1e-10; // Should be positive

// Numeric Moment Error Flag Bits
const int ERROR_FZ_NEGATIVE = 9;
const int ERROR_FZ_UNSTABLE = 8;

const int ERROR_MEAN_AT_CURRENT_STEP_DNE = 7;
const int ERROR_MEAN_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT = 6;
const int ERROR_MEAN_UNSTABLE_CURRENT_STEP_FINAL_MSMT = 5;
const int ERROR_MEAN_UNSTABLE_ANY_STEP = 4;

const int ERROR_COVARIANCE_AT_CURRENT_STEP_DNE = 3;
const int ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT = 2;
const int ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_FINAL_MSMT = 1;
const int ERROR_COVARIANCE_UNSTABLE_ANY_STEP = 0;

// Cauchy to Gaussian Conversion Parameter
const double CAUCHY_TO_GAUSS_NOISE = 1.3898;
const double GAUSS_TO_CAUCHY_NOISE = 1.0 / CAUCHY_TO_GAUSS_NOISE;

#endif
