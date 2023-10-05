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
const double COALIGN_MU_EPS = 1e-8;
const bool SKIP_LAST_STEP = false;
const bool WITH_MSMT_UPDATE_ORTHOG_WARNING = true;

// Shared TP and MU Coalignment Settings
const uint8_t COALIGN_MAP_NOVAL = 255;
const bool WITH_COALIGN_REALLOC = true;

// Chunked Packed Storage Settings
const unsigned long long CP_STORAGE_PAGE_SIZE = 5 * 1024 * 1024; // Should be around 50 MB or larger (can be like 250MB or 1G too)
const int CP_STORAGE_ALLOC_METHOD = 0; // 0: malloc, 1: calloc 2: page touching (after a malloc)

// Gtable-Settings
const uint8_t kByteEmpty = 0xff;
const uint32_t kEmpty = 0xffffffff;
const bool WITH_TERM_APPROXIMATION = false;
const double TERM_APPROXIMATION_EPS = 1e-18;
const bool WITH_WARNINGS = true; // Used to flag when something may be fishy
const bool WITH_GB_TABLE_REALLOC = true;
const bool EXIT_ON_FAILURE = true;
bool INTEGRABLE_FLAG = true;

// Differential Cell Enumeration Settings
const int DCE_STORAGE_MULT = 4;
const bool FAST_TP_DCE = false; // If true, we only perturb Gamma, and use moshe's new method to do so.
const bool STABLE_SOLVE = true; // setting to false increases speed but uses a very weak condition number approximation
const double GAMMA_PERTURB_EPS = 1; // Must be positive
const double PLU_EPS = 1e-15;
const double COND_EPS = 1e12;

// Fast term reduction settings
const double REDUCTION_EPS = 1e-8; // 1e-8
const bool WITH_FTR_P_CHECK = true;
const bool WITH_FTR_HPA_CHECK = true;
int TR_SEARCH_IDXS_ORDERING[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

// Distributed Computation Settings
const int NUM_CPUS = 8;
const int MIN_TERMS_PER_THREAD_TP_TO_MUC = 200;


// Log Error Warnings and Colors
#define RED "\e[0;31m"
#define NC "\e[0m"
#define YEL  "\e[0;33m" 

#endif