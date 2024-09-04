#ifndef _CPDF_2D_HPP_
#define _CPDF_2D_HPP_

#include "cauchy_estimator.hpp"


int cpdf_compare_dless(const void* p1, const void* p2)
{
    double a = *((double*)p1);
    double b = *((double*)p2);
    if(a > b)
        return 1;
    else if(a < b)
        return -1;
    else 
        return 0;
} 

// returns the angles of the cell walls for all cells encompassing A
void get_cell_wall_angles(double* thetas, double* A, const int m)
{
    const int d = 2;
    double point[d]; // point on hyperplane i
    double* a;
    int count = 0;
    for(int i = 0; i < m; i++)
    {
        a = A + i*d;
        if( fabs(a[0]) < fabs(a[1]) )
        {
            point[0] = 1;
            point[1] = -a[0] / a[1];
        }
        else 
        {
            point[0] = - a[1] / a[0];
            point[1] = 1;
        }
        double t1 = atan2(point[1], point[0]);
        if(t1 < 0)
            t1 += PI;
        double t2 = t1 + PI;
        thetas[count++] = t1;
        thetas[count++] = t2;
    }
    // Now sort the thetas, the theta at idx m corresponds to a half rotation around the arrangement
    qsort(thetas, 2*m, sizeof(double), &cpdf_compare_dless);
}

// Returns half of the SVs ( i.e, m/(2m) ),
// to get all the SVs, we simply flip these
// SVs is size 2*m x m 
// thetas 
void get_SVs_of_2D_HPA(double* SVs, double* A, double* thetas, const int m, const bool flip_svs = true)
{
    const int d = 2;
    double point[d];
    double* SV;
    for(int i = 0; i < m; i++)
    {
        double t = (thetas[i+1] + thetas[i]) / 2.0;
        point[0] = cos(t); point[1] = sin(t);
        SV = SVs + i*m;
        for(int j = 0; j < m; j++)
        {
            double* a = A + j*d;
            double sum = a[0] * point[0] + a[1] * point[1];
            SV[j] = sum > 0 ? 1 : -1;
        }
    }

    // Now flip these signs, they now correspond to full arrangement
    if(flip_svs)
    {
        SV = SVs + m*m;
        for(int i = 0; i < m*m; i++)
            SV[i] = -1*SVs[i];
    }
}

C_COMPLEX_TYPE ucpdf_2d_term_integral(C_COMPLEX_TYPE gamma1, C_COMPLEX_TYPE gamma2, double theta)
{
    return sin(theta) / (gamma1*gamma1*cos(theta) + gamma1*gamma2*sin(theta));
}

// Optimized version of eval_term_for_2D_ucpdf
C_COMPLEX_TYPE eval_term_for_2D_ucpdf_v2(double x1, double x2, double* A, double* p, double* b, GTABLE gtable, const int gtable_size, const int m)
{
    const int d = 2;
    const int two_m = 2*m;
    const int two_to_m_minus1 = 1 << (m-1);
    const int two_to_m = (1<<m);
    const int rev_m_mask = two_to_m - 1;
    int enc_sv;
    const double RECIPRICAL_TWO_PI = 1.0 / (2.0*PI);
    const double RECIPRICAL_TWO_PI_SQUARED = RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI;
    
	#if _WIN32 
		double thetas[2*MAX_HP_SIZE];
		double SVs[MAX_HP_SIZE*MAX_HP_SIZE];
		double A_scaled[MAX_HP_SIZE*MAX_HP_SIZE];
	#else
		double thetas[two_m];
		double SVs[m*m];
		double A_scaled[m*d];
	#endif 
	
	double* SV;
    double* a;
    get_cell_wall_angles(thetas, A, m); // angles of cell walls
    get_SVs_of_2D_HPA(SVs, A, thetas, m, false); // SVs corresponding to cells within the above (sequential) cell walls
    for(int i = 0; i < m; i++)
        for(int j = 0; j < d; j++)
            A_scaled[i*d + j] = A[i*d + j] * p[i];

    double gam1_real;
    double gam2_real;
    double gam1_imag = b[0] - x1;
    double gam2_imag = b[1] - x2;
    C_COMPLEX_TYPE g_val;
    C_COMPLEX_TYPE gamma1;
    C_COMPLEX_TYPE gamma2;
    C_COMPLEX_TYPE integral_low_lim;
    C_COMPLEX_TYPE integral_high_lim;
    C_COMPLEX_TYPE integral_over_cell;
    double term_integral = 0;
    double theta1;
    double theta2;
    double sin_t1;
    double cos_t1;
    double sin_t2;
    double cos_t2;
    for(int i = 0; i < m; i++)
    {

        // 1.) Encode SV and extract G in the i-th cell
        SV = SVs + i*m;
        enc_sv = 0;
        for(int j = 0; j < m; j++)
            if( SV[j] < 0 )
                enc_sv |= (1 << j);
        
        // Overloading numerator lookup function to lookup gtable Gs (just replace "phc" definitions with "m" definitions, which is done here)
        g_val = lookup_g_numerator(enc_sv, two_to_m_minus1, rev_m_mask, gtable, gtable_size, true);
        //g_val = enc_sv & two_to_m_minus1 ? conj(gtable[enc_sv ^ rev_m_mask]) : gtable[enc_sv];
        
        // Evaluating the piece-wise integral within this cell

        // 2.) Evaluate the real part of gamma1 parameter and real part of gamma2 parameter
        gam1_real = 0;
        gam2_real = 0;
        for(int j = 0; j < m; j++)
        {
            a = A_scaled + j*d;
            gam1_real -= a[0] * SV[j];
            gam2_real -= a[1] * SV[j];
        }
        
        // 3.) Form gamma1 and gamma2 and evaluate the analytic form of the integral
        //gamma1 = gam1_real + I*gam1_imag;
        //gamma2 = gam2_real + I*gam2_imag;
        gamma1 = MAKE_CMPLX(gam1_real, gam1_imag);
        gamma2 = MAKE_CMPLX(gam2_real, gam2_imag);

        // Faster integral, some caching has been added
        //sin(theta) / (gamma1*gamma1*cos(theta) + gamma1*gamma2*sin(theta));
        gamma2 *= gamma1;
        gamma1 *= gamma1;
        theta1 = thetas[i];
        theta2 = thetas[i+1];
        sin_t1 = sin(theta1);
        cos_t1 = cos(theta1);
        sin_t2 = sin(theta2);
        cos_t2 = cos(theta2);

        integral_low_lim = sin_t1 / (gamma1*cos_t1 + gamma2*sin_t1);
        integral_high_lim = sin_t2 / (gamma1*cos_t2 + gamma2*sin_t2);
        integral_over_cell = integral_high_lim - integral_low_lim;
        integral_over_cell *= g_val;
        term_integral += creal(integral_over_cell);
    }
    //return (2 * term_integral * RECIPRICAL_TWO_PI_SQUARED) + 0*I;
    return MAKE_CMPLX(2 * term_integral * RECIPRICAL_TWO_PI_SQUARED, 0);
}

// Non-Optimized version of eval_term_for_2D_ucpdf
C_COMPLEX_TYPE _eval_term_for_2D_ucpdf_v2(double x1, double x2, double* A, double* p, double* b, GTABLE gtable, const int gtable_size, const int m)
{
    const int d = 2;
    const int two_m = 2*m;
    const int two_to_m_minus1 = 1 << (m-1);
    const int two_to_m = (1<<m);
    const int rev_m_mask = two_to_m - 1;
    int enc_sv;
    const double RECIPRICAL_TWO_PI = 1.0 / (2.0*PI);
    const double RECIPRICAL_TWO_PI_SQUARED = RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI;
	#if _WIN32
		double thetas[2*MAX_HP_SIZE + 1];
		double A_scaled[MAX_HP_SIZE*MAX_HP_SIZE];
		double SVs[2*MAX_HP_SIZE*MAX_HP_SIZE];
	#else
		double thetas[two_m + 1];
		double A_scaled[m*d];
		double SVs[2 * m*m];
	#endif 
	double* SV;
    double* a;
    get_cell_wall_angles(thetas, A, m); // angles of cell walls
    get_SVs_of_2D_HPA(SVs, A, thetas, m, true); // SVs corresponding to cells within the above (sequential) cell walls
    thetas[two_m] = thetas[0] + 2*PI;
    for(int i = 0; i < m; i++)
        for(int j = 0; j < d; j++)
            A_scaled[i*d + j] = A[i*d + j] * p[i];

    double gam1_real;
    double gam2_real;
    double gam1_imag = b[0] - x1;
    double gam2_imag = b[1] - x2;
    C_COMPLEX_TYPE g_val;
    C_COMPLEX_TYPE gamma1;
    C_COMPLEX_TYPE gamma2;
    C_COMPLEX_TYPE integral_low_lim;
    C_COMPLEX_TYPE integral_high_lim;
    C_COMPLEX_TYPE integral_over_cell;
    C_COMPLEX_TYPE term_integral = 0;
    double theta1;
    double theta2;
    double sin_t1;
    double cos_t1;
    double sin_t2;
    double cos_t2;
    for(int i = 0; i < 2*m; i++)
    {

        // 1.) Encode SV and extract G in the i-th cell
        SV = SVs + i*m;
        enc_sv = 0;
        for(int j = 0; j < m; j++)
            if( SV[j] < 0 )
                enc_sv |= (1 << j);
        
        // Overloading numerator lookup function to lookup gtable Gs (just replace "phc" definitions with "m" definitions, which is done here)
        g_val = lookup_g_numerator(enc_sv, two_to_m_minus1, rev_m_mask, gtable, gtable_size, true);
        //g_val = enc_sv & two_to_m_minus1 ? conj(gtable[enc_sv ^ rev_m_mask]) : gtable[enc_sv];
        
        // Evaluating the piece-wise integral within this cell

        // 2.) Evaluate the real part of gamma1 parameter and real part of gamma2 parameter
        gam1_real = 0;
        gam2_real = 0;
        for(int j = 0; j < m; j++)
        {
            a = A_scaled + j*d;
            gam1_real -= a[0] * SV[j];
            gam2_real -= a[1] * SV[j];
        }
        
        // 3.) Form gamma1 and gamma2 and evaluate the analytic form of the integral
        //gamma1 = gam1_real + I*gam1_imag;
        //gamma2 = gam2_real + I*gam2_imag;
        gamma1 = MAKE_CMPLX(gam1_real, gam1_imag);
        gamma2 = MAKE_CMPLX(gam2_real, gam2_imag);

        // Faster integral, some caching has been added
        //sin(theta) / (gamma1*gamma1*cos(theta) + gamma1*gamma2*sin(theta));
        gamma2 *= gamma1;
        gamma1 *= gamma1;
        theta1 = thetas[i];
        theta2 = thetas[i+1];
        sin_t1 = sin(theta1);
        cos_t1 = cos(theta1);
        sin_t2 = sin(theta2);
        cos_t2 = cos(theta2);

        integral_low_lim = sin_t1 / (gamma1*cos_t1 + gamma2*sin_t1);
        integral_high_lim = sin_t2 / (gamma1*cos_t2 + gamma2*sin_t2);
        integral_over_cell = integral_high_lim - integral_low_lim;
        integral_over_cell *= g_val;
        term_integral += integral_over_cell;
    }
    return term_integral * RECIPRICAL_TWO_PI_SQUARED;
}

// Evaluates the un-normalized conditional pdf the CF generates
double eval_2D_ucpdf_at_x(double x1, double x2, CauchyEstimator* cauchyEst)
{
    double norm_factor = creal(cauchyEst->fz); // should always be one, or veryy close
    //C_COMPLEX_TYPE term_ucpdf_val = 0 + 0*I;
    //C_COMPLEX_TYPE ucpdf_val = 0 + 0*I;
    C_COMPLEX_TYPE term_ucpdf_val = MAKE_CMPLX(0,0);
    C_COMPLEX_TYPE ucpdf_val = MAKE_CMPLX(0,0);

    for(int m = 1; m < cauchyEst->shape_range; m++)
    {
        CauchyTerm* terms = cauchyEst->terms_dp[m];
        for(int i = 0; i < cauchyEst->terms_per_shape[m]; i++)
        {
            // After measurement update, the current gtable is located in the "_p" pointer...i.e, gtable_p (as well as cells_gtable ->cells_gtable_p)
            CauchyTerm* term = terms + i;
            term_ucpdf_val = eval_term_for_2D_ucpdf_v2(x1, x2, term->A, term->p, term->b, term->gtable_p, term->cells_gtable_p, m);
            ucpdf_val += term_ucpdf_val;
        }
    }
    double real_ucpdf_val_at_x1_x2 = creal(ucpdf_val) / norm_factor;
    if( fabs(cimag(ucpdf_val) / norm_factor) > 1e-10)
        printf("[Warn: PDF Integration] At x=%.2lf, y=%.2lf, pdf real part is: %lf,  imag part is: %.3E\n", x1, x2, real_ucpdf_val_at_x1_x2, cimag(ucpdf_val) / norm_factor);
    return real_ucpdf_val_at_x1_x2;
}

struct ThreadedEvalPDF
{
    int start;
    int end;
    CauchyEstimator* cauchyEst;
    CauchyPoint3D* cpdf_points;
};

void* threaded_evaluate_point_wise_cpdf(void* arg)
{   
    ThreadedEvalPDF* tepdf_arg = (ThreadedEvalPDF*) arg;
    int start = tepdf_arg->start;
    int end = tepdf_arg->end;
    CauchyEstimator* cauchyEst = tepdf_arg->cauchyEst;
    CauchyPoint3D* cpdf_points = tepdf_arg->cpdf_points;
    for(int i = start; i < end; i++)
        cpdf_points[i].z = eval_2D_ucpdf_at_x(cpdf_points[i].x, cpdf_points[i].y, cauchyEst);
    return NULL;
}

struct PointWise2DCauchyCPDF
{
    double gridx_low;
    double gridx_high;
    double gridx_resolution;
    uint num_gridx;
    double gridy_low;
    double gridy_high;
    double gridy_resolution;
    uint num_gridy;
    CauchyPoint3D* cpdf_points;
    char* dirname;
    int len_dirname;
    char* cpdf_fname; //"cpdf_";
    char* grid_sizes_fname; //"cpdf_grid_elems.txt"


    PointWise2DCauchyCPDF(
                char* _dirname,
                double _gridx_low, 
                double _gridx_high,
                double _gridx_resolution,
                double _gridy_low,
                double _gridy_high,
                double _gridy_resolution)
    {
        gridx_low = _gridx_low;
        gridx_high = _gridx_high;
        gridx_resolution = _gridx_resolution;
        gridy_low = _gridy_low;
        gridy_high = _gridy_high;
        gridy_resolution = _gridy_resolution;
        assert(gridx_high > gridx_low);
        assert(gridy_high > gridy_low);
        assert(gridx_resolution > 0);
        assert(gridy_resolution > 0);

        num_gridx = ((uint) ( (gridx_high - gridx_low + gridx_resolution - 1e-15) / gridx_resolution )) + 1;
        num_gridy = ((uint) ( (gridy_high - gridy_low + gridy_resolution - 1e-15) / gridy_resolution )) + 1;

        cpdf_points = (CauchyPoint3D*) malloc( num_gridx * num_gridy * sizeof(CauchyPoint3D) );
        null_ptr_check(cpdf_points);

        // If the dirname directory already exists, remove its contents
        if(_dirname != NULL)
        {
            dirname = (char*) malloc( (strlen(_dirname)+1) * sizeof(char));
            strcpy(dirname, _dirname);
            len_dirname = strlen(dirname);
            if(dirname[len_dirname-1] == '/')
            {
                dirname[len_dirname-1] = '\0';
                len_dirname -= 1;
            }
            cpdf_fname = (char*) malloc(len_dirname + 25 * sizeof(char)); //"cpdf_";
            grid_sizes_fname = (char*) malloc(len_dirname + 35 * sizeof(char)); // "cpdf_grid_elems.txt"
            sprintf(grid_sizes_fname, "%s/%s", dirname, "cpdf_grid_elems.txt");
            DIR* dir = opendir(dirname);
            // if it exists, delete all data generated by this structure
            // otherwise, just create the directory
            if(dir)
            {
                int status = remove(grid_sizes_fname);
                if(status == -1)
                {
                    printf("[Warn: pdf_2d] The file %s was not found...although it should in this directory...\n", grid_sizes_fname);
                }
            }
            else 
            {
                // Directory doesnt exist, create the directory
				int success;
				#if (__linux__ || __APPLE__)
					success = mkdir(dir_path, 0777);
				#else 
					success = _mkdir(dirname);
				#endif
                if(success == -1)
                {
                    printf("Failure making the directory %s. mkdir returns %d. Exiting!\n", dirname, success);
                    assert(false);
                }
            }
            closedir(dir);
        }
        else 
        {
            dirname = NULL;
            cpdf_fname = NULL;
            grid_sizes_fname = NULL;
        }
        // Set x-y points on rectangular grid, defined by user
        for(uint i = 0; i < num_gridy; i++)
        {
            for(uint j = 0; j < num_gridx; j++)
            {
                double point_x = gridx_low + j * gridx_resolution;
                double point_y = gridy_low + i * gridy_resolution;
                point_x = point_x > gridx_high ? gridx_high : point_x;
                point_y = point_x > gridx_high ? gridx_high : point_y;
                uint cpdf_idx = i*num_gridx + j;
                cpdf_points[cpdf_idx].x = point_x;
                cpdf_points[cpdf_idx].y = point_y;
                cpdf_points[cpdf_idx].z = -1;
            }
        }
    }

    void reset_grid(double _gridx_low, double _gridx_high, double _gridx_resolution, double _gridy_low, double _gridy_high, double _gridy_resolution)
    {
        if( (gridx_low != _gridx_low) || (gridx_high != _gridx_high) || (gridx_resolution != _gridx_resolution) ||
            (gridy_low != _gridy_low) || (gridy_high != _gridy_high) || (gridy_resolution != _gridy_resolution) )
        {
            gridx_low = _gridx_low;
            gridx_high = _gridx_high;
            gridx_resolution = _gridx_resolution;
            gridy_low = _gridy_low;
            gridy_high = _gridy_high;
            gridy_resolution = _gridy_resolution;
            assert(gridx_high > gridx_low);
            assert(gridy_high > gridy_low);
            assert(gridx_resolution > 0);
            assert(gridy_resolution > 0);
            num_gridx = ((uint) ( (gridx_high - gridx_low + gridx_resolution - 1e-15) / gridx_resolution )) + 1;
            num_gridy = ((uint) ( (gridy_high - gridy_low + gridy_resolution - 1e-15) / gridy_resolution )) + 1;
            cpdf_points = (CauchyPoint3D*) realloc(cpdf_points, num_gridx * num_gridy * sizeof(CauchyPoint3D) );
            null_ptr_check(cpdf_points);

            // Set x-y points on rectangular grid, defined by user
            for(uint i = 0; i < num_gridy; i++)
            {
                for(uint j = 0; j < num_gridx; j++)
                {
                    double point_x = gridx_low + j * gridx_resolution;
                    double point_y = gridy_low + i * gridy_resolution;
                    point_x = point_x > gridx_high ? gridx_high : point_x;
                    point_y = point_x > gridx_high ? gridx_high : point_y;
                    uint cpdf_idx = i*num_gridx + j;
                    cpdf_points[cpdf_idx].x = point_x;
                    cpdf_points[cpdf_idx].y = point_y;
                    cpdf_points[cpdf_idx].z = 0;
                }
            }
        }
    }

    // Single threaded evaluation
    int evaluate_point_wise_cpdf(CauchyEstimator* cauchyEst)
    {
        if( (cauchyEst->master_step == cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(RED "[Error CPDF Evaluation:] Cannot evaluate cauchy estimator cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }
        const int end = num_gridy * num_gridx;
        // Initialize the x-y locations of these points
        for(int i = 0; i < end; i++)
            cpdf_points[i].z = eval_2D_ucpdf_at_x(cpdf_points[i].x, cpdf_points[i].y, cauchyEst);
        return 0;
    }

    // Calls threaded version of the above function
    int evaluate_point_wise_cpdf(CauchyEstimator* cauchyEst, const int num_threads)
    {
        assert(num_threads > 1);
        if( (cauchyEst->master_step == cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(RED "[Error CPDF Evaluation:] Cannot evaluate cauchy estimator cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }

        const uint points_per_thread = (num_gridx * num_gridy) / num_threads;
        const uint last_thread_extra = (num_gridx * num_gridy) % num_threads;
        assert((points_per_thread*num_threads + last_thread_extra) == (num_gridx * num_gridy) );
        pthread_t* tids = (pthread_t*) malloc(num_threads*sizeof(pthread_t));
        ThreadedEvalPDF* tepdf_args = (ThreadedEvalPDF*) malloc(num_threads*sizeof(ThreadedEvalPDF));
        for(int i = 0; i < num_threads; i++)
        {
            tepdf_args[i].start = i*points_per_thread;
            tepdf_args[i].end = (i+1)*points_per_thread;
            if(i==(num_threads-1))
                tepdf_args[i].end += last_thread_extra;
            tepdf_args[i].cauchyEst = cauchyEst;
            tepdf_args[i].cpdf_points = cpdf_points;
        }

        for(int i = 0; i < num_threads; i++)
            pthread_create(&tids[i], NULL, &threaded_evaluate_point_wise_cpdf, &tepdf_args[i]);
        for(int i = 0; i < num_threads; i++)
            pthread_join(tids[i], NULL);
		free(tepdf_args);
		free(tids);
        return 0;
    }

    void store_2d_cpdf(const int cpdf_idx)
    {
        if(dirname == NULL)
        {
            printf(RED "[Error CPDF Storage]: Cannot store data, log directory not provided!" NC "\n");
            return;
        }
        assert(cpdf_idx >= 0);
        sprintf(cpdf_fname, "%s/%s%d%s", dirname, "cpdf_", cpdf_idx, ".bin");

        FILE* f_cpdf = fopen(cpdf_fname, "wb");
        if(f_cpdf == NULL)
        {
            printf("Could not store data at %s\n!", cpdf_fname);
            exit(1);
        }
        FILE* f_sizes = fopen(grid_sizes_fname, "a");
        if(f_sizes == NULL)
        {
            printf("Could not store data at %s\n!", grid_sizes_fname);
            exit(1);
        }
        // If data directoryies were created successfully, now write out the data
        fwrite(cpdf_points, num_gridx * num_gridy, sizeof(CauchyPoint3D), f_cpdf);
        char str_idx_and_gridpoints[20];
        sprintf(str_idx_and_gridpoints, "%d,%d,%d\n", cpdf_idx, num_gridx , num_gridy);
        fwrite(str_idx_and_gridpoints, strlen(str_idx_and_gridpoints), sizeof(char), f_sizes);
        fclose(f_cpdf);
        fclose(f_sizes);
    }

    ~PointWise2DCauchyCPDF()
    {
        free(cpdf_points);
        if(dirname != NULL)
            free(dirname);
        if(cpdf_fname != NULL)
            free(cpdf_fname);
        if(grid_sizes_fname != NULL)
            free(grid_sizes_fname);
    }
    
}; 


#endif //_CPDF_2D_HPP_