#ifndef _CAUCHY_TERM_HPP_
#define _CAUCHY_TERM_HPP_

#include "cauchy_types.hpp"
#include "eval_gs.hpp"
#include "cpu_linalg.hpp"
#include "gtable.hpp"

struct ChildTermWorkSpace
{
    double* A;
    double* p;
    double* q;
    double* b;
    uint8_t* c_map;
    int8_t* cs_map;

    void init(const int max_shape, const int d)
    {
        int max_child_terms = max_shape + 1;
        A = (double*) malloc( max_child_terms * max_shape * d * sizeof(double));
        null_ptr_check(A);
        p = (double*) malloc( max_child_terms * max_shape * sizeof(double));
        null_ptr_check(p);
        q = (double*) malloc( max_child_terms * max_shape * sizeof(double));
        null_ptr_check(q);
        b = (double*) malloc( max_child_terms * max_shape * d * sizeof(double));
        null_ptr_check(b);
        c_map = (uint8_t*) malloc( max_child_terms * max_shape * sizeof(uint8_t));
        null_ptr_check(c_map);
        cs_map = (int8_t*) malloc( max_child_terms * max_shape * sizeof(int8_t));
        null_ptr_check(cs_map);
    }

    void deinit()
    {
        free(A);
        free(p);
        free(q);
        free(b);
        free(c_map);
        free(cs_map);
    }
};

struct CauchyTerm
{
    int m;
    int d;
    double* A;
    double* p;
    double* q;
    double* b;
    double c_val;
    double d_val;
    int cells_gtable_p;
    GTABLE gtable_p;
    int cells_gtable;
    BKEYS enc_B;
    GTABLE gtable;
    int enc_lhp;
    uint Horthog_flag;
    uint8_t* c_map;
    int8_t* cs_map;
    int8_t phc;
    uint8_t pbc;
    uint8_t z;
    bool is_new_child;

    // initialize the t-th child using temporary workspace
    void init_mem(ChildTermWorkSpace* workspace, const int t, int _m, int _d)
    {
        m = _m;
        d = _d;
        const int tm = t*m;
        A = workspace->A + tm*d;
        p = workspace->p + tm;
        q = workspace->q + tm;
        b = workspace->b + t*d;
        c_map = workspace->c_map + tm;
        cs_map = workspace->cs_map + tm;
    }
    
    int msmt_update(CauchyTerm* child_terms, double msmt, double* H, double gamma, bool first_update, bool last_update, ChildTermWorkSpace* workspace)
    {
        const int num_new_terms = m+1;
        // Run msmt update
        int sign_AH[m];
        bool F_integrable[m];
        // Create mu and rho terms
        const int rho_size = (m+1);
        const int mu_size = rho_size * d;
        double mu[mu_size];
        double rho[rho_size];
        memcpy(mu, A, m*d*sizeof(double));
        memcpy(rho, p, m*sizeof(double));
        rho[m] = gamma;
        memset(mu + m*d, 0, d*sizeof(double));
        F_integrable[m] = true;
        for(int l = 0; l < m; l++)
        {
            const int ld = l*d;
            double* mu_il = mu + ld;
            double H_mu_il = dot_prod(H, mu_il, d);
            double abs_H_mu_il = fabs(H_mu_il);
            if( abs_H_mu_il < MU_EPS )
            {
                if(WITH_MSMT_UPDATE_ORTHOG_WARNING)
                {
                    printf(RED "[WARNING MU:]" 
                        YEL "mu_il @ H < eps=%.4E for shape m=%d, hp=%d"
                        NC "\n", MU_EPS, m, l);
                }
                rho[l] = p[l];
                sign_AH[l] = 1;
                F_integrable[l] = 0;
            }
            else 
            {
                scale_vec(mu_il, 1.0 / H_mu_il, d);
                rho[l] = p[l] * abs_H_mu_il;
                sign_AH[l] = (H_mu_il > 0) ? 1 : -1;
                F_integrable[l] = 1;
            }
        }
        
        // Check to see if Gamma was orthogonal to H
        // Cannot integrate if this is the case
        if(!first_update)
        {
            for(int i = phc; i < m; i++)
                if(!F_integrable[i])
                    INTEGRABLE_FLAG = false; // Raise non-integrable flag
        }
        // For all integrable hyperplanes, create children using temporary memory space
        int num_integrable_terms = 0;
        for(int i = 0; i < m; i++)
        {
            if(F_integrable[i])
            {
                child_terms[num_integrable_terms].init_mem(workspace, num_integrable_terms+1, m, d);
                num_integrable_terms++;
            }
        }
        // Create zeta scalar using measurement
        double zeta = msmt - dot_prod(H, b, d);
        CauchyTerm* child;
        num_integrable_terms = 0;
        for(int t = 0; t < num_new_terms; t++)
        {
            if(F_integrable[t]) // only create child if the t-th hyperplane is not orthogonal to H
            {
                if(t==m)
                    child = this;
                else
                {
                    child = child_terms + num_integrable_terms;
                    num_integrable_terms++;
                }
                if(!first_update)
                {
                    // Set helper variables for children
                    child->gtable_p = gtable_p;
                    child->cells_gtable_p = cells_gtable_p;
                    child->phc = phc;
                    child->pbc = m;
                    child->z = t;
                }
                // Store scalar values for child terms c_t and d_t
                child->c_val = zeta;
                child->d_val = rho[t];
                // Helper variable for creation of b_t and A_t
                double* A_t = child->A;
                double* A_tl;
                double* p_t = child->p;
                double* mu_it = mu + t*d;
                double* mu_il;
                // Create vector term b_t = zeta * mu_it + b_i
                add_vecs(b, mu_it, child->b, d, zeta);
                // Create vector p_t and well as the matrix term A_t
                int l = 0;
                uint hofs_t = 0;
                for(int _l = 0; _l < m+1; _l++)
                {
                    if(_l != t)
                    {
                        A_tl = A_t + l*d;
                        mu_il = mu + _l*d;
                        p_t[l] = rho[_l];
                        if(F_integrable[_l])
                            sub_vecs(mu_il, mu_it, A_tl, d);
                        else 
                        {
                            memcpy(A_tl, mu_il, d * sizeof(double));
                            hofs_t |= (1<<l);
                        }
                        l += 1;
                    }
                }
                child->Horthog_flag = hofs_t;
            }
        }
        is_new_child = false;
        // If this is not the first estimation step, update the parent B using sign_AH 
        if(!first_update)
        {
            // NEW WAY -- MAY FIX ERROR, BUT NEEDS SOME TESTING
            int enc_sgn_AH = 0;
            int mask_last_bit = (1<<(m-1));
            for(int l = 0; l < m; l++)
                if(sign_AH[l] == -1)
                    enc_sgn_AH |= (1 << l);
            
            enc_lhp = enc_sgn_AH;
            if(phc < m)
            {
                int clear_bits_above_phc = (1<<phc)-1;
                enc_lhp &= clear_bits_above_phc;
            }
            if(HALF_STORAGE)
            {
                int mask_rev = (1<<m) - 1;
                if( enc_sgn_AH & mask_last_bit )
                    enc_sgn_AH ^= mask_rev;
            }

            for(int i = 0; i < num_integrable_terms; i++)
            {
                child_terms[i].enc_lhp = enc_lhp;
                child_terms[i].is_new_child = true;
                // These are brought in as input to the make new child btable function
                child_terms[i].enc_B = enc_B;
                child_terms[i].cells_gtable = cells_gtable;
            }
            // Update the parents B^{k|k-1} to B_mu^{k|k-1}
            if(!DENSE_STORAGE)
            {
                if(!last_update)
                    for(int i = 0; i < cells_gtable; i++)
                        enc_B[i] ^= enc_sgn_AH;
            }


            // OLD WAY -- I THINK THERE IS AN ERROR
            /*
            enc_lhp = 0;
            for(int l = 0; l < phc; l++)
                if(sign_AH[l] == -1)
                    enc_lhp |= (1 << l);
            for(int i = 0; i < num_integrable_terms; i++)
            {
                child_terms[i].enc_lhp = enc_lhp;
                child_terms[i].is_new_child = true;
                // These are brought in as input to the make new child btable function
                child_terms[i].enc_B = enc_B;
                child_terms[i].cells_gtable = cells_gtable;
            }
            // Update the parents B^{k|k-1} to B_mu^{k|k-1}
            if(!last_update && !DENSE_STORAGE)
            {
                // If we use full storage, then this will always work
                if(FULL_STORAGE)
                    for(int i = 0; i < cells_gtable; i++)
                        enc_B[i] ^= enc_lhp;
                // If we use half storage, then we need to consider the following conditions
                else
                {
                    // If phc < m, then we are okay, this means the last bit will never flip negative
                    if(phc < m)
                    {
                        for(int i = 0; i < cells_gtable; i++)
                            enc_B[i] ^= enc_lhp;
                    }
                    // phc == m
                    else
                    {   
                        // If phc == m, but enc_lhp has its (m-1)'th bit set, reverse enc_lhp when updating enc_B
                        // This is because enc_B must be positive in its (m-1)'th bit for half storage  
                        int two_to_m_minus1 = 1<<(m-1);
                        if(enc_lhp & two_to_m_minus1)
                        {
                            int rev_mask = (1<<m) - 1;
                            int rev_enc_lhp = enc_lhp ^ rev_mask;
                            for(int i = 0; i < cells_gtable; i++)
                                enc_B[i] ^= rev_enc_lhp;
                        }
                        // If phc == m, but enc_lhp does not have its (m-1)'th bit set, we are okay
                        else 
                        {
                            for(int i = 0; i < cells_gtable; i++)
                                enc_B[i] ^= enc_lhp;
                        }
                    }
                    
                }
                
            }
            */
        }
        return num_integrable_terms; // new children + old child
    }

    C_COMPLEX_TYPE eval_g_yei(double* root_point, C_COMPLEX_TYPE* yei, const bool first_update)
    {
        C_COMPLEX_TYPE g_num_p;
        C_COMPLEX_TYPE g_num_m;
        C_COMPLEX_TYPE g_val;
        double sign_A[m];
        double tmp_yei[d];
        double ygi;
        memset(tmp_yei, 0, d*sizeof(double));

        // If H_orthog flag is set, exclude As and ps where Horthog_flag has a bit set
        if(Horthog_flag)
        {
            ygi = 0;
            for(int l = 0; l < m; l++)
            {
                sign_A[l] = dot_prod(A + l*d, root_point, d) > 0 ? 1 : -1;
                add_vecs(tmp_yei, A + l*d, d, p[l] * sign_A[l]);
                if( !(Horthog_flag & (1<<l)) )
                {
                    ygi += p[l] * sign_A[l];
                }
            } 
        }
        // If H_orthog flag is not set, run normal formula
        else 
        {
            for(int l = 0; l < m; l++)
            {
                sign_A[l] = dot_prod(A + l*d, root_point, d) > 0 ? 1 : -1;
                add_vecs(tmp_yei, A + l*d, d, p[l] * sign_A[l]);
            } 
            ygi = dot_prod(p, sign_A, m);
        }

        if(first_update)
        {
            g_num_p = 1 + 0*I; //CMPLX(1,0);
            g_num_m = 1 + 0*I; //CMPLX(1,0);
        }
        else
        {
            const int two_to_phc_minus1 = (1<<(phc-1));
            const int rev_phc_mask = (1<<phc) - 1;  
            int enc_lp = 0;
            int enc_lm = 0;
            int k = 0;
            for(int l = 0; l < m; l++)
            { 
                // Form enc_lp and enc_lm
                if(k < phc)
                {
                    if(k == z)
                    {
                        enc_lm |= (1<<k);
                        k++;
                        if(k == phc)
                        continue;
                    }
                    if(sign_A[l] < 0)
                    {
                        enc_lp |= (1 << k);
                        enc_lm |= (1 << k);
                    }
                    k++;
                }
            }
            // Change key enc_lp and enc_lm by enc_lhp
            int size_gtable_p = GTABLE_SIZE_MULTIPLIER * cells_gtable_p;
            g_num_p = lookup_g_numerator(enc_lp ^ enc_lhp, two_to_phc_minus1, rev_phc_mask, gtable_p, size_gtable_p, true);
            g_num_m = lookup_g_numerator(enc_lm ^ enc_lhp, two_to_phc_minus1, rev_phc_mask, gtable_p, size_gtable_p, false);
        }
        // Compute G and add complex part to yei
        //g_val = g_num_p / CMPLX(ygi+d_val, c_val) -  g_num_m / CMPLX(ygi-d_val, c_val);
        g_val = g_num_p / (ygi+d_val + I*c_val) - g_num_m / (ygi-d_val + I*c_val);
        g_val *= RECIPRICAL_TWO_PI;
        for(int j = 0; j < d; j++)
            yei[j] = -tmp_yei[j] + I*b[j]; //yei[j] = CMPLX(-tmp_yei[j], b[j]);

        return g_val;
    }
  
    C_COMPLEX_TYPE eval_g_yei_after_ftr(double* root_point, C_COMPLEX_TYPE* yei)
    {
        int enc_sv = 0;
        double tmp_yei[d];
        memset(tmp_yei, 0, d*sizeof(double));

        for(int l = 0; l < m; l++)
        {
            double s = dot_prod(A + l*d, root_point, d) > 0 ? 1 : -1;
            add_vecs(tmp_yei, A + l*d, d, p[l] * s);
            if(s < 0)
                enc_sv |= 1<<l;
        } 
        // overloading g_num_hashtable lookup here for gtable
        // gtable is located in gtable_p after FTR (due to pointer swap for next step)
        const int two_to_m_minus1 = 1<<(m-1);
        const int rev_mask = (1<<m)-1;
        C_COMPLEX_TYPE g_val = lookup_g_numerator(enc_sv, two_to_m_minus1, rev_mask, gtable_p, cells_gtable_p * GTABLE_SIZE_MULTIPLIER, true);

        for(int j = 0; j < d; j++)
            yei[j] = -tmp_yei[j] + I*b[j]; //yei[j] = CMPLX(-tmp_yei[j], b[j]);

        return g_val;
    }
    
    void time_prop(double* Phi, double* B, double* u, const int cmcc)
    {
        double work[m*d];
        memcpy(work, A, m * d * sizeof(double));
        matmatmul(work, Phi, A, m, d, d, d, false, true); // A @ Phi.T
        memcpy(work, b, d*sizeof(double));
        matvecmul(Phi, work, b, d, d, false); // Phi @ b == b @ Phi.T

        // Shift b_{k+1|k} by B @ u if a control was provided
        // In the nonlinear case, cmcc may be > 0, 
        // however, the sliding window manager / python do not allow controls 
        // controls must be updated in the deterministic part for nonlinear problems
        if( cmcc > 0)
        {
            matvecmul(B, u, work, d, cmcc);
            add_vecs(b, work, d, 1);
        }
    }

    void normalize_hps(const bool set_q)
    {
        if(set_q)
            memcpy(q, p, m*sizeof(double));
        double norm1;
        for(int i = 0; i < m; i++)
        {   
            norm1 = 0;
            for(int j = 0; j < d; j++)
                norm1 += fabs(A[i*d + j]);
            p[i] *= norm1;
            for(int j = 0; j < d; j++)
                A[i*d+j] /= norm1;
        }
    }

    // This function takes in Gamma^T and beta (both normalized and pre-coaligned)
    int tp_coalign(double* Gamma_T, double* beta, const int cmcc)
    {
        normalize_hps(false);
        bool F[cmcc];
        memset(F, 1, cmcc * sizeof(bool));
        for(int j = 0; j < cmcc; j++)
        {
            double* Gam_root = Gamma_T + j*d;
            for(int k = 0; k < m; k++)
            {
                if(F[j])
                {
                    double* A_compare = A + k*d;
                    bool pos_gate = true;
                    bool neg_gate = true;
                    for(int l = 0; l < d; l++)
                    {
                        double root = Gam_root[l];
                        double compare = A_compare[l];
                        if(pos_gate)
                            pos_gate &= fabs(root - compare) < COALIGN_MU_EPS;
                        if(neg_gate)
                            neg_gate &= fabs(root + compare) < COALIGN_MU_EPS;
                        if(!(pos_gate || neg_gate) )
                            break;
                    }   
                    assert(!(pos_gate && neg_gate));
                    if(pos_gate || neg_gate)
                    {
                        //printf("Term %d with shape %d has (pos) coalignment of (%d, %d)\n", i, m, j, k);
                        F[j] = 0;
                        p[k] += beta[j];
                    }
                }
            }
        }
        int new_shape = m;
        for(int i = 0; i < cmcc; i++)
            new_shape += F[i];

        if(new_shape > m)
        {
            for(int i = 0; i < cmcc; i++)
            {
                if(F[i])
                {
                    memcpy(A + m*d, Gamma_T + i*d, d*sizeof(double));
                    p[m++] = beta[i];
                }
            }
        }
        return m;
    }

    int mu_coalign()
    {
        normalize_hps(true);
        bool F[m];
        bool Horthog_flag_unencoded[m];
        memset(F, 1, m);
        memset(c_map, COALIGN_MAP_NOVAL, m);
        memset(cs_map, 1, m);
        if(Horthog_flag)
            for(int j = 0; j < m; j++)
                Horthog_flag_unencoded[j] = Horthog_flag & (1 << j);

        int unique_count = 0;
        for(int j = 0; j < m-1; j++)
        {
            if(F[j])
            {
                double* A_root = A + j*d;
                c_map[j] = unique_count;
                for(int k = j+1; k < m; k++)
                {
                    if(F[k])
                    {
                        double* A_compare = A + k*d;
                        bool pos_gate = true;
                        bool neg_gate = true;
                        for(int l = 0; l < d; l++)
                        {
                            double root = A_root[l];
                            double compare = A_compare[l];
                            if(pos_gate)
                                pos_gate &= fabs(root - compare) < COALIGN_MU_EPS;
                            if(neg_gate)
                                neg_gate &= fabs(root + compare) < COALIGN_MU_EPS;
                            if(!(pos_gate || neg_gate) )
                                break;
                        }   
                        assert(!(pos_gate && neg_gate));
                        if(pos_gate)
                        {
                            if(Horthog_flag)
                            {
                                bool hflag_j = Horthog_flag_unencoded[j];
                                bool hflag_k = Horthog_flag_unencoded[k];
                                // If hyperplane at index j is not orthog to H...
                                // and hyperplane at index k is not orthog to H...
                                // set q[j] += q[j]
                                // no flag in Horthog_flag_unencoded need be modified
                                if( (!hflag_j) && (!hflag_k) )
                                    q[j] += q[k];
                                // If hyperplane at index j is orthog to H...
                                // and hyperplane at index k is not orthog to H...
                                // set q[j] = q[k]...
                                // discard the flag at Horthog_flag_unencoded[j]
                                else if( hflag_j && (!hflag_k) )
                                {
                                    q[j] = q[k];
                                    Horthog_flag_unencoded[j] = false;
                                }
                                // If hyperplane at index j is not orthog to H...
                                // and hyperplane at index k is orthog to H...
                                // keep q[j] = q[j]... do not add q[k] to q[j]
                                // discard the flag at Horthog_flag_unencoded[k]
                                else if(  (!hflag_j) && hflag_k )
                                {
                                    //q[j] = q[j];
                                    Horthog_flag_unencoded[k] = false;
                                }
                                // If both hyperplanes at index j and k are orthog to H...
                                // and somehow they coalign...there is an error...
                                // these HPs should have been removed during TP coalignment...
                                // raise error and exit(1)
                                else 
                                {
                                    printf(YEL "[WARN MUC #1:]" 
                                        YEL "Shape %d, HP1 index=%d, HP2 index=%d.\n"
                                        YEL "IT WAS SEEN THAT TWO HPS ORTHOGONAL TO H (BEFORE MSMT UPDATE) ARE COALIGNING.\n"
                                        YEL "THESE HPS SHOULD HAVE BEEN COALIGNED DURING COALIGNMENT AFTER TIME PROPAGATION (TPC).\n"
                                        YEL "THIS INDICATES EITHER AN ERROR IN THE CODE, OR THAT THE TPC EPSILON (%.3E) MAY BE SMALLER THAN IT SHOULD BE (THE MUC EPSILON IS (%.3E)."
                                        YEL "THIS ERROR CAN BE DISCARDED IF THIS NUMERICAL ERROR IS NOT OF RELEVANCE (COMMENT THIS WARNINH OUT)"
                                        YEL "PLEASE PLACE BREAKPOINT HERE AND DEBUG FURTHER IF DESIRED, OR REMOVE ERROR MESSAGE! EXITING FOR NOW! GOODBYE!"
                                        NC "\n", m, j, k, COALIGN_TP_EPS, COALIGN_MU_EPS);
                                    //sleep(2);
                                    //exit(1);
                                    // If the above error is commented, uncomment the line below
                                    Horthog_flag_unencoded[k] = false;
                                    // Do not really need this line of code....
                                    // It will not be summed over, due to the Horthog_flag being 1 at index j
                                    q[j] += q[k];
                                }
                            }
                            else
                            {
                                q[j] += q[k];
                            }
                            //printf("Term %d with shape %d has (pos) coalignment of (%d, %d)\n", i, m, j, k);
                            F[k] = 0;
                            p[j] += p[k];
                            c_map[k] = unique_count;
                            cs_map[k] = 1; 
                        }
                        if(neg_gate)
                        {
                            if(Horthog_flag)
                            {
                                bool hflag_j = Horthog_flag_unencoded[j];
                                bool hflag_k = Horthog_flag_unencoded[k];
                                // If hyperplane at index j is not orthog to H...
                                // and hyperplane at index k is not orthog to H...
                                // set q[j] -= q[j]
                                // no flag in Horthog_flag_unencoded need be modified
                                if( (!hflag_j) && (!hflag_k) )
                                    q[j] -= q[k];
                                // If hyperplane at index j is orthog to H...
                                // and hyperplane at index k is not orthog to H...
                                // set q[j] = -q[k]...
                                // discard the flag at Horthog_flag_unencoded[j]
                                else if( hflag_j && (!hflag_k) )
                                {
                                    q[j] = -q[k];
                                    Horthog_flag_unencoded[j] = false;
                                }
                                // If hyperplane at index j is not orthog to H...
                                // and hyperplane at index k is orthog to H...
                                // keep q[j] = q[j]... do not add q[k] to q[j]
                                // discard the flag at Horthog_flag_unencoded[k]
                                else if(  (!hflag_j) && hflag_k )
                                {
                                    //q[j] = q[j];
                                    Horthog_flag_unencoded[k] = false;
                                }
                                // If both hyperplanes at index j and k are orthog to H...
                                // and somehow they coalign...there is an error...
                                // these HPs should have been removed during TP coalignment...
                                // raise error and exit(1)
                                else 
                                {
                                    printf(RED "[ERROR MUC #2:]" 
                                        YEL "Shape %d, HP1 index=%d, HP2 index=%d.\n"
                                        YEL "IT WAS SEEN THAT TWO HPS ORTHOGONAL TO H (BEFORE MSMT UPDATE) ARE COALIGNING.\n"
                                        YEL "THESE HPS SHOULD HAVE BEEN COALIGNED DURING COALIGNMENT AFTER TIME PROPAGATION (TPC).\n"
                                        YEL "THIS INDICATES EITHER AN ERROR IN THE CODE, OR THAT THE TPC EPSILON (%.3E) MAY BE SMALLER THAN IT SHOULD BE (THE MUC EPSILON IS (%.3E)."
                                        YEL "THIS ERROR CAN BE DISCARDED IF THIS NUMERICAL ERROR IS NOT OF RELEVANCE (COMMENT THIS WARNINH OUT)"
                                        YEL "PLEASE PLACE BREAKPOINT HERE AND DEBUG FURTHER IF DESIRED, OR REMOVE ERROR MESSAGE! EXITING FOR NOW! GOODBYE!"
                                        NC "\n", m, j, k, COALIGN_TP_EPS, COALIGN_MU_EPS);
                                    sleep(2);
                                    exit(1);
                                    // If the above error is commented, uncomment the line below
                                    //Horthog_flag_unencoded[k] = false;
                                    // Do not really need this line of code....
                                    // It will not be summed over, due to the Horthog_flag being 1 at index j
                                    //q[j] -= q[k];
                                }
                            }
                            else
                            {
                                q[j] -= q[k];
                            }
                            //printf("Term %d with shape %d has (neg) coalignment of (%d, %d)\n", i, m, j, k);
                            F[k] = 0;
                            p[j] += p[k];
                            c_map[k] = unique_count;
                            cs_map[k] = -1;
                        }
                    }
                }
                unique_count += 1;
            }
        }
        if(F[m-1])
            c_map[m-1] = unique_count;
        
        int new_shape = 0;
        for(int i = 0; i < m; i++)
            new_shape += F[i];

        // Only if we have coalignments do we need to move memory
        if( new_shape != m)
        {
            unique_count = 1;
            for(int j = 1; j < m; j++)
            {
                if(F[j])
                {
                    if(unique_count < j)
                    {
                        memcpy(A + unique_count*d, A + j*d, d*sizeof(double));
                        p[unique_count] = p[j];
                        q[unique_count] = q[j];
                    }
                    unique_count++;
                }
            }

            if(Horthog_flag)
            {
                uint new_Horthog_flag = 0;
                unique_count = 0;
                for(int j = 0; j < m; j++)
                    if(F[j])
                        new_Horthog_flag |= (Horthog_flag_unencoded[j] << unique_count++);
                Horthog_flag = new_Horthog_flag;
            }
            m = new_shape;
        }
        return m;
    }

    // After term reduction, this term is now a parent
    void become_parent()
    {
        phc = m;
        cells_gtable_p = cells_gtable; // maybe should clear cells_gtable to zero...
        gtable_p = gtable;
        gtable = NULL;
        is_new_child = false;
    }

    // Helper function to set all pointers to NULL
    void become_null()
    {
        A = NULL;
        p = NULL;
        q = NULL;
        b = NULL;
        gtable_p = NULL;
        enc_B = NULL;
        gtable = NULL;
        c_map = NULL;
        cs_map = NULL;
    }
};

void setup_first_term(ChildTermWorkSpace* workspace, CauchyTerm* first_term, double* A0, double* p0, double* b0, const int d)
{
    memcpy(workspace->A, A0, d*d*sizeof(double));
    memcpy(workspace->p, p0, d*sizeof(double));
    memcpy(workspace->b, b0, d*sizeof(double));
    first_term->A = workspace->A;
    first_term->p = workspace->p;
    first_term->q = workspace->q;
    first_term->b = workspace->b;
    first_term->c_map = NULL;
    first_term->cs_map = NULL;
    first_term->m = d;
    first_term->d = d;
}

void transfer_term_to_workspace(ChildTermWorkSpace* workspace, CauchyTerm* term)
{
    memcpy(workspace->A, term->A, term->m * term->d * sizeof(double));
    memcpy(workspace->p, term->p, term->m * sizeof(double));
    memcpy(workspace->b, term->b, term->d * sizeof(double));
    term->A = workspace->A;
    term->p = workspace->p;
    term->q = workspace->q;
    term->b = workspace->b;
    term->c_map = NULL;
    term->cs_map = NULL;
}

#endif //_CAUCHY_TERM_HPP_
