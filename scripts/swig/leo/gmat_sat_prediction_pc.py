import cauchy_estimator as ce 
import numpy as np 


def _cubic_poly_root_find(dfd_dt, d2fd_dt2):
    # Form the smooth cubic polynomial of the derivative of the cost function on interval i_star-1 to i_star -- eqns 20-23
    a0 = dfd_dt[0]
    a1 = d2fd_dt2[0]
    a2,a3 = np.linalg.inv([[1,1],[2,3]]) @ np.array([dfd_dt[1] - a0 - a1, d2fd_dt2[1] - a1])
    # Check for real positive roots between (0,1) for this polynomial
    eps = 1e-14
    P1_roots = np.roots([a3,a2,a1,a0])
    good_root = None 
    for pr in P1_roots:
        real_pr = np.real(pr)
        imag_pr = np.abs(np.imag(pr))
        if (imag_pr < eps) and (real_pr >= 0) and (real_pr <= 1) :
            if good_root is not None:
                print("ERROR PLZ DEBUG -- GOOD ROOT WAS SET TWICE!")
                exit(1)
            good_root = pr 
    # If good root was set, check to see if d P(good_root)/dt > 0
    if good_root is not None:
        gr = good_root
        if (a1 + 2*a2*gr + 3*a3*gr**2) < 0:
            good_root = None
    return good_root

def _quintic_poly_fit_and_eval(troot, r, drdt, d2rdt2):
    # fit poly on closest approach interval and for each xyz axis -- eqns 25-29
    a0 = r[0]
    a1 = drdt[0]
    a2 = 0.5 * d2rdt2[0]
    b = np.array([r[1] - a0 - a1 - a2, drdt[1] - a1 - 2*a2, d2rdt2[1] - 2*a2 ])
    a3,a4,a5 = np.linalg.inv([[1,1,1],[3,4,5],[6,12,20]]) @ b
    poly_coefs = np.array([a0,a1,a2,a3,a4,a5])
    cord = 0
    for i in range(len(poly_coefs)):
        cord += troot**i * poly_coefs[i]
    return cord


def closest_approach_info(tks, prim_tup, sec_tup):
    # Unpack the projected position, velocity and acceleration of the primary satellite
    p_pks, p_vks, p_aks = prim_tup 
    # Unpack the projected position, velocity and acceleration of the secondary satellite
    s_pks, s_vks, s_aks = sec_tup
    # Check sizings
    assert(p_pks.shape == s_pks.shape)
    assert(p_vks.shape == s_vks.shape)
    assert(p_aks.shape == s_aks.shape)
    assert(s_pks.shape == p_vks.shape == p_aks.shape)
    assert(p_pks.shape[0] >  p_pks.shape[1])
    assert(p_vks.shape[0] >  p_vks.shape[1])
    assert(p_aks.shape[0] >  p_aks.shape[1])
    # Find the index of closest position between the two vehicles
    i_star = np.argmin( np.sum( (s_pks - p_pks)**2, axis=1) ) # eq 13
    idxs = [i_star-1,i_star,i_star+1]
    # Form relative differences and the derivatives on the idxs interval -- eqns 14-16
    rd = s_pks[idxs,:] - p_pks[idxs,:]
    drd_dt = s_vks[idxs,:] - p_vks[idxs,:]
    d2rd_dt2 = s_aks[idxs,:] - p_aks[idxs,:]
    # Form cost function and derivatives of the cost function of the differences -- eqns 17-19
    fd = np.sum(rd**2, axis=1)
    dfd_dt = 2*np.sum(drd_dt * rd, axis=1)    
    d2fd_dt2 = 2*np.sum(d2rd_dt2 * rd + drd_dt**2, axis=1)
    # Check for real positive roots between (0,1) on first interval -- eqns 20-23
    second_interval = False
    troot = _cubic_poly_root_find(dfd_dt[0:2], d2fd_dt2[0:2])
    # Repeat check for real positive roots between (0,1) on second interval -- eqns 20-23
    if troot is None:
        troot = _cubic_poly_root_find(dfd_dt[1:3], d2fd_dt2[1:3])
        # Error if nothing is found still
        if troot is None:
            print("ERROR PLZ DEBUG -- NO ROOT FOUND!")
            exit(1)
        else:
            # This is the closest apprach time
            t_c = tks[idxs[1]] + (tks[idxs[2]]-tks[idxs[1]]) * troot #-- eqn 24
            second_interval = True
    else:
        # This is the closest apprach time
        t_c = tks[idxs[0]] + (tks[idxs[1]]-tks[idxs[0]]) * troot #-- eqn 24
    
    # Now, find the point for the primary satellite at closest approach -- eqns 25-29
    i = idxs[0] + second_interval
    pp_c = np.zeros(3)
    for j in range(3):
        pp_c[j] = _quintic_poly_fit_and_eval(troot, p_pks[i:i+2,j], p_vks[i:i+2,j], p_aks[i:i+2,j])
    # Now, find the point for the secondary satellite at closest approach -- eqns 25-29
    sp_c = np.zeros(3)
    for j in range(3):
        sp_c[j] = _quintic_poly_fit_and_eval(troot, s_pks[i:i+2,j], s_vks[i:i+2,j], s_aks[i:i+2,j])

    # Return 
    # 1.) closest approach start index "i"
    # 2.) closest approach time "t_c"
    # 3.) position of primary at closest approach "pp_c"
    # 4.) position of secondary at closest approach "sp_c"
    # 5.) relative position secondary - primary (redundant...dont know if necessary)
    return i, t_c, pp_c, sp_c, sp_c - pp_c 




