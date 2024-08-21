import numpy as np 
import math 
import cauchy_estimator as ce
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
import grads

# Examples of LQR, Iterative LQR for trajectory following 

# runge kutta integrator
def runge_kutta4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt*k1/2.0, u)
    k3 = f(x + dt*k2/2.0, u)
    k4 = f(x + dt*k3, u)
    x_new = x + 1.0 / 6.0 * (k1 + 2*k2 + 2*k3 + k4) * dt 
    return x_new

# Set convert_Jac_to_meters to True if the Jacobian comes in w.r.t km and you wanna convert it to meters before the Power Series
def get_STM(Jac, dt, taylor_order):
    n = Jac.shape[0]
    Phi = np.eye(n) + Jac * dt
    for i in range(2, taylor_order+1):
        Phi += np.linalg.matrix_power(Jac, i) * dt**i / math.factorial(i)
    return Phi

def get_STM_Integral(Jac, dt, taylor_order):
    n = Jac.shape[0]
    IPhi = np.eye(n) * dt + Jac * dt**2 / 2 
    for i in range(2,taylor_order):
        IPhi += np.linalg.matrix_power(Jac, i) * dt**(i+1) / math.factorial(i+1)
    return IPhi

def get_Gam(JacA, JacB, dt, taylor_order):
    IPhi = get_STM_Integral(JacA, dt, taylor_order)
    return IPhi @ JacB

def get_Jac_x(f,x,u):
    # numerical gradient 
    n = x.size
    m = f(x, u).size
    r = u.size
    ep = 2e-5
    Jac_x = np.zeros((m,n))
    zr = np.zeros(n)
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        Jac_x[:,i] = (-1.0 * f(x + 2.0*ep*ei, u) + 8.0*f(x + ep*ei, u) - 8.0 * f(x - ep*ei, u) + f(x - 2.0*ep*ei, u) ) / (12.0*ep) 
    return Jac_x

# returns Central Difference Gradient of vector f, the matrix Jacobian, 4th Order expansion
def get_Jac_u(f,x,u):
    # numerical gradient 
    m = f(x, u).size
    r = u.size
    ep = 2e-5
    Jac_u = np.zeros((m,r))
    zr = np.zeros(r)
    for i in range(r):
        ei = zr.copy()
        ei[i] = 1.0
        Jac_u[:,i] = (-1.0 * f(x, u + 2.0*ep*ei) + 8.0*f(x, u + ep*ei) - 8.0 * f(x, u - ep*ei) + f(x, u - 2.0*ep*ei) ) / (12.0*ep) 
    return Jac_u

# returns Central Difference Gradient of vector f, the matrix Jacobian, 4th Order expansion
def get_Jacs(f,x,u):
    return get_Jac_x(f,x,u), get_Jac_u(f,x,u)

def lqr_dp_step(A, B, Q, R, Sk1):
    K_k = np.linalg.solve(B.T @ Sk1 @ B + R, B.T @ Sk1 @ A)
    S_last = Q + A.T @ Sk1 @ A - A.T @ Sk1 @ B @ np.linalg.inv(R + B.T @ Sk1 @ B) @ B.T @ Sk1 @ A
    return K_k, S_last 

def lqr_gains(A, B, Q, R, Qf, steps = 50):
    Ks_rev = [None]
    Ss_rev = [Qf.copy()]
    S = Qf.copy()
    for k in reversed(range(steps)):
        K, S = lqr_dp_step(A, B, Q, R, S)
        Ks_rev.append(K)
        Ss_rev.append(S)
    return list(reversed(Ks_rev)), list(reversed(Ss_rev))

def lqr_ltv_gains(As, Bs, Q, R, Qf, steps):
    Ks_rev = [None]
    Ss_rev = [Qf.copy()]
    S = Qf.copy()
    for k in reversed(range(steps)):
        K, S = lqr_dp_step(As[k], Bs[k], Q, R, S)
        Ks_rev.append(K)
        Ss_rev.append(S)
    return list(reversed(Ks_rev)), list(reversed(Ss_rev))

# uses perturbed system about x,u bars 
def lqr_ltv_rt1(xs, x0, x_bars, u_bars, ctime_ack_model, ctime_ack_model_JacX, ctime_ack_model_JacU, Q, Qf, R, dt):
    
    ref_n = x_bars.shape[1]
    if x_bars.shape[1] < x0.size:
        x_bars = np.hstack((x_bars, x0[x_bars.shape[1]:]*np.ones((x_bars.shape[0], x0.size - x_bars.shape[1]))))

    taylor_order = 4
    Phis = [] 
    Gams = []
    # Form Phis and Gams 
    for j in range(15):
        N = x_bars.shape[0]
        for i in range(N-1):
            xbar = x_bars[i] 
            ubar = u_bars[i]
            A1 = ctime_ack_model_JacX(x_bars[i], u_bars[i])
            B1 = ctime_ack_model_JacU(x_bars[i], u_bars[i])
            A2 = ctime_ack_model_JacX(x_bars[i+1], u_bars[i])#+1] if i < N-2 else u_bars[i] )
            B2 = ctime_ack_model_JacU(x_bars[i+1], u_bars[i])#+1] if i < N-2 else u_bars[i] )
            JacA = (A1 + A2) / 2.0 # average the jacobian
            JacB = (B1 + B2) / 2.0 # average the jacobian
            Phi = get_STM(JacA, dt, taylor_order)
            Gam = get_Gam(JacA, JacB, dt, taylor_order)
            Phis.append(Phi)
            Gams.append(Gam)
        Ks, Ss = lqr_ltv_gains(Phis, Gams, Q, R, Qf, N-1)
        
        xk = x0.copy()
        x_news = [xk.copy()]
        u_news = []
        for i in range(N-1): 
            K = Ks[i]
            dx = xk - x_bars[i]
            du = - K @ dx
            u_new = du + u_bars[i]
            xk = runge_kutta4(ctime_ack_model, xk, u_new, dt)
            x_news.append(xk.copy())
            u_news.append(u_new.copy())
        x_news = np.array(x_news)
        u_news = np.array(u_news)
        u_bars = u_news
        if ref_n  < x0.size:
            x_bars[ref_n:,:] = x_news[ref_n:,:]

        Ts = np.arange(N) * dt
        plt.plot(xs[:, 0], xs[:, 1], 'b')
        plt.plot(x_news[:, 0], x_news[:, 1], 'g')
        plt.show()
        plt.figure()
        plt.plot(Ts[:-1], u_news[:,0], 'g')
        plt.plot(Ts[:-1], u_news[:,1], 'b')
        plt.show()
        plt.close()
        foobar=2

# uses perturbed system about x,u bars 
def lqr_ltv_rt2(xs, x0, x_bars, u_bars, ctime_ack_model, ctime_ack_model_JacX, ctime_ack_model_JacU, Q, Qf, R, dt):
    
    ref_n = x_bars.shape[1]
    if x_bars.shape[1] < x0.size:
        x_bars = np.hstack((x_bars, x0[x_bars.shape[1]:]*np.ones((x_bars.shape[0], x0.size - x_bars.shape[1]))))
    n = x0.size
    m = u_bars.shape[1]
    taylor_order = 4
    Phis = [] 
    Gams = []
    _Q = np.zeros((n+1,n+1))
    _Qf = np.zeros((n+1,n+1))
    _Q[0:n,0:n] = Q.copy() 
    _Qf[0:n,0:n] = Qf.copy() 
    # Form Phis and Gams 
    for j in range(15):
        N = x_bars.shape[0]
        for i in range(N-1):
            xbar = x_bars[i] 
            ubar = u_bars[i]
            A1 = ctime_ack_model_JacX(x_bars[i], u_bars[i])
            B1 = ctime_ack_model_JacU(x_bars[i], u_bars[i])
            A2 = ctime_ack_model_JacX(x_bars[i+1], u_bars[i])
            B2 = ctime_ack_model_JacU(x_bars[i+1], u_bars[i])
            JacA = (A1 + A2) / 2.0 # average the jacobian
            JacB = (B1 + B2) / 2.0 # average the jacobian
            c = -(ctime_ack_model(x_bars[i], u_bars[i]) + ctime_ack_model(x_bars[i+1], u_bars[i]))/2 * (j > 3)
            Phi = get_STM(-JacA, dt, taylor_order)
            Gam = get_Gam(-JacA, -JacB, dt, taylor_order)
            ck = get_STM_Integral(-JacA, dt,taylor_order) @ c
            Phi_k = np.zeros((n+1+m,n+1+m))
            Phi_k[0:n,0:n] = Phi.copy()
            Phi_k[0:n,n] = ck.copy()
            Phi_k[n,n] = 1
            Phi_k[n+1:,n+1:] = np.eye(m)
            Gam_k = np.zeros((n+1+m,m))
            Gam_k[0:n,:] = Gam.copy()
            Gam_k[n+1:,:] = np.eye(m)
            Phis.append(Phi_k)
            Gams.append(Gam_k)
        Ks, Ss = lqr_ltv_gains(Phis, Gams, _Q, R, _Qf, N-1)
        
        xk = x0.copy()
        x_news = [xk.copy()]
        u_news = []
        for i in range(N-1): 
            K = Ks[i]
            dx = np.ones(n+1)
            dx[0:n] = x_bars[i] - xk
            du = - K @ dx
            #du[0] = np.clip(du[0], -0.1, 0.1)
            #du[1] = np.clip(du[1], -0.04, 0.04)
            u_new = du + u_bars[i]
            #u_new[1] = np.clip(u_new[1], -np.pi/6, np.pi/6)
            xk = runge_kutta4(ctime_ack_model, xk, u_new, dt)
            x_news.append(xk.copy())
            u_news.append(u_new.copy())
        x_news = np.array(x_news)
        u_news = np.array(u_news)
        u_bars = u_news
        if ref_n  < x0.size:
            x_bars[ref_n:,:] = x_news[ref_n:,:]

        Ts = np.arange(N) * dt
        plt.plot(xs[:, 0], xs[:, 1], 'b')
        plt.plot(x_news[:, 0], x_news[:, 1], 'g')
        plt.show()
        plt.figure()
        plt.plot(Ts[:-1], u_news[:,0], 'g')
        plt.plot(Ts[:-1], u_news[:,1], 'b')
        plt.show()
        plt.close()
        foobar=2


def mpc_make_H(Q,Qf,R,S,N):
    m = R.shape[0]
    n= Q.shape[0]
    HB = np.zeros((n+m,n+m))
    HB[0:n,0:n] = Q
    HB[0:n,n:n+m] = S
    HB[n:n+m,0:n] = S.T
    HB[n:,n:] = R 
    h_size = N * (n+m)
    H = np.zeros((h_size,h_size))
    H[0:m,0:m] = R 
    count = m 
    while count < h_size-n:
        H[count:count+n+m,count:count+n+m] = HB   
        count += n+m
    H[-n:,-n:] = Qf
    return H

def mpc_make_C(As,Bs,N):
    n = As[0].shape[0]
    m = Bs[0].shape[1]

    size_Cx = N*(n+m)
    size_Cy = N*n
    C = np.zeros((size_Cy,size_Cx))
    CB = np.zeros((n,2*n+m))
    CB[0:n,0:n] = -As[0] # not used first iteration
    CB[0:n,n:n+m] = -Bs[0]
    CB[0:n,n+m:n+m+n] = np.eye(n)
    C[0:n,0:n+m] = CB[:,n:]
    count_x = m
    count_y = n
    count = 1
    while count_x < size_Cx-n:
        CB = np.zeros((n,2*n+m))
        CB[0:n,0:n] = -As[count] # not used first iteration
        CB[0:n,n:n+m] = -Bs[count]
        CB[0:n,n+m:n+m+n] = np.eye(n)
        C[count_y:count_y+n, count_x:count_x+2*n+m] = CB
        count_x += n+m 
        count_y += n
        count += 1
    return C 

def mpc_make_b(A0, x0, cs, N):
    n = A0.shape[0]
    b_size = N*n 
    b = np.zeros(b_size)
    b[0:n] = A0 @ x0 + cs[0]
    count = n 
    i = 1
    while count < b_size:
        b[count:count+n] = cs[i]
        i += 1
        count += n 
    return b

def mpc_make_h(f,ff,Fx,x0,N):
    n = x0.size
    p = Fx.shape[0]
    m = int(p-2*n)//2
    size_h = (N+1)*p - 2*m
    h = np.zeros(size_h)
    h[0:p] = f - Fx @ x0 
    count = p
    while count < size_h-p:
        h[count:count+p] = f
        count += p
    h[-2*n:] = ff[:2*n]
    return h

def mpc_make_P(Fx,Fu,Ff, N):
    n = Fx.shape[1]
    m = Fu.shape[1]
    F = np.hstack((Fx,Fu))
    p = F.shape[0]
    size_Px = N*(n+m)
    size_Py = (N+1)*p - 2*m
    P = np.zeros((size_Py,size_Px))
    P[0:p,0:m] = Fu
    count_x = m 
    count_y = p
    while count_y < size_Py-p:
        P[count_y:count_y+p,count_x:count_x+m+n] = F 
        count_x += n+m
        count_y += p
    P[-2*n:,-n:] = Ff[0:2*n,:]
    return P

def mpc_make_d(h, P, x):
    return 1.0 / (h - P @ x)

def mpc_make_rd(rt, x, nu, H, t, P, d, C, S, x0):
    n = x0.size
    m = S.shape[1]
    rd = 2*H @ (x-rt) + 1/t * P.T @ d + C.T @ nu 
    rd[0:m] += 2.0 * S.T @ x0
    return rd 

def mpc_make_rp(C,z,b):
    return C @ z - b

def mpc_make_init_vars(x0, u0, ref_traj, m):
    n = x0.size
    r = ref_traj.shape[1]
    N = ref_traj.shape[0]
    x_size = (m+n)*(N-1)
    x = np.zeros(x_size)
    x[0:m] = u0
    rt = np.zeros(x_size)
    count = m
    i = 1
    while count < x_size-n-m:
        x[count:count+r] = ref_traj[i] 
        x[count+r:count+n] = 0.02 * np.random.randn(n-r)
        x[count+n:count+n+m] = u0
        rt[count:count+r] = ref_traj[i]
        count += n+m
        i += 1
    x[count:count+r] = ref_traj[i]
    x[count+r:count+n] = 0.02 * np.random.randn(n-r)
    rt[count:count+r] = ref_traj[-1]
    nu = np.random.randn(n*(N-1))
    return x, nu, rt

def mpc_make_states_controls(z, x0, n, m):
    size_z = z.size 
    N = int( size_z / (n+m) )
    xs = np.zeros((N+1, n))
    us = np.zeros((N, m))
    xs[0] = x0 
    count = 0
    i = 1
    while count < size_z:
        us[i-1] = z[count:count+m]
        xs[i] = z[count+m:count+m+n]
        count += n+m
        i += 1
    return xs, us 

def mpc_make_opt_vector_from_state_control_arrays(xs,us):
    N = xs.shape[0]
    n = xs.shape[1]
    m = us.shape[1]
    z_size = (N-1)*(n+m)
    z = np.zeros(z_size)
    z[0:m] = us[0]
    count = m 
    i = 1
    while count < z_size - m - n: 
        z[count:count+n] = xs[i]
        z[count+n:count+n+m] = us[i]
        count += m+n 
        i += 1
    z[count:count+n] = xs[N-1]
    return z

def mpc_make_box_constaints(n,m, constr_tup):
    p = 2*n + 2*m 
    Fx = np.zeros((p,n))
    Fu = np.zeros((p,m))
    for i in range(n):
        Fx[2*i,i] = 1
        Fx[2*i+1,i] = -1
    for i in range(m):
        Fu[2*n + 2*i,i] = 1
        Fu[2*n + 2*i+1,i] = -1
    f = np.zeros(p)
    for i,v in enumerate(constr_tup):
        f[i] = (-1)**(i%2) * v
    return Fx, Fu, f

def mpc_make_nonlin_c2dtime(jac_dyn_fx, jac_dyn_fu, xs, us, dt):
    N = xs.shape[0]
    n = xs[0].size
    m = us[0].size
    Jac_A1 = jac_dyn_fx(xs[0], us[0])
    Jac_B1 = jac_dyn_fu(xs[0], us[0])
    Phis = np.zeros((N-1,n,n))
    Gams = np.zeros((N-1,n,m))
    for i in range(N-1):
        Jac_A2 = jac_dyn_fx(xs[i+1], us[i])
        Jac_B2 = jac_dyn_fu(xs[i+1], us[i])
        Jac_A = (Jac_A2 + Jac_A1) / 2.0
        Jac_B = (Jac_B1 + Jac_B2) / 2.0
        Phi = get_STM(Jac_A, dt, taylor_order=4)
        Gam = get_Gam(Jac_A, Jac_B, dt, taylor_order=4)
        Phis[i] = Phi
        Gams[i] = Gam 
        Jac_A1 = Jac_A2
        Jac_B1 = Jac_B2
    return Phis, Gams

def linear_fast_mpc(z,nu,rt,x0,R,Q,Qf,S,As,Bs,cs,Fx,Fu,Ff,f,ff):
    n = x0.size 
    m = R.shape[0]
    N = len(As)
    t = 1.0
    # Make Constant Matrices 
    # Make H
    H = mpc_make_H(Q,Qf,R,S,N)
    # Make P 
    P = mpc_make_P(Fx,Fu,Ff,N)
    # Make C 
    C = mpc_make_C(As,Bs,N)
    # Make h 
    h = mpc_make_h(f, ff, Fx, x0, N)
    # Make b 
    b = mpc_make_b(As[0], x0, cs, N)
    # Iterate on residual 
    iterate_true = True
    iterate_count = 0
    s = 0.99 
    while iterate_true:
        # Make d 
        d = mpc_make_d(h, P, z)
        # Make rd 
        rd = mpc_make_rd(rt, z, nu, H, t, P, d, C, S, x0)
        # Make rp 
        rp = mpc_make_rp(C,z,b)
        # Make d
        z_size = N*(n+m)
        nu_size = N*n
        KKT = np.zeros((z_size+nu_size,z_size+nu_size))
        KKT[0:z_size,0:z_size] = H + 1/t * P.T @ np.diag(d**2) @ P
        KKT[0:z_size,z_size:] = C.T
        KKT[z_size:, 0:z_size] = C
        r = np.concatenate((rd,rp))
        delta = - np.linalg.solve(KKT, r)
        dz = delta[0:z_size]
        dnu = delta[z_size:]

        # Backtracking 
        MAX_ITS_LS = 50
        eps = 5e-5
        mu = 5
        delta = 0.1
        alpha = 1.0
        beta = 0.7
        #s = 0.99 * alpha

        # Must keep new point strictly feasible
        effort_counts = 0
        while np.any( mpc_make_d(h,P,z+s*dz) < 0 ):
            s *= beta
            effort_counts += 1
            if effort_counts > MAX_ITS_LS:
                print("Feasibility Check! Cannot Find Feasible New Point!")
                return z, nu
            
        # we then continue to multiply s by beta until we have norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2
        z_new = z + s * dz
        nu_new = nu + s * dnu
        # Make d 
        d = mpc_make_d(h, P, z_new)
        rd_new = mpc_make_rd(rt, z_new, nu_new, H, t, P, d, C, S, x0)
        rp_new = mpc_make_rp(C, z_new, b)
        r_new = np.concatenate((rd_new,rp_new))
        r_old = r 
        is_rnew_smaller = np.linalg.norm(r_new) <=  (1.0 - delta * s) * np.linalg.norm(r_old)
        effort_counts = 0
        while( not is_rnew_smaller):
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2...exiting!")
                return z, nu
            effort_counts += 1
            s *= beta 
            z_new = z + s * dz
            nu_new = nu + s * dnu
            # Make d 
            d = mpc_make_d(h, P, z_new)
            rd_new = mpc_make_rd(rt, z_new, nu_new, H, t, P, d, C, S, x0)
            rp_new = mpc_make_rp(C, z_new, b)
            r_new = np.concatenate((rd_new,rp_new))
            is_rnew_smaller = (np.linalg.norm(r_new) <=  (1.0 - delta * s) * np.linalg.norm(r_old))
        # Check residual magnitude
        resid_mag = np.linalg.norm(r_new)
        resid_mag_old = np.linalg.norm(r)
        print("Step {}, Resid Magnitude Old v New: {} vs. {}, backtrack step of s*dz where s={}, barrier t={}".format(iterate_count+1, np.round(resid_mag_old,5), np.round(resid_mag,5),s,t))
        iterate_true = resid_mag > eps
        # Increase barrier parameter, update opt variables
        t *= mu
        z = z_new.copy() 
        nu = nu_new.copy()
        # Check if barrier param is too large 
        iterate_true *= (1/t > eps**2)
        # Increase iteration count
        iterate_count += 1
    return z, nu

def nonlinear_fast_mpc(x0, u0, dt, R, Q, Qf, S, dyn_f, jac_dyn_fx, jac_dyn_fu, Fx, Fu, Ff, f, ff, ref_traj):
    n = Q.size
    m = R.shape[0]
    r = ref_traj.shape[1]
    N = ref_traj.shape[0]
    if( u0.size != m ):
        N = ref_traj.shape[0]-1
        if u0.size == (N*(n+m) + N*n):
            _, _, rt = mpc_make_init_vars(x0, np.zeros((N-1,m)), ref_traj, m)
            z = np.concatenate(( u0[n+m:N-n], ref_traj[-1], np.zeros(n-r) ))
    # create initial variables and 
    z, nu, rt = mpc_make_init_vars(x0, u0, ref_traj, m)
    # get state and control format 
    xs, us = mpc_make_states_controls(z, x0, n, m)
    As, Bs = mpc_make_nonlin_c2dtime(jac_dyn_fx, jac_dyn_fu, xs, us, dt)
    cs = np.array([ runge_kutta4(dyn_f,x,u,dt) - A @ x - B @ u for x,u,A,B in zip(xs,us,As,Bs)])
    # while the solution difference dz = z_new-z_old where z = [xs,us] (old/new) is less than eps 
    z_last = z
    dz_norm = 1
    dz_norm_eps = 1e-3 
    with_plot = True
    count = 0
    while dz_norm > dz_norm_eps:
        z, nu = linear_fast_mpc(z,nu,rt,x0,R,Q,Qf,S,As,Bs,cs,Fx,Fu,Ff,f,ff)
        xs, us = mpc_make_states_controls(z, x0, n, m)
        rks = xs.copy()
        rks[:,:r] = ref_traj
        As, Bs = mpc_make_nonlin_c2dtime(jac_dyn_fx, jac_dyn_fu, rks, us, dt)
        cs = np.array([runge_kutta4(dyn_f,x,u,dt) - A @ x - B @ u for x,u,A,B in zip(rks,us,As,Bs)])
        dz_norm = np.linalg.norm(z_last-z)
        z_last = z.copy()
        z = mpc_make_opt_vector_from_state_control_arrays(rks,us)
        count += 1
        print("Step {}: DZ Norm: {} -> to go to {}".format(count, dz_norm, dz_norm_eps))
        if with_plot:
            xs_prop = [x0.copy()]
            x = x0.copy()
            for u in us:
                x = runge_kutta4(dyn_f, x, u, dt)
                xs_prop.append(x)
            xs_prop = np.array(xs_prop)
            Ts = np.arange(N) * dt 
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)
            ax1.title.set_text("State Traj")
            ax2.title.set_text("Theta Angle")
            ax3.title.set_text("Vel/Phi Deriv Cmd (b/g)")
            ax4.title.set_text("Vel/Phi (b/g)")
            plt.subplot(2,2,1)
            #plt.suptitle("State Traj")
            plt.plot(xs[:,0],xs[:,1], 'b')
            plt.plot(ref_traj[:,0],ref_traj[:,1], 'g')
            plt.plot(xs_prop[:,0],xs_prop[:,1], 'm')
            plt.subplot(2,2,2)
            #plt.suptitle("Theta Angle")
            plt.plot(Ts,xs[:,2], 'b')
            plt.plot(Ts,xs_prop[:,2], 'm')
            if r > 2:
                plt.plot(Ts,ref_traj[:,2], 'g')
            plt.subplot(2,2,3)
            #plt.suptitle("Vel/Phi Deriv Cmd (b/g)")
            plt.plot(Ts[:-1],us[:,0], 'b')
            plt.plot(Ts[:-1],us[:,1], 'g')
            plt.subplot(2,2,4)
            if n == 5:
                #plt.suptitle("Vel/Phi (b/g)")
                plt.plot(Ts,xs[:,3], 'b')
                plt.plot(Ts,xs[:,4], 'g')
            if n == 8:
                plt.plot(Ts,xs[:,6], 'b')
                plt.plot(Ts,xs[:,7], 'g')

            plt.show()
            if dz_norm > dz_norm_eps:
                plt.pause(4)
                plt.close()
    foobar = 2

def test_fast_mpc_linear():
    x0 = np.array([1.0,1.0])
    dt = 0.10
    steps = 40
    u0 = np.zeros(1)
    R = 0.1*np.eye(1)
    Q = 0.25*np.eye(2)
    Qf = 10*np.eye(2)
    S = np.zeros((2,1))
    _A = np.array([[0,1],[0,0]])
    _B = np.array([[0,1]]).T 
    A = get_STM(_A, dt, 4)
    B = get_Gam(_A, _B, dt, 4)
    As = np.array([A for _ in range(steps-1)])
    Bs = np.array([B for _ in range(steps-1)])
    cs = np.array([np.zeros(2) for _ in range(steps-1)])
    Fx = np.array([[1,0],[-1,0],[0,1],[0,-1],[0,0],[0,0]])
    Fu = np.array([[0],[0],[0],[0],[1],[-1]])
    x1_high, x1_low, x2_high, x2_low, u1_high, u1_low = (10,-10,  10,-10,  5, -5)
    f = np.array([x1_high,-x1_low,x2_high,-x2_low,u1_high,-u1_low])
    Ff = Fx 
    ff = f 
    ref_traj = np.zeros((steps,2))
    z, nu, rt = mpc_make_init_vars(x0, u0, ref_traj, 1)
    z, nu = linear_fast_mpc(z,nu,rt,x0,R,Q,Qf,S,As,Bs,cs,Fx,Fu,Ff,f,ff)
    xs, us = mpc_make_states_controls(z, x0, 2, 1)
    plt.figure()
    plt.plot(xs[:,0], xs[:,1], 'b')
    plt.figure()
    Ts = np.arange(steps)*dt
    plt.plot(Ts[:-1], us.reshape(-1))
    plt.show()
    foobar = 2

def test_fast_mpc_nonlinear():
    x0 = np.array([0.0,0.0,0.0])
    L = 0.14 # meters
    dt = 1.0 / 15.0
    R = np.diag(np.array([0.01, 0.01]))
    Q = 1.0*np.eye(3)
    Q[2,2] = 0.5
    Qf = 8*np.eye(3)
    Qf[2,2] = 2.0
    S = np.zeros((3,2))
    r_states = 3
    if r_states == 2:
        Q[2,2] = 0
        Qf[2,2] = 0

    Fx = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
    Fu = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    constr_tup = (100,-100,  100,-100, 30, -30, 3, -3, np.pi/5.5, -np.pi/5.5)
    x1_high, x1_low, x2_high, x2_low, x3_high, x3_low, u1_high, u1_low, u2_high, u2_low = constr_tup
    f = np.array([x1_high,-x1_low,x2_high,-x2_low,x3_high, -x3_low,u1_high,-u1_low,u2_high, -u2_low])
    Ff = Fx 
    ff = f 

    steps = int( 1 / dt * 16 )
    ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])

    # Create a true control policy and state trajectory
    #v = np.linspace(0.5, 1.25, steps)
    #phi = np.pi/8*np.sin([t for t in np.linspace(0,2*np.pi, steps)])
    v = 0.25 + 1.0*np.cos([t for t in np.linspace(0,2*np.pi, steps)]) #np.linspace(0.5, 1.25, steps)
    phi = np.pi/6*np.sin([t for t in np.linspace(0,4*np.pi, steps)])    
    us = np.vstack((v,phi)).T 
    x0 = np.zeros(3)
    xs = np.zeros((steps+1, 3))
    xs[0] = x0.copy()
    xk = x0.copy()
    count = 1
    for u in us: 
        xk = runge_kutta4(ctime_ack_model, xk, u, dt)
        xs[count] = xk.copy()
        count += 1
    #plt.plot(xs[:,0], xs[:,1], 'b')
    #plt.show()
    # Use xs as a trajectory
    ref_traj = xs[:,0:r_states]
    u0 = us[0] + np.random.randn(2)*0.01
    # call nonlinear fast mpc 
    nonlinear_fast_mpc(x0, u0, dt, R, Q, Qf, S, ctime_ack_model, ctime_ack_model_JacX, ctime_ack_model_JacU, Fx, Fu, Ff, f, ff, ref_traj)
    
def test_fast_mpc_nonlinear2():
    L = 0.14 # meters
    dt = 1.0 / 15.0
    R = np.diag(np.array([0.15, 0.15]))
    Q = np.eye(5)
    Qf = 5*np.eye(5)
    Q[2,2] = 0.5
    Qf[2,2] = 2
    Q[3:,3:] *= 0
    Qf[3:,3:] *= 0
    S = np.zeros((5,2))
    r_states = 3
    if r_states == 2:
        Q[2,2] = 0
        Qf[2,2] = 0

    state_control_box_constraints = (100,-100,  100,-100, 30, -30, 3, -3, np.pi/5.5, -np.pi/5.5, 2.5,-2.5, 2.5, -2.5)
    Fx,Fu,f = mpc_make_box_constaints(5,2, state_control_box_constraints)
    Ff = Fx 
    ff = f 

    steps = int( 1 / dt * 16 )
    ctime_ack_model5 = lambda x, u : np.array([x[3] * np.cos(x[2]), x[3] * np.sin(x[2]), x[3]/L * np.tan(x[4]), u[0], u[1] ])
    ctime_ack_model_JacX5 = lambda x, u : np.array([ [0, 0, -x[3] * np.sin(x[2]), np.cos(x[2]), 0], 
                                                    [0,0,x[3] * np.cos(x[2]), np.sin(x[2]), 0],
                                                    [0, 0, 0, 1/L * np.tan(x[4]), x[3]/L /np.cos(x[4])**2],
                                                    [0,0,0,0,0],
                                                    [0,0,0,0,0] ])
    ctime_ack_model_JacU5 = lambda x, u : np.array([0,0, 0,0, 0,0, 1,0,0,1]).reshape((5,2))
    ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])

    # Create a true control policy and state trajectory
    v = 0.25 + 1.0*np.cos([t for t in np.linspace(0,2*np.pi, steps)]) #np.linspace(0.5, 1.25, steps)
    phi = np.pi/6*np.sin([t for t in np.linspace(0,4*np.pi, steps)])    
    #v = 0.5 + 1.5*np.cos([t for t in np.linspace(0,2*np.pi, steps)]) #np.linspace(0.5, 1.25, steps)
    #phi = np.pi/8*np.sin([t for t in np.linspace(0,4*np.pi, steps)]) 
    us = np.vstack((v,phi)).T 
    x0 = np.zeros(3)
    xs = np.zeros((steps+1, 3))
    xs[0] = x0.copy()
    xk = x0.copy()
    count = 1
    for u in us: 
        xk = runge_kutta4(ctime_ack_model, xk, u, dt)
        xs[count] = xk.copy()
        count += 1
    plt.plot(xs[:,0], xs[:,1], 'b')
    plt.show()
    # Use xs as a trajectory
    ref_traj = xs[:,0:r_states]
    u0 = us[0] + np.random.randn(2)*0.01
    x0 = np.concatenate((x0, us[0]))
    u0 = np.zeros(2)
    # call nonlinear fast mpc 
    nonlinear_fast_mpc(x0, u0, dt, R, Q, Qf, S, ctime_ack_model5, ctime_ack_model_JacX5, ctime_ack_model_JacU5, Fx, Fu, Ff, f, ff, ref_traj)
    foobar = 2


# TODO: Explain sizes and structure of f, g, h, what the source the algorithm is (boyd txtbook?), references, examples.
# TODO: The code does look nicely organized tho.
# Numerical IPOPT Implementation
# Phase I IPOPT with finite differencing for gradients
def ipopt_phaseI(n,g,h):
    x = np.random.uniform(0,1.0, size = n) 
    m = g(x).size 
    p = h(x).size 

    nu = 100.0*np.random.uniform(5,10, size = m) 
    lam = 100.0*np.random.uniform(5,10, size = p) 
    s = np.max(g(x))

    mu = 2.0 
    beta = .7
    alpha = .1
    ep_feas = 1e-4
    MAX_ITS = 100
    MAX_ITS_LS = 50 
    num_its = 0
    eta_hat = 1.0
    while(s > -0.20 or eta_hat < ep_feas**2): 
        eta_hat = -1.0 * (g(x) - s*np.ones(m)) @ nu 
        t = mu * m / eta_hat
        r_dual = np.zeros(n+1)
        r_dual[0:n] = grads.cd4_gvf(x,g).T @ nu + grads.cd4_gvf(x,h).T @ lam 
        r_dual[n] = -1.0 + np.ones(m) @ nu 
        r_cen = -1.0 / t - np.diag(g(x) - s*np.ones(m)) @ nu 
        r_primal = h(x)
        r =  np.concatenate((r_dual,r_cen,r_primal))

        KKT = np.zeros((n+1+m+p,n+1+m+p))
        DGs = grads.cd4_dgvf(x,g)
        DHs = grads.cd4_dgvf(x,h)
        KKT[0:n,0:n] = np.sum([DGs[i]*nu[i] for i in range(m)], axis = 0) + np.sum([DHs[i]*lam[i] for i in range(p)], axis = 0)
        KKT[0:n,n+1:n+1+m] = grads.cd4_gvf(x,g).T 
        KKT[0:n,n+1+m:n+1+m+p] = grads.cd4_gvf(x,h).T 
        KKT[n,n+1:n+1+m] = np.ones(m)
        KKT[n+1:n+1+m, 0:n] = -1.0 * np.diag(nu) @ grads.cd4_gvf(x,g)
        KKT[n+1:n+1+m,n] = nu 
        KKT[n+1:n+1+m,n+1:n+1+m] = -1.0 * np.diag(g(x) - s*np.ones(m))
        KKT[n+1+m:n+1+m+p,0:n] = grads.cd4_gvf(x,h)

        delta = -1.0 * np.linalg.solve(KKT, r)

        delt_x = delta[0:n]
        delt_s = delta[n]
        delt_nu = delta[n+1:n+1+m]
        delt_lam = delta[n+1+m:n+1+m+p]

        # Line search for z
        delt_nu_feas = delt_nu[delt_nu < 0]
        nu_feas = nu[delt_nu < 0]
        if(delt_nu_feas.size == 0):
            z_max = 1.0
        else:
            z_max = np.min(-1.0*nu_feas / delt_nu_feas)
            z_max = z_max if z_max < 1.0 else 1.0
        z = 0.99 * z_max 
        x_plus = x + z*delt_x
        s_plus = s + z*delt_s
        is_ineq_invalid = np.sum(( (g(x_plus) - s_plus*np.ones(m)) > 0.0))
        effort_counts = 0
        while( is_ineq_invalid ):
            z *= beta 
            x_plus = x + z * delt_x 
            s_plus = s + z * delt_s
            is_ineq_invalid = np.sum( (g(x_plus) - s_plus*np.ones(m)) > 0.0 )
            effort_counts += 1
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! s is: ", s, ",  Cannot find a x_plus s.t. g(x_plus) < s and s < 0...exiting!")
                return s, x, nu, lam
        nu_plus = nu + z*delt_nu
        lam_plus = lam + z*delt_lam
        eta_hat_plus = -1.0 * (g(x_plus) - s_plus*np.ones(m)) @ nu_plus
        t_plus = mu * m / eta_hat_plus
        r_dual_plus = np.zeros(n+1)
        r_dual_plus[0:n] = grads.cd4_gvf(x_plus,g).T @ nu_plus + grads.cd4_gvf(x_plus,h).T @ lam_plus
        r_dual_plus[n] = -1.0 + np.ones(m) @ nu_plus
        r_cen_plus = -1.0 / t_plus - np.diag(g(x_plus) - s_plus*np.ones(m)) @ nu_plus
        r_primal_plus = h(x_plus)
        r_plus =  np.concatenate((r_dual_plus,r_cen_plus,r_primal_plus))
        is_rplus_smaller = np.linalg.norm(r_plus) <=  (1.0 - alpha * z) * np.linalg.norm(r)
        effort_counts = 0
        while( not is_rplus_smaller):
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! s is: ", s, ", Cannot find a x_plus s.t. norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2...exiting!")
                return s, x, nu, lam
            effort_counts += 1
            z *= beta 
            x_plus = x + z*delt_x
            s_plus = s + z*delt_s 
            nu_plus = nu + z*delt_nu
            lam_plus = lam + z*delt_lam
            eta_hat_plus = -1.0 * (g(x_plus) - s_plus*np.ones(m)) @ nu_plus
            if(eta_hat_plus < 0.0):
                #print("New Surrogate Duality Gap Flipped...Using old gap to stabilize!")
                eta_hat_plus = eta_hat
            t_plus = mu * m / eta_hat_plus
            r_dual_plus = np.zeros(n+1)
            r_dual_plus[0:n] = grads.cd4_gvf(x_plus,g).T @ nu_plus + grads.cd4_gvf(x_plus,h).T @ lam_plus
            r_dual_plus[n] = -1.0 + np.ones(m) @ nu_plus
            r_cen_plus = -1.0 / t_plus - np.diag(g(x_plus) - s_plus*np.ones(m)) @ nu_plus
            r_primal_plus = h(x_plus)
            r_plus =  np.concatenate((r_dual_plus,r_cen_plus,r_primal_plus))
            is_rplus_smaller = np.linalg.norm(r_plus) <=  (1.0 - alpha * z) * np.linalg.norm(r)
        x = x_plus.copy()
        s = s_plus.copy() 
        nu = nu_plus.copy()
        lam = lam_plus.copy() 
        e_vals, _ = np.linalg.eig(KKT)
        print("Duality Gap: ", np.round(eta_hat_plus,8), ", Objective: ", np.round(s,4), 
                " z: ", np.round(z,5), "KKT cond. #: ", np.round(np.linalg.cond(KKT),5), 
                "KKT min/max e-vals: ", np.round(np.min(e_vals),5), ", ", np.round(np.max(e_vals),5))
        num_its += 1
        if num_its > MAX_ITS:
            print("Reached Max Iterations!...exiting!")
            return s, x, nu, lam
    
    print("IPOPT Phase I Took ", num_its, " iterations")
    return s, x, nu, lam

# Phase II Primal-Dual Interior Point Method: Use this method if you have a viable point in the domain of f
# f is the function to optimize
# g is the inequality constraints 
# h is the equality constraints
# x is the starting guess - does not have to be feasible, if not, runs phaseI
# NOTE: All functions f,g,h must return a 1-D numpy array 
def ipopt(f, g, h, x):
    # Program constants
    mu = 100.0
    beta = .7
    ep_feas = 1e-6
    ep = 1e-6
    alpha = .1

    # Program Iteration Constants
    MAX_ITS = 100 # for ipopot
    MAX_ITS_LS = 50 # for line search used by ipopt

    # Variable Sizes
    n = x.size # number of primal variables
    m = g(x).size # number of dual inequality constrained variables
    p = h(x).size # number of dual equality constrained variables

    # test whether we know of a feasible starting point in problem domain
    is_ineq_satisfied = np.all( (g(x) <= 0) == True)
    is_eq_satisfied = np.all( (np.abs(h(x)) <= ep) == True )
    if(not is_ineq_satisfied and not is_eq_satisfied):
        print("\nRunning Phase One First, the given x is not feasible for both ineq / eq\n")
        pI_gate, x, _, _ = ipopt_phaseI(n,g,h)
        if(pI_gate > 0.0):
            print("Could not find Feasible Point during phase I...exiting!")
            #exit(1)
            return None, None, None
    elif(not is_ineq_satisfied):
        print("\nRunning Phase One First, the given x is not feasible for ineq\n")
        pI_gate, x, _, _ = ipopt_phaseI(n,g,h)
        if(pI_gate > 0.0):
            print("Could not find Feasible Point during phase I...exiting!")
            #exit(1)
            return None, None, None
    print("\nPhase II will proceed...")

    nu = 100.0 * np.random.uniform(15.0,20.0,size = m) #np.array([81.80196417, 85.83896045, 79.80644929])  
    lam = 100.0 * np.random.uniform(15.0,20.0,size = p) #np.array([98.86721197, 85.19552515, 91.72376991, 84.95309117, 92.01650833, 96.39129462]) 

    print("Init lagrange vars: ", nu, lam)

    # Start IPOPT
    ipopt_done_flag = False 
    num_its = 0
    # Find initial t
    eta_hat = -1.0 * g(x) @ nu 
    t = m * mu / eta_hat
    # Compute initial residual r
    r_dual = grads.cd4_gf(x, f) + grads.cd4_gvf(x,g).T @ nu + grads.cd4_gvf(x,h).T @ lam
    r_central = -1.0 / t * np.ones(m) - np.diag(g(x)) @ nu 
    r_primal = h(x)
    r = np.concatenate((r_dual,r_central,r_primal))
    while(not ipopt_done_flag):
        # Form KKT Matrix 
        Grad_G = grads.cd4_gvf(x,g)
        Grad_H = grads.cd4_gvf(x,h)
        KKT = np.zeros((n+m+p,n+m+p))
        DG_Gs = grads.cd4_dgvf(x,g) #double gradient tensor
        DG_G = np.sum([DG_Gs[i]*nu[i] for i in range(m)],axis = 0)
        DG_Hs = grads.cd4_dgvf(x,h) #double gradient tensor
        DG_H = np.sum([DG_Hs[i]*lam[i] for i in range(p)],axis = 0) 
        KKT[0:n,0:n] = grads.cd4_dgf(x,f) + DG_G + DG_H #+ 1e-4*np.eye(n)
        KKT[0:n,n:n+m] = Grad_G.T
        KKT[0:n,n+m:n+m+p] = Grad_H.T
        KKT[n:n+m,0:n] = -1.0 * np.diag(nu) @ Grad_G
        KKT[n:n+m,n:n+m] = -1.0 * np.diag(g(x))
        KKT[n+m:n+m+p,0:n] = Grad_H
        # Compute Newton Step 
        KKT = KKT + 1e-8*np.eye(n+m+p)
        delta = -1.0 * np.linalg.solve(KKT, r)
        # Perform Line Search to Update x, nu, lam 
        delt_x = delta[0:n]
        delt_nu = delta[n:n+m]
        delt_lam = delta[n+m:n+m+p]

        # first compute largest positive step length, not exceeding one, that gives lam_plus >= 0
        # Step 3: Line search for s
        delt_nu_feas = delt_nu[delt_nu < 0]
        nu_feas = nu[delt_nu < 0]
        if(delt_nu_feas.size == 0):
            s_max = 1.0
        else:
            s_max = np.min(-1.0*nu_feas / delt_nu_feas)
            s_max = s_max if s_max < 1.0 else 1.0
        # start backtracting at 0.99*s_max and multiply s by beta until we have g(x_plus) < 0
        s = 0.99 * s_max 
        x_plus = x + s*delt_x
        is_ineq_invalid = np.sum((g(x_plus) > 0.0))
        effort_counts = 0
        while( is_ineq_invalid ):
            s *= beta 
            x_plus = x + s * delt_x 
            is_ineq_invalid = np.sum((g(x_plus) > 0.0))
            effort_counts += 1
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. g(x_plus) < 0...exiting!")
                return x, nu, lam
        # we then continue to multiply s by beta until we have norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2
        nu_plus = nu + s*delt_nu
        lam_plus = lam + s*delt_lam
        r_dual_plus = grads.cd4_gf(x_plus, f) + grads.cd4_gvf(x_plus,g).T @ nu_plus + grads.cd4_gvf(x_plus,h).T @ lam_plus
        r_primal_plus = h(x_plus)
        eta_hat_plus = -1.0 * g(x_plus) @ nu_plus
        t_plus = m * mu / eta_hat_plus
        r_central_plus = -1.0 / t_plus * np.ones(m) - np.diag(g(x_plus)) @ nu_plus
        r_plus = np.concatenate((r_dual_plus, r_central_plus, r_primal_plus))
        is_rplus_smaller = np.linalg.norm(r_plus) <=  (1.0 - alpha * s) * np.linalg.norm(r)
        effort_counts = 0
        while( not is_rplus_smaller):
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2...exiting!")
                return x, nu, lam
            effort_counts += 1
            s *= beta 
            x_plus = x + s*delt_x
            nu_plus = nu + s*delt_nu
            lam_plus = lam + s*delt_lam
            r_dual_plus = grads.cd4_gf(x_plus, f) + grads.cd4_gvf(x_plus,g).T @ nu_plus + grads.cd4_gvf(x_plus,h).T @ lam_plus
            r_primal_plus = h(x_plus)
            eta_hat_plus = -1.0 * g(x_plus) @ nu_plus
            if(eta_hat_plus < 0.0):
                #print("New Surrogate Duality Gap Flipped...Using old gap to stabilize!")
                eta_hat_plus = eta_hat
            #else:
            t_plus = m * mu / eta_hat_plus
            r_central_plus = -1.0 / t_plus * np.ones(m) - np.diag(g(x_plus)) @ nu_plus
            r_plus = np.concatenate((r_dual_plus, r_central_plus, r_primal_plus))
            is_rplus_smaller = (np.linalg.norm(r_plus) <=  (1.0 - alpha * s) * np.linalg.norm(r))
        x = x_plus.copy() #+ 1e-6*np.random.randn(n)
        nu = nu_plus.copy() #+ 1e-6*np.random.randn(m)
        lam = lam_plus.copy() #+ 1e-6*np.random.randn(p)
        t = t_plus
        eta_hat = eta_hat_plus
        #e_vals, _ = np.linalg.eig(KKT)
        
        num_its += 1 
        # Compute new Residual r
        r_dual = grads.cd4_gf(x, f) + grads.cd4_gvf(x,g).T @ nu + grads.cd4_gvf(x,h).T @ lam
        r_central = -1.0 / t * np.ones(m) - np.diag(g(x)) @ nu 
        r_primal = h(x)
        r = np.concatenate((r_dual,r_central,r_primal))
        print("Duality Gap: ", np.round(eta_hat,8), ", Objective: ", np.round(f(x),8), 
                " s: ", np.round(s,5), "KKT cond. #: ", np.round(np.linalg.cond(KKT),5), "Norm resid: ", np.linalg.norm(r))
                #"KKT min/max e-vals: ", np.round(np.min(e_vals),5), ", ", np.round(np.max(e_vals),5))
        # Check end conditions
        gate1 = np.linalg.norm(r_dual) <= ep_feas and np.linalg.norm(r_primal) <= ep_feas and eta_hat <= ep
        gate2 = False #f(x) < ep_feas
        if(gate1 or gate2 ):
            if(gate1):
                print("Primal-Dual Residuals are Below ep_feas...completed optimization")
            if(gate2):
                print("Cost is below ep_feas...completed optimization")
            
            ipopt_done_flag = True
        if(num_its > MAX_ITS):
            print("Max Number of IPOPT Iterations Reached...Termininating!")
            ipopt_done_flag = True
    print("IPOPT Took ", num_its, " iterations")
    return x, nu, lam

# This set of code implements methods to solve constrained and nonconstrained optimization problems 
def gradx_lagrange(x,lam,f,h,g,t):
    grad_fx = grads.cd4_gf(x, f)
    hx = h(x)
    m = hx.size
    grad_hx = grads.cd4_gvf(x, h).T # denom wise gradient
    Jac_hx = grad_hx.T 
    grad_gx = grads.cd4_gvf(x, g).T # denom wise gradient
    lagange_gradx = grad_fx.copy()
    for i in range(m):
        lagange_gradx -= 1/t * Jac_hx[i] / hx[i]
    lagange_gradx += grad_gx @ lam
    return lagange_gradx

def gradxx_lagrange(x,lam,f,h,g,t):
    hx = h(x)
    m = hx.size
    gx = g(x)
    p = gx.size
    n = x.size
    grad_fxx = grads.cd4_dgf(x, f)
    gradT_hx = grads.cd4_gvf(x, h)
    grads_hxx = grads.cd4_dgvf(x, h)
    grads_gxx = grads.cd4_dgvf(x, g)
    lagrange_gradxx = grad_fxx.copy()
    for i in range(m):
        lagrange_gradxx += 1/t * ( grads_hxx[i]/(-hx[i]) + np.outer(gradT_hx[i], gradT_hx[i])/(hx[i]**2) )
    for i in range(p):
        lagrange_gradxx += lam[i] * grads_gxx[i]

    if np.any( np.abs(np.linalg.eig(lagrange_gradxx)[0]) < 1e-5 ):
        lagrange_gradxx += np.eye(n)*1e-3
    return lagrange_gradxx

def gradxlam_lagrange(x,g):
    grad_gx = grads.cd4_gvf(x, g).T # denom wise gradient
    return grad_gx

def log_barrier_phaseI(f,g,h,x):
    s = np.array([ np.max(h(x)) + 1 ]).reshape(-1)
    x = np.concatenate((s,x))
    h_phaseI = lambda x : (h(x[1:]) - x[0]).reshape(-1)
    f_phaseI = lambda x : np.array([x[0]]).reshape(-1)
    g_phaseI = lambda x : g(x[1:]).reshape(-1)
    x, _ = log_barrier(f_phaseI, g_phaseI, h_phaseI, x, zero_val_stop = True) # stop once cost less than 0
    return x[1:], x[0]

# f-> minimization function
def log_barrier(f, g, h, x, zero_val_stop = False):
    t = 1 
    mu = 3
    eps = 1e-8
    norm2_gradL = 1
    n = x.size
    p = g(x).size
    m = h(x).size
    if np.any(h(x) > 0):
        x, s = log_barrier_phaseI(f, g, h, x)
        if s > 0:
            print("Cannot find a feasible solution!")
            return None, None
    lam = np.random.uniform(-10,10, p)
    step = 0
    while (norm2_gradL > eps) and (m/t > eps):
        if zero_val_stop:
            if f(x) < 0:
                return x, lam
        Grad_G = gradxlam_lagrange(x,g)
        gradx_L = gradx_lagrange(x,lam,f,h,g,t)
        Hessxx_L = gradxx_lagrange(x,lam,f,h,g,t)
        norm2_gradL = np.linalg.norm(gradx_L)**2
        print("After Step ", step, "Norm^2 Grad Lagrangian: ", norm2_gradL)
        resid = np.concatenate((gradx_L, g(x)))
        Hess = np.vstack(( np.hstack(( Hessxx_L , Grad_G )) , np.hstack(( Grad_G.T, np.zeros((p,p))  )) ))
        if( np.any( np.abs(np.linalg.eig(Hess)[0]) < 1e-8 ) ):
            print("Hess SPD : Boosting!")
            Hess += np.eye(n+p)*1e-6
        delta = -1.0 * np.linalg.solve(Hess, resid)
        dx = delta[0:n]
        dlam = delta[n:]
        # backtrack
        alpha = 1.0 
        beta = 0.7
        count = 0
        MAX_ITS_LS = 35
        while np.any( h(x + alpha * dx) > 0 ):
            alpha *= beta 
            count += 1
            if(count > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. g(x_plus) < 0...exiting!")
                return x, lam
                # start backtracting at 0.99*s_max and multiply s by beta until we have g(x_plus) < 0

        # we then continue to multiply s by beta until we have norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2
        delta = 0.1
        s = 0.99 * alpha
        x_plus = x + s * dx
        lam_plus = lam + s*dlam
        r_dual_plus = gradx_lagrange(x_plus, lam_plus, f, h, g, t)
        r_primal_plus = g(x_plus)
        r_plus = np.concatenate((r_dual_plus, r_primal_plus))
        is_rplus_smaller = np.linalg.norm(r_plus) <=  (1.0 - alpha * s) * np.linalg.norm(resid)
        effort_counts = 0
        while( not is_rplus_smaller):
            if(effort_counts > MAX_ITS_LS):
                print("MAX_ITS_LS Reached! Cannot find a x_plus s.t. norm(r_plus)_2 <= (1-alpha*s)*norm(r)_2...exiting!")
                return x, lam
            effort_counts += 1
            s *= beta 
            x_plus = x + s*dx
            lam_plus = lam + s*dlam
            r_dual_plus = gradx_lagrange(x_plus, lam_plus, f, h, g, t)
            r_primal_plus = g(x_plus)
            r_plus = np.concatenate((r_dual_plus, r_primal_plus))
            is_rplus_smaller = (np.linalg.norm(r_plus) <=  (1.0 - delta * s) * np.linalg.norm(resid))

        x = x + s * dx 
        lam = lam + s * dlam
        t *= mu 
        step += 1
    return x, lam

def test_ipopt():
    #f = lambda x : np.array([ (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 ])#np.array([ np.sin(x[1])*np.exp((1-np.cos(x[0]))**2) + np.cos(x[0])*np.exp( (1-np.sin(x[1]))**2 ) + ([0] - x[1])**2 ])
    #g = lambda x : np.array([ x[0]**2 + x[1]**2 - 2])
    #h = lambda x : np.array([])
    #x = np.array([-0.5, -0.5])
    
    f = lambda x : np.array([ -np.prod(x) ])#np.array([ np.sin(x[1])*np.exp((1-np.cos(x[0]))**2) + np.cos(x[0])*np.exp( (1-np.sin(x[1]))**2 ) + ([0] - x[1])**2 ])
    g = lambda x : np.array([ 2*(x[0]*x[1] + x[0]*x[2] + x[1]*x[2]) - 150, -x[0], -x[1], -x[2] ])
    h = lambda x : np.array([])
    x = np.array([0.1, 0.1, 0.1])

    ipopt(f, g, h, x)

def test_logbar():
    #f = lambda x : np.array([ (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 ])#np.array([ np.sin(x[1])*np.exp((1-np.cos(x[0]))**2) + np.cos(x[0])*np.exp( (1-np.sin(x[1]))**2 ) + ([0] - x[1])**2 ])
    #g = lambda x : np.array([ x[0]**2 + x[1]**2 - 2])
    #h = lambda x : np.array([])
    #x = np.array([-0.5, -0.5])
    rand_num = np.random.randint(0,10000)
    np.random.seed(rand_num)
    print("Seeding as ", rand_num)
    n = 3
    m = 2
    p = 2
    Q = np.random.randn(n,n)
    Q = Q @ Q.T
    print(Q)
    c = np.random.randn(n)
    A = np.random.randn(p,n)
    b = np.random.randn(p)
    C = np.random.randn(m,n)
    d = np.random.randn(m)
    x = np.random.randn(n)
    f = lambda x : np.array([ c.T @ x ]).reshape(-1) #np.array([ np.sin(x[1])*np.exp((1-np.cos(x[0]))**2) + np.cos(x[0])*np.exp( (1-np.sin(x[1]))**2 ) + ([0] - x[1])**2 ])
    g = lambda x : np.array([ A @ x - b ]).reshape(-1)
    h = lambda x : np.array([ C @ x - d ]).reshape(-1)
    print("A:\n{}\nb:\n{}\nC:\n{}\nd:\n{}\n".format(A,b,C,d))

    #x, _ = ipopt(f,g,h,x)

    import cvxpy as cp 
    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T @ x),
                    [A @ x == b, C @ x <= d])
    prob.solve()
    cvx_sol = x.value.copy() if x.value is not None else None
    cvx_sol_min = f(cvx_sol) if x.value is not None else None
    x = np.random.randn(n)
    x_ipopt,_,_ = ipopt(f,h,g,x)
    x = np.random.randn(n)
    x_lb, _ = log_barrier(f,g,h,x)
    
    print("cvxpy answer: f(x)={}, x={}".format(cvx_sol_min, cvx_sol))
    print("ipopt answer: f(x)={}, x={}".format(f(x_ipopt) if x_ipopt is not None else None, x_ipopt))
    print("logbar answer: f(x)={}, x={}".format(f(x_lb) if x_lb is not None else None, x_lb))


    if x is not None:
        fmin = f(x_lb)
    else:
        fmin = None
    print("\nCVX Solution has status {}, min {} and x={}".format(prob.status, cvx_sol_min, cvx_sol))
    print("HBrew Solution has min {} and x={}".format(fmin, x))

    foobar = 2

def dyn_obj(Q, Qf, R, _x, x0):
    x = np.concatenate((x0,_x))
    n = Q.shape[0]
    m = R.shape[0]
    nm = n+m
    num_props = int( (x.size-n) / nm )
    cost = x[-n:].T @ Qf @ x[-n:]
    for i in range(num_props):
        Xi = x[i*nm:]
        xi = Xi[0:n]
        ui = Xi[n:nm]
        cost += xi.T @ Q @ xi + ui.T @ R @ ui
    return np.array([cost])

def dyn_eq_constrs(_x, Phik, Gamk, x0):
    x = np.concatenate((x0,_x))
    n = Phik.shape[0]
    m = Gamk.shape[1]
    nm = n+m
    rs = []
    num_props = int( (x.size-n) / nm )
    for i in range(num_props):
        Xi = x[i*nm:]
        xi = Xi[0:n]
        ui = Xi[n:nm]
        xip1 = Xi[nm:nm+n]
        r = xip1 - Phik @ xi - Gamk @ ui
        rs.append(r)
    rs = np.array(rs).reshape(-1)
    return rs

def dyn_ineq_constrs(_x, Phik, Gamk, x0):
    x = np.concatenate((x0,_x))
    n = Phik.shape[0]
    m = Gamk.shape[1]
    nm = n+m
    rs = []
    num_props = int( (x.size-n) / nm )
    for i in range(num_props):
        Xi = x[i*nm:]
        ui = Xi[n:nm]
        rs.append([ui - 10])
        rs.append([-ui - 10])
    rs = np.array(rs).reshape(-1)
    return rs

def dyn_sim(_x, Phik, Gamk, x0):
    x = np.concatenate((x0,_x))
    n = Phik.shape[0]
    m = Gamk.shape[1]
    nm = n+m
    num_props = int( (x.size-n) / nm )
    xk = x0.copy()
    for i in range(num_props):
        Xi = x[i*nm:]
        xi = Xi[0:n]
        ui = Xi[n:nm]
        xip1 = Xi[nm:nm+n]
        print("\nxk: ", xk)
        print("xi: ", xi)
        print("ui: ", ui)
        xk = Phik @ xk + Gamk @ ui
        print("xk1: ", xk)
        print("xk1i: ", xip1)

def test_logbar2():
    #f = lambda x : np.array([ (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 ])#np.array([ np.sin(x[1])*np.exp((1-np.cos(x[0]))**2) + np.cos(x[0])*np.exp( (1-np.sin(x[1]))**2 ) + ([0] - x[1])**2 ])
    #g = lambda x : np.array([ x[0]**2 + x[1]**2 - 2])
    #h = lambda x : np.array([])
    #x = np.array([-0.5, -0.5])
    rand_num = np.random.randint(0,10000)
    np.random.seed(rand_num)
    print("Seeding as ", rand_num)
    n = 2
    m = 1
    nm = n+m
    dt = 0.1
    steps = 15
    Q = 0.25*np.eye(n)
    Qf = 8*np.eye(n)
    R = 0.05 * np.eye(m)
    A = np.array([0,1,-3,-1]).reshape((2,2))
    B = np.array([[0,1.0]]).T
    Phik = get_STM(A, dt, 4)
    Gamk = get_Gam(A, B, dt, 4)
    x0 = x = np.random.randn(n)
    print(Q,R)
    
    f = lambda x : dyn_obj(Q, Qf, R, x, x0)
    g = lambda x : dyn_eq_constrs(x, Phik, Gamk, x0)
    h = lambda x : dyn_ineq_constrs(x, Phik, Gamk, x0)
    
    #f = lambda x : cvxpy_dyn_obj(Q, Qf, R, x, x0)
    #g = lambda x : cvxpy_dyn_eq_constrs(x, Phik, Gamk, x0)
    #h = lambda x : cvxpy_dyn_ineq_constrs(x, Phik, Gamk, x0)

    import cvxpy as cp 
    # Define and solve the CVXPY problem.
    x = cp.Variable((n, steps + 1))
    u = cp.Variable((m, steps))

    cost = 0
    constr = []
    for t in range(steps):
        cost += cp.quad_form(x[:, t],Q) + cp.quad_form(u[:, t],R)
        constr += [x[:, t + 1] == Phik @ x[:, t] + Gamk @ u[:, t], cp.norm(u[:, t], "inf") <= 10]
    cost += cp.quad_form(x[:, steps],Qf)
    # sums problem objectives and concatenates constraints.
    constr += [x[:, 0] == x0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()
    cvx_sol = x.value.copy() if x.value is not None else None
    cvx_sol_min = problem.value if x.value is not None else None


    x = np.random.randn(steps*nm)
    x_ipopt,_,_ = ipopt(f,h,g,x)
    dyn_sim(x_ipopt, Phik, Gamk, x0)
    x = np.random.randn(steps*nm)
    x_lb, _ = log_barrier(f,g,h,x)
    dyn_sim(x_lb, Phik, Gamk, x0)

    Ks, Ss = lqr_gains(Phik, Gamk, Q, R, Qf, steps)
    print("LQR Cost: ", x0.T @ Ss[0] @ x0)
    xk = x0.copy()
    for i in range(steps):
        _xk = xk.copy()
        uk = - Ks[i] @ xk 
        xk = Phik @ xk + Gamk @ uk
        print("LQR k={}: xk={}, uk={}, xk1={}".format(i, _xk, uk, xk))

    
    #print("cvxpy answer: f(x)={}, x={}".format(cvx_sol_min, cvx_sol))
    print("ipopt answer: f(x)={}, x={}".format(f(x_ipopt) if x_ipopt is not None else None, x_ipopt))
    print("logbar answer: f(x)={}, x={}".format(f(x_lb) if x_lb is not None else None, x_lb))


    if x is not None:
        fmin = f(x_lb)
    else:
        fmin = None
    #print("\nCVX Solution has status {}, min {} and x={}".format(prob.status, cvx_sol_min, cvx_sol))
    print("HBrew Solution has min {} and x={}".format(fmin, x_lb))

    foobar = 2

# Iterative LQR Schemes
def test_ref_track_follow():
    L = 0.14 # meters
    dt = 1.0 / 20.0
    steps = int( 1 / dt * 8 )
    ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])
    v = np.linspace(0.5, 1.25, steps)
    phi = np.pi/8*np.sin([t for t in np.linspace(0,2*np.pi, steps)])
    us = np.vstack((v,phi)).T 
    x0 = np.zeros(3)
    xs = np.zeros((steps+1, 3))
    xs[0] = x0.copy()
    xk = x0.copy()
    count = 1
    for u in us: 
        xk = runge_kutta4(ctime_ack_model, xk, u, dt)
        xs[count] = xk.copy()
        count += 1
    # Use xs as a trajectory

    x_bars = xs 
    #x_bars[:,2] *= 0
    u_bars = np.zeros((steps, 2))
    u_bars[:,0] = 0.5
    u_bars[:,1] = 0.1

    Q = 0.25*np.array([[1,0,0], [0,1,0], [0,0,1]])
    #Q[2,2] = 0
    Qf = 8*np.eye(3)
    #Qf[2,2] = 0
    R = 0.20 * np.eye(2)

    tderiv_ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])


    lqr_ltv_rt1(xs, x0, x_bars, u_bars, ctime_ack_model, ctime_ack_model_JacX, ctime_ack_model_JacU, Q, Qf, R, dt)

    foo = 2 

# Iterative LQR Schemes
def test_ref_track_follow2():
    L = 0.14 # meters
    dt = 1.0 / 20.0
    steps = int( 1 / dt * 8 )
    ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])
    v = np.linspace(0.5, 1.25, steps)
    phi = np.pi/8*np.sin([t for t in np.linspace(0,2*np.pi, steps)])
    us = np.vstack((v,phi)).T 
    x0 = np.zeros(3)
    xs = np.zeros((steps+1, 3))
    xs[0] = x0.copy()
    xk = x0.copy()
    count = 1
    for u in us: 
        xk = runge_kutta4(ctime_ack_model, xk, u, dt)
        xs[count] = xk.copy()
        count += 1
    # Use xs as a trajectory

    x_bars = xs 
    #x_bars[:,2] *= 0
    u_bars = np.zeros((steps, 2))
    u_bars[:,0] = 0.5
    u_bars[:,1] = 0.1

    Q = 0.25*np.array([[1,0,0], [0,1,0], [0,0,1]])
    #Q[2,2] = 0
    Qf = 4*np.eye(3)
    #Qf[2,2] = 0
    R = 0.1 * np.eye(2)

    tderiv_ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])


    lqr_ltv_rt2(xs, x0, x_bars, u_bars, ctime_ack_model, ctime_ack_model_JacX, ctime_ack_model_JacU, Q, Qf, R, dt)

    foo = 2 


def test_good_start():
    L = 0.14 # meters
    dt = 1.0 / 20.0
    steps = int( 1 / dt * 8 )
    ctime_ack_model = lambda x, u : np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[0]/L * np.tan(u[1]) ])
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])
    v = np.linspace(0.5, 1.25, steps)
    phi = np.pi/8*np.sin([t for t in np.linspace(0,2*np.pi, steps)])
    us = np.vstack((v,phi)).T 
    x0 = np.zeros(3)
    xs = np.zeros((steps+1, 3))
    xs[0] = x0.copy()
    xk = x0.copy()
    count = 1
    for u in us: 
        xk = runge_kutta4(ctime_ack_model, xk, u, dt)
        xs[count] = xk.copy()
        count += 1
    # Use xs as a trajectory to find some us
    x_bars = xs 
    ctime_ack_model_JacX = lambda x, u : np.array([ [0, 0, -u[0] * np.sin(x[2])], [0,0,u[0] * np.cos(x[2])], [0, 0, 0] ])
    ctime_ack_model_JacU = lambda x, u : np.array([ [np.cos(x[2]), 0], [np.sin(x[2]), 0], [1/L * np.tan(u[1]), u[0]/L /np.cos(u[1])**2] ])
    taylor_order = 4
    n_ref = 3
    rp = x_bars[:,:n_ref]
    m_ref = 3 - n_ref
    u = np.array([v[0], phi[0]]) + np.random.randn(2) * 0.01
    x = x0.copy() + np.random.randn(3) * 0.01
    us_good = [] 
    xs_good = [] 
    Q = np.eye(3)
    #Q[2,2] = 1e-4
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.4f}'.format}, linewidth=160)
    for i in range(us.shape[0]):
        xb1 = rp[i+1]
        xk1 = runge_kutta4(ctime_ack_model,x,u,dt)
        JacA = ctime_ack_model_JacX(x,u)
        JacB = ctime_ack_model_JacU(x,u)
        Phi = get_STM(JacA, dt, taylor_order)
        Gam = get_Gam(JacA, JacB, dt, taylor_order)
        #Lc = -np.vstack(( np.zeros((n_ref,m_ref)), np.eye(m_ref) ))
        #A = np.hstack((Phi, Gam)) #, Lc))
        H = Gam
        e_k1 = np.concatenate((xb1, np.zeros(m_ref))) - xk1 - (Phi @ (np.concatenate((rp[i], np.zeros(m_ref))) - x))
        y = np.linalg.inv(H.T @ Q @ H) @ H.T @ Q @ e_k1 # y=
        du = np.linalg.inv(H.T @ Q @ H) @ H.T @ Q @ e_k1 # y=
        du[0] = np.clip(du[0], -.1, .1)
        du[1] = np.clip(du[1], -.04, .04)
        #ek = A.T @ y 
        #dx = ek[0:3]
        #du = ek[3:5]
        #x1 = ek[5]
        #x = x + dx 
        u = u + du + np.random.randn(2) * 0.0001
        u[1] = np.clip(u[1], -np.pi/6, np.pi/6)
        print("\nStep i={}, Real State: {}, Proj State: {}".format(i, xs[i], x) )
        print("Step i={}, Real Control: {}, Proj Control: {}".format(i, us[i], u) )
        us_good.append(u)
        xs_good.append(x)
        x = runge_kutta4(ctime_ack_model,x,u,dt)
    
    print("foobar!")





if __name__ == "__main__":
    #test_ref_track_follow2()
    #test_ipopt()
    #test_logbar()
    #test_logbar2()
    #test_good_start()
    #test_fast_mpc_linear()
    test_fast_mpc_nonlinear()
    #test_fast_mpc_nonlinear2()