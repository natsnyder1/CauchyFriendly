import numpy as np
import pickle 
import sys, os
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg',force=True)
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
file_dir = os.path.dirname(os.path.abspath(__file__))
import cvxpy as cp
from scipy.stats import chi2 


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def euler_angles_to_rotation_matrix(yaw, pitch, roll):
    cosY = np.cos(yaw)
    sinY = np.sin(yaw)
    cosP = np.cos(pitch)
    sinP = np.sin(pitch)
    cosR = np.cos(roll)
    sinR = np.sin(roll)
    R = np.array([[ cosP*cosY, cosP*sinY, -sinP],
                  [-cosR*sinY + cosY*sinP*sinR, cosR*cosY + sinP*sinR*sinY, cosP*sinR],
                  [ cosR*cosY*sinP + sinR*sinY, cosR*sinP*sinY - cosY*sinR, cosP*cosR]])
    return R

def create_unit_sphere(yaw_increment, angle_increment):
    num_ang_incrs = int(np.pi / angle_increment + 1)
    num_yaw_incrs = int(2*np.pi / yaw_increment + 1)
    angs = np.linspace(0,np.pi, num_ang_incrs)
    yaws = np.linspace(0,2*np.pi, num_yaw_incrs)
    plane_points = np.array([ (np.cos(y), np.sin(y), 0) for y in yaws])

    points = plane_points.copy()
    # Roll rotation
    for ang in angs[1:-1]:
        R = euler_angles_to_rotation_matrix(0, 0, ang)
        points = np.vstack((points, plane_points @ R)) # (R.T @ plane_points[i,:].T)
    
    # Roll rotation
    for ang in angs[1:-1]:
        R = euler_angles_to_rotation_matrix(0, ang, 0)
        points = np.vstack((points, plane_points @ R)) # (R.T @ plane_points[i,:].T)
    return points 

def get_cross_along_radial_errors_cov(xhat, Phat, xt):
    # position and velocity 3-vector components
    rh = xhat[0:3]
    rhn = np.linalg.norm(rh)
    vh = xhat[3:6]
    vhn = np.linalg.norm(xhat[3:6])
    # Unit directions 
    uv = vh / vhn # x-axis - along track
    ur = rh / rhn # z-axis - radial
    uc = np.cross(ur, uv) # y-axis - cross track
    R = np.vstack((uv,uc,ur))

    # Error w.r.t input coordinate frame 
    e = xt[0:3] - xhat[0:3]
    # Error w.r.t track frame
    e_track = R @ e
    # Error Covariance w.r.t track frame
    P_track = R @ Phat @ R.T 

    return e_track, P_track, R

def test_along_cross_track_covariance_transformations():
    # Load Data 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_gausstrials_25_1709746486.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    # Obtain a trial
    mc_trial = 0
    xs,zs,ws,vs = mc_dic["sim_truth"][mc_trial]
    xs_kf,Ps_kf = mc_dic["ekf_runs"][mc_trial]
    
    # Get 1-sig covariance ellipse
    it = 1000
    x_kf = xs_kf[it]
    P_kf = Ps_kf[it]
    xt = xs[it]
    et = xt - x_kf
    D, V = np.linalg.eig(P_kf[0:3,0:3])
    E1 = V @ np.diag(D)**0.5
    unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
    ellipse_points = (E1 @ unitsphere.T).T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.scatter(unitsphere[:,0], unitsphere[:,1], unitsphere[:,2], color='b')
    ax.scatter(ellipse_points[:,0], ellipse_points[:,1], ellipse_points[:,2], color='r')
    ax.arrow3D(0,0,0, et[0], et[1], et[2], color='cyan')
    # unit velocity direction 
    uv = (x_kf[3:6] / np.linalg.norm(x_kf[3:6])) * 0.001
    ax.arrow3D(0,0,0, uv[0], uv[1], uv[2], color='black')
    ur = (x_kf[0:3] / np.linalg.norm(x_kf[0:3])) * 0.001
    ax.arrow3D(0,0,0, ur[0], ur[1], ur[2], color='pink')
    ax.arrow3D(0,0,0, ur[0], ur[1], ur[2], color='green')
    theta = np.arccos(uv @ ur)
    track_x, track_P, R = get_cross_along_radial_errors_cov(x_kf, P_kf[0:3,0:3], xt)
    D, V = np.linalg.eig(track_P)
    E2 = V @ np.diag(D)**0.5
    ellipse_points2 = (E2 @ unitsphere.T).T
    ax.scatter(ellipse_points2[:,0], ellipse_points2[:,1], ellipse_points2[:,2], color='m')
    ax.arrow3D(0,0,0, track_x[0], track_x[1], track_x[2], color='brown')
    plt.show()
    foobar = 3

def cartesian2keplerian(x, is_units_km = True, with_print = False):
    r = x[0:3].copy() # radius from earth center 
    v = x[3:6].copy() # velocity of spacecraft
    if not is_units_km:
        r /= 1e3 # convert to kmeters
        v /= 1e3 # convert to kmeters
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    mu = 3.9860044188e5 #km^3/(kg*s^2)
    # Calculate Angular Momentum: h
    h = np.cross(r,v)
    hn = np.linalg.norm(h)
    # Node vector (which points towards ascending node and the true anomoly)
    n = np.cross( np.array([0,0,1.0]), h )
    nn = np.linalg.norm(n)
    # Eccentricity vector: evec
    evec = ( (vn**2 - mu/rn) * r - (r @ v ) * v ) / mu 
    # Eccentricity: e
    e = np.linalg.norm(evec)
    # Parabolic Orbit
    if np.isclose(e, 1, rtol=0,atol=1e-6):
        print("Warning, parabolic orbit has been detected! Not implemented!")
        return None 
    else:
        # Specific mechanical energy: E
        SME = vn**2/2 - mu/rn
        # Semi major axis: a
        a = -mu / (2*SME)
        # Semi parameter: p
        p = a*(1-e**2)
        # Semi minor axis: b
        b = a * (1-e**2)**0.5
        # Apogee and Perigee 
        apogee = a*(1+e)
        perigee = a*(1-e)
        # Orbit inclination angle: i 
        i = np.arccos(h[2]/hn)
        # Longitude of ascending node (or Right Ascension of Ascending Node - RAAN): Omega
        if np.isclose(nn, 0, rtol=0, atol=1e-6):
            Omega = "undefined for equitorial orbits"
        else:
            Omega = np.arccos(n[0]/nn)
            if n[1] < 0:
                Omega = 2*np.pi - Omega
        # Circular Orbit 
        if np.isclose(e, 0, rtol=0,atol=1e-6):
            print("Circular orbit has been detected!") 
            # True anomoly: nu
            nu = "undefined for circular orbit"
            # Eccentricity Anomoly: E
            E = "undefined for circular orbit"
            # argument of perigee: omega
            omega = "undefined for circular orbit"
            # Mean anomoly: M
            M = "undefined for circular orbit"
            # Longitude of perigee 
            omega_tild = "undefined for circular orbit"
            # True longitude of perigee
            omega_tild_true = "undefined for circular orbit"
            # Argument of latitude 
            u = "undefined for circular orbit"
            # True longitude
            lam_true = "undefined for circular orbit"

        # Elliptic orbit
        else:
            print("Elliptic orbit has been detected!") 
            # True anomoly: nu
            nu = np.arccos( (evec @ r) / (e*rn) )
            if r @ v < 0:
                nu = 2*np.pi - nu
            # Eccentricity Anomoly: EA
            E = 2 * np.arctan2(np.tan(nu/2) , np.sqrt( (1+e)/(1-e)) )
            # Argument of perigee: omega
            omega = np.arccos( (n @ evec) / (nn * e) )
            if evec[2] < 0:
                omega = 2*np.pi - omega
            # Mean anomoly: M
            M = E - e*np.sin(E)
            # Longitude of perigee: omega_tild
            if np.isclose(nn, 0, rtol=0, atol=1e-6):
                omega_tild = "undefined for equitorial orbit"
            else:
                omega_tild = Omega + omega 
            # True longitude of perigee: omega_tild_true
            omega_tild_true = np.arccos( evec[0] / e)
            if evec[1] < 0:
                omega_tild_true = 2*np.pi - omega_tild_true
            # Argument of latitude: u
            if np.isclose(nn, 0, rtol=0, atol=1e-6):
                u = "undefined for equitorial orbit"
            else:
                u = np.arccos( (n @ r) / (nn*rn) )
                if r[2] < 0:
                    u = 2*np.pi - u
            # True longitude: lam_true
            lam_true = np.arccos( r[0] / rn )
            if r[1] < 0:
                lam_true = 2*np.pi - lam_true

    kepler_dic = {}
    kepler_dic["evec"] = [evec, "eccentricity vector"]
    kepler_dic["e"] = [e, "eccentricity"]
    kepler_dic["n"] = [n, "node vector pointing towards ascending node / true anomoly"]
    kepler_dic["SME"] = [SME, "Specific Mechanical Energy"]
    kepler_dic["a"] = [a, "Semimajor axis"]
    kepler_dic["b"] = [b, "Semiminor axis"]
    kepler_dic["p"] = [p, "Semi-parameter"]
    kepler_dic["E"] = [E, "Eccentric Anomoly"]
    kepler_dic["apogee"] = [apogee, "farthest point to earth in orbit"]
    kepler_dic["perigee"] = [perigee, "closest point to earth in orbit"]
    kepler_dic["i"] = [i, "orbit inclination angle from plane of reference"]
    kepler_dic["nu"] = [nu, "true anomoly"]
    kepler_dic["omega"] = [omega, "argument of periapsis"]
    kepler_dic["Omega"] = [Omega, "longitude of the ascending node, or Right Ascension of Ascending Node (RAAN)"]
    kepler_dic["M"] = [M, "Mean anomoly"]
    kepler_dic["omega_tild"] = [omega_tild, "Longitude of perigee"]
    kepler_dic["omega_tild_true"] = [omega_tild_true, "True longitude of perigee"]
    kepler_dic["u"] = [u, "Argument of latitude"]
    kepler_dic["lam_true"] = [lam_true, "True longitude"]
    kepler_dic["aeiOov"] = [ [a,e,i,Omega,omega,nu], kepler_dic["a"][1] + "; " + kepler_dic["e"][1] + "; " + kepler_dic["i"][1] + "; " + kepler_dic["Omega"][1] + "; " + kepler_dic["omega"][1] + "; " + kepler_dic["nu"][1] ]
    if with_print:
        print("Symbology Taken from 'Fundamentals of Astrodynamics and Applications'")
        for k,v in kepler_dic.items():
            print(k,":", v[0], "->", v[1]) 
    return kepler_dic

def convert_to_along_cross_radial_track(xkf_preds, Pkf_preds, xs, start_idx, with_plot = True):
    es_track = [] 
    Ps_track = []
    Rs_track = []
    for xhat,Phat,xt in zip(xkf_preds,Pkf_preds,xs[start_idx:,:]):
        et, Pt, R = get_cross_along_radial_errors_cov(xhat, Phat[0:3,0:3], xt)
        es_track.append(et)
        Ps_track.append(Pt)
        Rs_track.append(R)
    es_track = np.array(es_track)
    Ps_track = np.array(Ps_track)
    Rs_track = np.array(Rs_track)
    if with_plot:
        sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_track])
        Ts = np.arange(xs.shape[0])
        plt.figure()
        plt.suptitle("Predictor Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
        plt.subplot(3,1,1)
        plt.plot(Ts[start_idx:], es_track[:, 0], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 0], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 0], 'm--')
        plt.subplot(3,1,2)
        plt.plot(Ts[start_idx:], es_track[:, 1], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 1], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 1], 'm--')
        plt.subplot(3,1,3)
        plt.plot(Ts[start_idx:], es_track[:, 2], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 2], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 2], 'm--')
        plt.show()
        foobar=2
    return es_track, Ps_track, Rs_track

def position_quantiles(quantiles, qs, xs_kf, Ps_kf, xs, start_idx, with_plot = True):
    trial_filter_hits = np.zeros(qs.size)
    min_n = np.min([xs_kf.shape[1], xs.shape[1]])
    assert min_n >= 3
    N = xs.shape[0]
    for i in range(start_idx, N):
        x_kf = xs_kf[i-start_idx]
        P_kf = Ps_kf[i-start_idx]
        e_kf = (xs[i, 0:min_n] - x_kf[0:min_n])[0:3]
        Pinv = np.linalg.inv(P_kf[0:3, 0:3])
        score = e_kf.T @ Pinv @ e_kf
        hits = score < qs
        #print("Score: {}, qs:{}".format(score, qs))
        trial_filter_hits += hits
    trial_filter_hits /= (N-start_idx)
    print("Checking % Quantiles of:\n  ", quantiles)
    print("Filter % Quantiles:\n  ", trial_filter_hits)
    if with_plot:
        xlabels = ["Q=" + str(q*100)+"%\n(Realized {})".format(np.round(trial_filter_hits[i]*100,2)) for i,q in enumerate(quantiles)]
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
        plt.figure()
        plt.title(r"Trial Realization Error Quantiles of the Filtered Position Covariance $P_k$" + "\n" + r"Computed as $R_Q=\frac{\sum_{k=1}^N\mathbb{1}(s_k \leq s_{Q})}{N}, \quad s_k = e^T_k P^{-1}_k e_k,\quad e_k = x^k_{1:3} - \hat{x}^k_{1:3}$")
        plt.ylabel("Trial Realization Quantiles Percentages " + r"$100*R_Q$")
        plt.bar(xlabels, trial_filter_hits*100, color=bar_colors)
        plt.xlabel("Total Filtering Steps in Realization={}, dt={} sec between steps".format(N, 60))
        plt.show()
        foobar = 2
    return trial_filter_hits

def test_convex_proc_noise_fit():
    from test_gmat_leo7 import FermiSatelliteModel
    import cauchy_estimator as ce
    # Load Data 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_gausstrials_2_1709759984.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    
    start_percent = 0.25

    # Load up the fermi sat model
    fermiSat = FermiSatelliteModel(mc_dic["x0"], mc_dic["dt"], mc_dic["std_gps_noise"])
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for("Cd", mc_dic["sas_Cd"], mc_dic["std_Cd"], mc_dic["tau_Cd"], alpha=mc_dic["sas_alpha"])

    # Get Chi-Squared Stats for 1 state, 3 states 
    n_pred_states = 3
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    qs1 = np.array([chi2.ppf(q, 1) for q in quantiles]) # quantile scores
    qs3 = np.array([chi2.ppf(q, 3) for q in quantiles]) # quantile scores

    # Get sim length and set start index 
    N = mc_dic["sim_truth"][0][0].shape[0]
    start_idx = int(N * start_percent)
    mc_trials = mc_dic["trials"]

    # Get array of prediction steps vs chi squared bucket counts
    Ts = np.arange(N) #* mc_dic["dt"]
    print("Loaded Data From: ", dir_path)
    # For each trial, 
    # 1.) Propagate the state and covariance from start idx to end
    # 2.) At each propagation, compute whether the error lies within the covariances selected confidence ellipsoid percents
    for trial in range(mc_trials):
        print("Processing Trial {}/{}".format(trial+1, mc_trials))
        xs,zs,ws,vs = mc_dic["sim_truth"][trial]
        xs_kf,Ps_kf = mc_dic["ekf_runs"][trial]


        # Set satellite at start index 
        x_kf = xs_kf[start_idx]
        P_kf = Ps_kf[start_idx]
        fermiSat.reset_state(x_kf, start_idx)
        # Propagate state and variance in prediction mode
        xkf_preds = [] 
        Pkf_preds = [] 
        for idx in range(start_idx, N):
            # Propagate State and Covariance
            if idx != start_idx:
                Phi = fermiSat.get_transition_matrix(mc_dic["STM_order"])
                x_kf = fermiSat.step()
                P_kf = Phi @ P_kf @ Phi.T
            xkf_preds.append(x_kf[0:6])
            Pkf_preds.append(P_kf[0:3, 0:3])
        xkf_preds = np.array(xkf_preds)
        Pkf_preds = np.array(Pkf_preds)
        
        #position_quantiles(quantiles, qs3, xkf_preds, Pkf_preds, xs, start_idx, with_plot = True)
        es_track, Ps_track, Rs_track = convert_to_along_cross_radial_track(xkf_preds, Pkf_preds, xs, start_idx, with_plot = False)
        
        # Setup convex optimization problem in along track direction
        from_end = 5000 # Only optimize over start_idx to N minus from_end steps
        s = qs1[1]
        track_sigmas = np.zeros(3)
        dt = mc_dic["dt"]
        for j in range(3):
            sigma2 = cp.Variable(1)
            constr_list = [sigma2 >= 0] 
            for idx in range(start_idx+1, N-from_end):
                i = idx - start_idx
                Ptt = Ps_track[i][j,j] * 1e6
                ett = es_track[i][j] * 1e3
                constr_list.append(sigma2 >=  (ett**2/(i*s*dt) - Ptt/(i*dt)) ) 
            prob = cp.Problem(cp.Minimize(sigma2), constr_list)
            prob.solve()
            # Print result.
            print("\nThe optimal value for j={} is {}".format(j,prob.value))
            print("Optimal solution for j={} is {}".format(j, sigma2.value))
            print(sigma2.value)
            track_sigmas[j] = sigma2.value
        track_sigmas = np.clip(track_sigmas, 0, np.inf)
        track_sigmas /= 1e6

        # Compute old variance bound 
        sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_track])
        # Now we can recompute the new variance bounds using this additive found bound 
        Ps_track = np.array([_P + np.diag(track_sigmas)*i*dt for i,_P in enumerate(Ps_track)])
        sig_bound2 = np.array([np.diag(_P)**0.5 for _P in Ps_track])
        
        plt.figure()
        plt.suptitle("Corrected Predictor Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
        plt.subplot(3,1,1)
        plt.plot(Ts[start_idx:], es_track[:, 0], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 0], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 0], 'm--')
        plt.plot(Ts[start_idx:], sig_bound2[:, 0], 'r--')
        plt.plot(Ts[start_idx:], -sig_bound2[:, 0], 'r--')
        plt.subplot(3,1,2)
        plt.plot(Ts[start_idx:], es_track[:, 1], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 1], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 1], 'm--')
        plt.plot(Ts[start_idx:], sig_bound2[:, 1], 'r--')
        plt.plot(Ts[start_idx:], -sig_bound2[:, 1], 'r--')
        plt.subplot(3,1,3)
        plt.plot(Ts[start_idx:], es_track[:, 2], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 2], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 2], 'm--')
        plt.plot(Ts[start_idx:], sig_bound2[:, 2], 'r--')
        plt.plot(Ts[start_idx:], -sig_bound2[:, 2], 'r--')
        plt.show()
        foobar=2

        Ps = np.array([R.T @ P @ R for R,P in zip(Rs_track,Ps_track)])
        sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps])
        es = xs[start_idx:, :6] - xkf_preds
        plt.figure()
        plt.suptitle("Corrected Predictor Errors vs 1-Sigma Bound")
        plt.subplot(3,1,1)
        plt.plot(Ts[start_idx:], es[:, 0], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 0], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 0], 'm--')
        plt.subplot(3,1,2)
        plt.plot(Ts[start_idx:], es[:, 1], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 1], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 1], 'm--')
        plt.subplot(3,1,3)
        plt.plot(Ts[start_idx:], es[:, 2], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 2], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 2], 'm--')
        plt.show()
        position_quantiles(quantiles, qs3, xkf_preds, Ps, xs, start_idx, with_plot = True)
        foobar=2

def test_kepler():
    x = np.array([6524.834, 6862.875, 6448.296, 4.901327, 5.533756, -1.976341])
    cartesian2keplerian(x, is_units_km = True, with_print = True)

if __name__ == '__main__':
    #test_along_cross_track_covariance_transformations()
    #test_parametric_fit()
    #test_convex_proc_noise_fit()
    test_kepler()