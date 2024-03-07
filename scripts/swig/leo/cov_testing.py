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

def test_parametric_fit():
    # Load Data 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_gausstrials_25_1709746486.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    # Obtain a trial
    mc_trial = 0
    xs,zs,ws,vs = mc_dic["sim_truth"][mc_trial]
    xs_kf,Ps_kf = mc_dic["ekf_runs"][mc_trial]

    # Fit line to a single orbit and plot fit and points
    # Seems as though half orbit fit with 8th order is the best
    fit_order = 8
    Y = xs[0:44, 0:3]
    ts = np.arange(Y.shape[0]) * (1.0*mc_dic["dt"])
    ts /= ts[-1]
    X = np.zeros((ts.size, fit_order+1))
    for i in range(fit_order+1):
        X[:,i] = ts**i
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    Yest = X @ theta 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Y[:,0], Y[:,1], Y[:,2], color='r')
    ax.plot(Yest[:,0], Yest[:,1], Yest[:,2], color='b')
    print("Max Residual is: ", np.max( np.sum((Y-Yest)**2,axis=1) ) )
    # Now take later half of the orbit, reverse it
    foobar=2
    Y2 = xs[43:87, 0:3]
    Y2 = np.flip(Y2, axis = 0)
    ts = np.arange(Y2.shape[0]) * (1.0*mc_dic["dt"])
    ts /= ts[-1]
    X2 = np.zeros((ts.size, fit_order+1))
    for i in range(fit_order+1):
        X2[:,i] = ts**i
    theta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ Y2
    Yest2 = X2 @ theta2
    ax.scatter(Y2[:,0], Y2[:,1], Y2[:,2], color='k')
    ax.plot(Yest2[:,0], Yest2[:,1], Yest2[:,2], color='g')
    print("Max Residual is: ", np.max( np.sum((Y2-Yest2)**2,axis=1) ) )

    plt.figure()
    diff = X @ (theta - theta2)
    Ts = np.arange(diff.shape[0])
    plt.subplot(3,1,1)
    plt.plot(Ts, diff[:,0])
    plt.subplot(3,1,2)
    plt.plot(Ts, diff[:,1])
    plt.subplot(3,1,3)
    plt.plot(Ts, diff[:,2])
    plt.show()

    foobar=2


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

    # Get Chi-Squared Stats for 1 state
    n_pred_states = 1
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    qs = np.array([chi2.ppf(q, n_pred_states) for q in quantiles]) # quantile scores

    # Get sim length and set start index 
    N = mc_dic["sim_truth"][0][0].shape[0]
    start_idx = int(N * start_percent)
    mc_trials = mc_dic["trials"]

    # Get array of prediction steps vs chi squared bucket counts
    sim_steps = N-start_idx
    ellipsoid_trial_hits = np.zeros( (sim_steps, qs.size) )
    ellipsoid_hits = np.zeros(qs.size)
    ellipsoid_trial_scores = np.zeros(sim_steps)
    ellipsoid_scores = np.zeros(sim_steps)
    # Monte Carlo Stats
    eIs = np.zeros((mc_trials, sim_steps, n_pred_states))
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
        sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_track])
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

        # Setup convex optimization problem in along track direction
        from_end = 3000 # Only optimize over start_idx to N minus from_end steps
        s = qs[0]
        dt = mc_dic["dt"]
        sigma2 = cp.Variable(1)
        constr_list = [sigma2 >= 0] 
        for idx in range(start_idx+1, N-from_end):
            i = idx - start_idx
            Ptt = Ps_track[i][0,0]
            ett = es_track[i][0]
            constr_list.append(sigma2 >= ett**2/(i*s*dt) - Ptt/(i*dt)) 
        prob = cp.Problem(cp.Minimize(sigma2), constr_list)
        prob.solve()
        # Print result.
        print("\nThe optimal value is", prob.value)
        print("Optimal solution x is")
        print(sigma2.value)
        foobar=2
        sigma2 = sigma2.value
        # Now we can recompute the variance bounds using this additive found bound 
        Ps_track[:,0,0] += np.arange(0,N-start_idx) * dt * sigma2
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
        plt.subplot(3,1,3)
        plt.plot(Ts[start_idx:], es_track[:, 2], 'g--')
        plt.plot(Ts[start_idx:], sig_bound[:, 2], 'm--')
        plt.plot(Ts[start_idx:], -sig_bound[:, 2], 'm--')
        plt.show()
        foobar=2

        Ps = np.array([R.T @ P @ R for R,P in zip(Rs_track,Ps_track)])
        sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps])
        es = xs[start_idx:, :6] - xkf_preds
        plt.figure()
        plt.suptitle("Corrected Predictor Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
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




if __name__ == '__main__':
    #test_along_cross_track_covariance_transformations()
    #test_parametric_fit()
    test_convex_proc_noise_fit()