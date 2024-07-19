import numpy as np 
import matplotlib 
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt 
import sys, os
import pickle 
file_dir = os.path.dirname(os.path.abspath(__file__))

# Note: Fit really should not be more than 1/2 of an orbit
# Y: Set of points observed: N x 3
# T: Set of times (seconds) points are observed at (relative to first point in OD)
# fit_order: Order of the polynomial
# with_plot: Shows plot of fit
# returns: coefficients -> [a_0, a_1, ..., a_order] for a_0 + a_1*t + a_2*t**2 + ... + a_order*t**order
def poly_fit(Ys, Ts, fit_order, with_plot=True):
    ts = Ts.copy() - Ts[0]
    ts /= ts[-1]
    X = np.zeros((ts.size, fit_order+1))
    for i in range(fit_order+1):
        X[:,i] = ts**i
    #if X.shape[0] == X.shape[1]:
    thetas = np.linalg.solve(X.T @ X, X.T @ Ys)
    #else:
    #    thetas,_,_,_ = np.linalg.lstsq(X, Ys)
    Yest = X @ thetas 
    print("Max Residual is: ", np.max( np.sum((Ys-Yest)**2,axis=1) ) )
    if with_plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Ys[:,0], Ys[:,1], Ys[:,2], color='r')
        ax.plot(Yest[:,0], Yest[:,1], Yest[:,2], color='r')
        plt.show()
    return thetas, X

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

def poly_deriv(thetas, Xs):
    assert thetas.shape[0] == Xs.shape[1]
    n = thetas.shape[0]
    scale = np.arange(1,n).reshape((n-1,1))
    dthetas = thetas[1:,:] * scale 
    dXs = Xs.copy()[:,:-1]
    return dthetas, dXs

def poly_mul(thetas1, thetas2):
    assert thetas1.ndim == 1
    assert thetas2.ndim == 1
    n = thetas1.size
    m = thetas2.size
    # for each of the d-polynomials, expand the coefficients
    new_thetas = np.zeros(n+m-1)
    for i in range(n):
        for j in range(m):
            new_thetas[i+j] += thetas1[i]*thetas2[j]
    return new_thetas

def poly_time_matrix(ts, poly_order_plus_1):
    N = ts.size
    n = poly_order_plus_1
    Xs = np.zeros((N, n))
    for i in range(poly_order_plus_1):
        Xs[:,i] = ts**i
    return Xs

# returns the polynomial expression for thetas1.T @ thetas2
# thetas1 and thetas2 have shape (n+1) x d, 
# where n is the polynomial order (plus its constant bias) and d is the number of polynomials to be inner producted over
def poly_inner_prod(thetas1, thetas2):
    assert thetas1.shape[1] == thetas2.shape[1]
    n,d = thetas1.shape
    m,d = thetas2.shape 
    new_thetas = np.zeros(n+m-1)
    for i in range(d):
        new_thetas += poly_mul(thetas1[:,i], thetas2[:,i])
    return new_thetas

def test_poly_deriv():
    f = lambda ts : 1 + 2*ts + 4*ts**2 - 0.5*ts**3
    df = lambda ts : 2 + 8*ts - 1.5*ts**2
    ddf = lambda ts : 8 - 3*ts
    ts = np.linspace(0,1,50)

    plt.scatter(ts, f(ts), color='r')
    plt.scatter(ts, df(ts), color='b')
    plt.scatter(ts, ddf(ts), color='g')
    Xs = np.zeros((ts.size, 4))
    thetas = np.array([[1,2,4,-0.5]]).T
    for i in range(4):
        Xs[:,i] = ts**i
    dthetas, dXs = poly_deriv(thetas, Xs)
    ddthetas, ddXs = poly_deriv(dthetas, dXs)
    yp = Xs @ thetas
    dyp = dXs @ dthetas
    ddyp = ddXs @ ddthetas
    plt.plot(ts, yp, 'r')
    plt.plot(ts, dyp, 'b')
    plt.plot(ts, ddyp, 'g')
    plt.show()
    foobar = 2 # looks good
    
def test_fits():
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
    Ys = xs[30:85, 0:3]
    Ts = np.arange(Ys.shape[0]) * (1.0*mc_dic["dt"])
    thetas = poly_fit(Ys,Ts,fit_order, with_plot=True)
    
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

if __name__ == "__main__":
    #test_fits()
    test_poly_deriv()