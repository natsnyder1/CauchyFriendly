from doctest import testfile
import numpy as np 
import ctypes 
import os 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
from scipy.stats import chi2  

class CauchyPoint3(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double), ('z', ctypes.c_double)]

class CauchyPoint2(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double)]

def read_1D_cpdf_binary(dir_name, grid_numel_x, cpdf_fname):
    with open(dir_name + cpdf_fname, 'rb') as file:
        result = []
        p = CauchyPoint2()
        while file.readinto(p) == ctypes.sizeof(p):
            result.append([p.x, p.y])
    result = np.array(result)
    X = result[:,0]
    Y = result[:,1]
    assert(X.size == grid_numel_x)
    return X, Y

def read_2D_cpdf_binary(dir_name, grid_numel_x, grid_numel_y, cpdf_fname):
    with open(dir_name + cpdf_fname, 'rb') as file:
        result = []
        p = CauchyPoint3()
        while file.readinto(p) == ctypes.sizeof(p):
            result.append([p.x, p.y, p.z])
    result = np.array(result)
    X = result[:,0].reshape((grid_numel_y, grid_numel_x))
    Y = result[:,1].reshape((grid_numel_y, grid_numel_x))
    Z = result[:,2].reshape((grid_numel_y, grid_numel_x))
    return X, Y, Z

def read_in_2D_pdfs(dir_name, state_tup = (0,1), cpdf_prefix = "cpdf_"):
    s1 = state_tup[0]
    s2 = state_tup[1]
    str_tup = str(s1) + str(s2)
    if dir_name[-1] != '/':
        dir_name += "/"
    grid_elems_file = "grid_elems_" + str_tup + ".txt"
    cpdf_grid_elems = np.atleast_2d( np.genfromtxt(dir_name + grid_elems_file, delimiter=',').astype(np.int64) )
    Xs = []
    Ys = []
    Zs = []
    N = cpdf_grid_elems.shape[0]
    for i in range(N):
        cge = cpdf_grid_elems[i]
        grid_numel_x, grid_numel_y = cge
        cpdf_fname = "cpdf_" + str_tup + "_" + str(i+1) + ".bin"
        X, Y, Z = read_2D_cpdf_binary(dir_name, grid_numel_x, grid_numel_y, cpdf_fname)
        Xs.append(X)
        Ys.append(Y)
        Zs.append(Z)
    return Xs, Ys, Zs

def read_in_1D_pdfs(dir_name, state_idx = 0, cpdf_prefix = "cpdf_"):
    str_idx = str(state_idx)
    if dir_name[-1] != '/':
        dir_name += "/"
    grid_elems_file = "grid_elems_" + str_idx + ".txt"
    cpdf_grid_elems = np.genfromtxt(dir_name + grid_elems_file, delimiter=',').astype(np.int64)
    Xs = []
    Ys = []
    N = cpdf_grid_elems.size
    for i in range(N):
        grid_numel_x = cpdf_grid_elems[i]
        cpdf_fname = "cpdf_" + str_idx + "_" + str(i+1) + ".bin"
        X, Y = read_1D_cpdf_binary(dir_name, grid_numel_x, cpdf_fname)
        Xs.append(X)
        Ys.append(Y)
    return Xs, Ys

def test_one_state():
    log_1d = os.path.dirname(os.path.abspath(__file__)) + "/bin/one_state/log_1d"
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/bin/one_state/"
    xs0, ys0 = read_in_1D_pdfs(log_1d, 0)
    cond_means = np.loadtxt(log_dir + "cond_means.txt")
    true_states = np.loadtxt(log_dir + "true_states.txt")
    N = len(xs0)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.title("1D Density")
    plt.xlabel("x-axis (State-1)")
    plt.ylabel("z-axis (CPDF Probability)")
    for i in range(N):
        print("Step {}/{}".format(i+1,N))
        x0 = xs0[i]
        y0 = ys0[i]
        ax.plot(x0, y0, 'b')
        foo = plt.scatter(cond_means[i], 0, marker='x', color='b')
        bar = plt.scatter(true_states[i], 0, marker='x', color='r')
        plt.pause(1)
        ax.lines.pop(0)
        foo.remove()
        bar.remove()
    print("Thats all folks!")

def test_two_state():
    log_2d = os.path.dirname(os.path.abspath(__file__)) + "/bin/two_state/log_2d"
    log_1d = os.path.dirname(os.path.abspath(__file__)) + "/bin/two_state/log_1d"
    Xs, Ys, Zs = read_in_2D_pdfs(log_2d)
    xs0, ys0 = read_in_1D_pdfs(log_1d, 0)
    xs1, ys1 = read_in_1D_pdfs(log_1d, 1)
    N = len(xs0)
    for i in range(2,N):
        print("Step {}/{}".format(i+1,N))
        X = Xs[i]
        Y = Ys[i]
        Z = Zs[i]
        x0 = xs0[i]
        y0 = ys0[i]
        x1 = xs1[i]
        y1 = ys1[i]
        fig = plt.figure()
        plt.title("2D Density")
        ax = fig.add_subplot(projection='3d')
        ax.plot_wireframe(X, Y, Z, zorder=2, color='b')
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        plt.figure()
        plt.title("1D Marg Densities (State 1 and State 2)")
        plt.subplot(211)
        plt.plot(x0,y0)
        plt.xlabel("State 1")
        plt.ylabel("CPDF Probability")
        plt.subplot(212)
        plt.plot(x1,y1)
        plt.xlabel("State 2")
        plt.ylabel("CPDF Probability")
        plt.show()
        foobar = 2
    print("Thats all folks!")

def test_three_state():
    log_2d = os.path.dirname(os.path.abspath(__file__)) + "/bin/three_state/log_2d"
    log_1d = os.path.dirname(os.path.abspath(__file__)) + "/bin/three_state/log_1d"
    Xs01, Ys01, Zs01 = read_in_2D_pdfs(log_2d, (0,1))
    Xs02, Ys02, Zs02 = read_in_2D_pdfs(log_2d, (0,2))
    Xs12, Ys12, Zs12 = read_in_2D_pdfs(log_2d, (1,2))
    xs0, ys0 = read_in_1D_pdfs(log_1d, 0)
    xs1, ys1 = read_in_1D_pdfs(log_1d, 1)
    xs2, ys2 = read_in_1D_pdfs(log_1d, 2)
    N = len(xs0)

    for i in range(N):
        print("Step {}/{}".format(i+1,N))
        X01 = Xs01[i]
        Y01 = Ys01[i]
        Z01 = Zs01[i]

        X02 = Xs02[i]
        Y02 = Ys02[i]
        Z02 = Zs02[i]
        
        X12 = Xs12[i]
        Y12 = Ys12[i]
        Z12 = Zs12[i]

        x0 = xs0[i]
        y0 = ys0[i]
        x1 = xs1[i]
        y1 = ys1[i]
        x2 = xs2[i]
        y2 = ys2[i]

        # set up a figure three times as wide as it is tall
        fig1 = plt.figure(figsize = (18,5))
        ax12 = fig1.add_subplot(1,3,1,projection='3d')
        ax13 = fig1.add_subplot(1,3,2,projection='3d')
        ax23 = fig1.add_subplot(1,3,3,projection='3d')
        plt.tight_layout()
        # Marg 2D
        # Marg (0,1)
        ax12.set_title("Marginal of States 1 and 2", pad=-15)
        ax12.plot_wireframe(X01, Y01, Z01, zorder=2, color='b')
        ax12.set_xlabel("x-axis (State-1)")
        ax12.set_ylabel("y-axis (State-2)")
        ax12.set_zlabel("z-axis (CPDF Probability)")
        # Marg (0,2)
        ax13.set_title("Marginal of States 1 and 3", pad=-8)
        ax13.plot_wireframe(X02, Y02, Z02, zorder=2, color='g')
        ax13.set_xlabel("x-axis (State-1)")
        ax13.set_ylabel("y-axis (State-3)")
        ax13.set_zlabel("z-axis (CPDF Probability)")
        # Marg (1,2)
        ax23.set_title("Marginal of States 2 and 3", pad=-8)
        ax23.plot_wireframe(X12, Y12, Z12, zorder=2, color='r')
        ax23.set_xlabel("x-axis (State-2)")
        ax23.set_ylabel("y-axis (State-3)")
        ax23.set_zlabel("z-axis (CPDF Probability)")

        # set up a figure three times as wide as it is tall
        fig2 = plt.figure(figsize = (18,4))
        ax1 = fig2.add_subplot(1,3,1)
        ax2 = fig2.add_subplot(1,3,2)
        ax3 = fig2.add_subplot(1,3,3)
        # Marg 1D
        # Marg 1
        ax1.set_title("1D Marg of State 1")
        ax1.plot(x0,y0)
        ax1.set_xlabel("State 1")
        ax1.set_ylabel("CPDF Probability")
        # Marg 2
        ax2.set_title("1D Marg of State 2")
        ax2.plot(x1,y1)
        ax2.set_xlabel("State 2")
        ax2.set_ylabel("CPDF Probability")
        # # Marg 3
        ax3.set_title("1D Marg of State 3")
        ax3.plot(x2,y2)
        ax3.set_xlabel("State 3")
        ax3.set_ylabel("CPDF Probability")
        plt.show()
        plt.close()
        #foobar = 2
        
        
    print("Thats all folks!")

def get_2d_confidence_ellipse(x, P, quantile):
    assert (quantile > 0) and (quantile < 1)
    s2 = chi2.ppf(quantile, 2) # s3 is the value for which e^T @ P_2DProj^-1 @ e == s2 
    t1s = np.atleast_2d( np.array([ np.sin(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    t2s = np.atleast_2d( np.array([ np.cos(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    unit_circle = np.hstack((t1s,t2s))
    D, U = np.linalg.eig(P)
    E = U @ np.diag(D * s2)**0.5 # Ellipse is the matrix square root of covariance
    ell_points = (E @ unit_circle.T).T + x 
    return ell_points

def test_relative_cpdf_of_2d_systems():
    rsys_log = os.path.dirname(os.path.abspath(__file__)) + "/bin/rel_cpdf/2d/temp_realiz"
    sys1_log = rsys_log + "/sys1"
    sys2_log = rsys_log + "/sys2"
    
    
    # Simulation components of System 1
    sys1_Xs, sys1_Ys, sys1_Zs = read_in_2D_pdfs(sys1_log) # CPDF
    sys1_xtrues = np.genfromtxt(sys1_log + "/true_states.txt")
    sys1_xhats = np.genfromtxt(sys1_log + "/cond_means.txt")
    sys1_Phats = np.genfromtxt(sys1_log + "/cond_covars.txt")

    # Simulation components of System 2
    sys2_Xs, sys2_Ys, sys2_Zs = read_in_2D_pdfs(sys2_log) # CPDF
    sys2_xtrues = np.genfromtxt(sys2_log + "/true_states.txt")
    sys2_xhats = np.genfromtxt(sys2_log + "/cond_means.txt")
    sys2_Phats = np.genfromtxt(sys2_log + "/cond_covars.txt")

    # Simulation components of Relative System = Trel @ ( System 2 - System 1 )
    rsys_Xs, rsys_Ys, rsys_Zs = read_in_2D_pdfs(rsys_log) # CPDF
    rsys_xhats = np.genfromtxt(rsys_log + "/cond_means.txt")
    rsys_Phats = np.genfromtxt(rsys_log + "/cond_covars.txt")

    # Form True Relative and Transformed State History
    Trel = np.genfromtxt(rsys_log + "/reltrans.txt")
    rsys_xtrues = ( Trel @ ( sys2_xtrues - sys1_xtrues ).T ).T
    
    N = len(rsys_Xs)
    quantile = 0.70
    sys1_Phats = sys1_Phats.reshape((N, 2, 2))
    sys2_Phats = sys2_Phats.reshape((N, 2, 2))
    rsys_Phats = rsys_Phats.reshape((N, 2, 2))
    for i in range(N):
        print("Step {}/{}".format(i+1,N))
        #fig = plt.figure()
        #plt.title("2D Density: Sys1=blue, Sys2=green, Rel=red")
        #ax = fig.add_subplot(projection='3d')
        #ax.plot_wireframe(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], color='b')
        #ax.plot_wireframe(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], color='g')
        #ax.plot_wireframe(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], color='r')
        #ax.set_xlabel("x-axis (State-1)")
        #ax.set_ylabel("y-axis (State-2)")
        #ax.set_zlabel("z-axis (CPDF Probability)")
        z_low = -.2
        sys1_Zs[i][sys1_Zs[i] < 1e-3] = np.nan
        sys2_Zs[i][sys2_Zs[i] < 1e-3] = np.nan
        rsys_Zs[i][rsys_Zs[i] < 1e-3] = np.nan
        GRID_HEIGHT = 8
        GRID_WIDTH = 2

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')
        ax.set_title("System 1, Step {}\nTop Plot 3D CPDF, Bottom Plot 2D CPDF Contour Map".format(i+1))
        #ax.plot_wireframe(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], color='b')
        ax.plot_surface(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], cmap='coolwarm', alpha=0.9)
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        sys1_conf_ellipse = get_2d_confidence_ellipse(sys1_xhats[i], sys1_Phats[i], quantile)
        ax.plot(sys1_conf_ellipse[:,0], sys1_conf_ellipse[:,1], z_low, color='b')
        ax.plot(sys1_xhats[i][0], sys1_xhats[i][1], z_low, color='b', marker='*')
        ax.plot(sys1_xtrues[i][0], sys1_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        sys1_contour = ax2.contour(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i])
        ax2.plot(sys1_xhats[i][0], sys1_xhats[i][1], color='b', marker='*', label="Sys1 State Est.")
        ax2.plot(sys1_xtrues[i][0], sys1_xtrues[i][1], color='m', marker='*', label="Sys1 True State")
        ax2.plot(sys1_conf_ellipse[:,0], sys1_conf_ellipse[:,1], color='b', label="Sys1 {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        #fig.colorbar(sys1_contour, ax=ax2, shrink=0.5, aspect = 5)

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')
        ax.set_title("System 2, Step {}\nTop Plot 3D CPDF, Bottom Plot 2D CPDF Contour Map".format(i+1))
        ax.plot_surface(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], cmap='coolwarm', alpha=0.9)
        #ax.plot_wireframe(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], color='g')
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        sys2_conf_ellipse = get_2d_confidence_ellipse(sys2_xhats[i], sys2_Phats[i], quantile)
        ax.plot(sys2_conf_ellipse[:,0], sys2_conf_ellipse[:,1], z_low, color='g')
        ax.plot(sys2_xhats[i][0], sys2_xhats[i][1], z_low, color='g', marker='*')
        ax.plot(sys2_xtrues[i][0], sys2_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        sys2_contour = ax2.contour(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i])
        ax2.plot(sys2_xhats[i][0], sys2_xhats[i][1], color='g', marker='*', label="Sys2 State Est.")
        ax2.plot(sys2_xtrues[i][0], sys2_xtrues[i][1], color='m', marker='*', label="Sys2 True State")
        ax2.plot(sys2_conf_ellipse[:,0], sys2_conf_ellipse[:,1], color='g', label="Sys2 {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        #fig.colorbar(sys2_contour, ax=ax2, shrink=0.5, aspect = 5)

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')        
        ax.set_title("Relative System: RSys=Sys2-Sys1, Step {}\nTop Plot 3D CPDF, Bottom Plot 2D CPDF Contour Map".format(i+1))
        ax.plot_surface(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], cmap='coolwarm', alpha=0.9)
        #ax.plot_wireframe(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], color='r')
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        rsys_conf_ellipse = get_2d_confidence_ellipse(rsys_xhats[i], rsys_Phats[i], quantile)
        ax.plot(rsys_conf_ellipse[:,0], rsys_conf_ellipse[:,1], z_low, color='r')
        ax.plot(rsys_xhats[i][0], rsys_xhats[i][1], z_low, color='r', marker='*')
        ax.plot(rsys_xtrues[i][0], rsys_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        rsys_contour = ax2.contour(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i])
        ax2.plot(rsys_xhats[i][0], rsys_xhats[i][1], color='r', marker='*', label="RSys State Est.")
        ax2.plot(rsys_xtrues[i][0], rsys_xtrues[i][1], color='m', marker='*', label="RSys True State")
        ax2.plot(rsys_conf_ellipse[:,0], rsys_conf_ellipse[:,1], color='r', label="RSys {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        # Show
        plt.tight_layout()
        plt.show()
        foobar = 2
    print("Thats all folks!")

def test_relative_cpdf_of_3d_systems():
    rsys_log = os.path.dirname(os.path.abspath(__file__)) + "/bin/rel_cpdf/3d/rel_trans_unit" #rel_trans_non_unit"
    sys1_log = rsys_log + "/sys1"
    sys2_log = rsys_log + "/sys2"
    indices = np.array([0,1,2])
    index_mask = np.array([True, True, False]) # 2D Marginal States
    assert(np.sum(index_mask) == 2)
    cov_mask = np.outer(index_mask, index_mask)
    marg_idxs = indices[index_mask] #If you change this, then you will need to follow marg_idxs through the script
    

    # Simulation components of System 1
    sys1_Xs, sys1_Ys, sys1_Zs = read_in_2D_pdfs(sys1_log) # CPDF
    sys1_xtrues = np.genfromtxt(sys1_log + "/true_states.txt")
    sys1_xhats = np.genfromtxt(sys1_log + "/cond_means.txt")
    sys1_Phats = np.genfromtxt(sys1_log + "/cond_covars.txt")

    # Simulation components of System 2
    sys2_Xs, sys2_Ys, sys2_Zs = read_in_2D_pdfs(sys2_log) # CPDF
    sys2_xtrues = np.genfromtxt(sys2_log + "/true_states.txt")
    sys2_xhats = np.genfromtxt(sys2_log + "/cond_means.txt")
    sys2_Phats = np.genfromtxt(sys2_log + "/cond_covars.txt")

    # Simulation components of Relative System = Trel @ ( System 2 - System 1 )
    rsys_Xs, rsys_Ys, rsys_Zs = read_in_2D_pdfs(rsys_log) # CPDF
    rsys_xhats = np.genfromtxt(rsys_log + "/cond_means.txt")
    rsys_Phats = np.genfromtxt(rsys_log + "/cond_covars.txt")

    # Form True Relative and Transformed State History
    Trel = np.genfromtxt(rsys_log + "/reltrans.txt")
    rsys_xtrues = ( Trel @ ( sys2_xtrues - sys1_xtrues ).T ).T
    
    N = len(rsys_Xs)
    n = sys1_xhats.shape[1]
    quantile = 0.70
    sys1_xtrues = sys1_xtrues[:,marg_idxs]
    sys2_xtrues = sys2_xtrues[:,marg_idxs]
    sys1_xhats = sys1_xhats[:,marg_idxs]
    sys2_xhats = sys2_xhats[:,marg_idxs]
    sys1_Phats = np.array([ P[cov_mask].reshape((2,2)) for P in sys1_Phats.reshape((N, n, n))])
    sys2_Phats = np.array([ P[cov_mask].reshape((2,2)) for P in sys2_Phats.reshape((N, n, n))])
    rsys_Phats = rsys_Phats.reshape((N, 2, 2))
    for i in range(N):
        print("Step {}/{}".format(i+1,N))
        z_low = -.2
        num_levels = 11
        sys1_Zs[i][sys1_Zs[i] < 1e-3] = np.nan
        sys2_Zs[i][sys2_Zs[i] < 1e-3] = np.nan
        rsys_Zs[i][rsys_Zs[i] < 1e-3] = np.nan
        GRID_HEIGHT = 8
        GRID_WIDTH = 2

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')
        ax.set_title("3D System 1 (Marginal States {},{}), Step {}\nMarginal CPDF Plot, Bottom Plot Marginal 2D CPDF Contour Map".format(marg_idxs[0]+1,marg_idxs[1]+1,i+1))
        #ax.plot_wireframe(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], color='b')
        ax.plot_surface(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], cmap='coolwarm', alpha=0.95)
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        sys1_conf_ellipse = get_2d_confidence_ellipse(sys1_xhats[i], sys1_Phats[i], quantile)
        ax.plot(sys1_conf_ellipse[:,0], sys1_conf_ellipse[:,1], z_low, color='b')
        ax.plot(sys1_xhats[i][0], sys1_xhats[i][1], z_low, color='b', marker='*')
        ax.plot(sys1_xtrues[i][0], sys1_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        sys1_contour = ax2.contour(sys1_Xs[i], sys1_Ys[i], sys1_Zs[i], levels=num_levels)
        ax2.plot(sys1_xhats[i][0], sys1_xhats[i][1], color='b', marker='*', label="Sys1 State Est.")
        ax2.plot(sys1_xtrues[i][0], sys1_xtrues[i][1], color='m', marker='*', label="Sys1 True State")
        ax2.plot(sys1_conf_ellipse[:,0], sys1_conf_ellipse[:,1], color='b', label="Sys1 {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        #fig.colorbar(sys1_contour, ax=ax2, shrink=0.5, aspect = 5)

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')
        ax.set_title("3D System 2 (Marginal States {},{}), Step {}\nMarginal CPDF Plot, Bottom Plot Marginal 2D CPDF Contour Map".format(marg_idxs[0]+1,marg_idxs[1]+1,i+1))
        ax.plot_surface(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], cmap='coolwarm', alpha=0.95)
        #ax.plot_wireframe(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], color='g')
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        sys2_conf_ellipse = get_2d_confidence_ellipse(sys2_xhats[i], sys2_Phats[i], quantile)
        ax.plot(sys2_conf_ellipse[:,0], sys2_conf_ellipse[:,1], z_low, color='g')
        ax.plot(sys2_xhats[i][0], sys2_xhats[i][1], z_low, color='g', marker='*')
        ax.plot(sys2_xtrues[i][0], sys2_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        sys2_contour = ax2.contour(sys2_Xs[i], sys2_Ys[i], sys2_Zs[i], levels=num_levels)
        ax2.plot(sys2_xhats[i][0], sys2_xhats[i][1], color='g', marker='*', label="Sys2 State Est.")
        ax2.plot(sys2_xtrues[i][0], sys2_xtrues[i][1], color='m', marker='*', label="Sys2 True State")
        ax2.plot(sys2_conf_ellipse[:,0], sys2_conf_ellipse[:,1], color='g', label="Sys2 {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        #fig.colorbar(sys2_contour, ax=ax2, shrink=0.5, aspect = 5)

        fig = plt.figure(figsize=(6,12))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-2,:],projection='3d')        
        ax.set_title("Relative System: RSys = Trel*(Sys2-Sys1), Step {}, \nTrel=[{},{},{} ; {}, {}, {}]\nTop Plot 3D CPDF of RSys, Bottom Plot 2D CPDF Contour Map".format(i+1, Trel[0,0], Trel[0,1], Trel[0,2], Trel[1,0], Trel[1,1], Trel[1,2]) )
        ax.plot_surface(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], cmap='coolwarm', alpha=0.95)
        #ax.plot_wireframe(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], color='r')
        ax.set_xlabel("x-axis (State-1)")
        ax.set_ylabel("y-axis (State-2)")
        ax.set_zlabel("z-axis (CPDF Probability)")
        rsys_conf_ellipse = get_2d_confidence_ellipse(rsys_xhats[i], rsys_Phats[i], quantile)
        ax.plot(rsys_conf_ellipse[:,0], rsys_conf_ellipse[:,1], z_low, color='r')
        ax.plot(rsys_xhats[i][0], rsys_xhats[i][1], z_low, color='r', marker='*')
        ax.plot(rsys_xtrues[i][0], rsys_xtrues[i][1], z_low, color='m', marker='*')
        ax2 = plt.subplot(gs[(GRID_HEIGHT-2):,:])#2,1,2)
        rsys_contour = ax2.contour(rsys_Xs[i], rsys_Ys[i], rsys_Zs[i], levels=num_levels)
        ax2.plot(rsys_xhats[i][0], rsys_xhats[i][1], color='r', marker='*', label="RSys State Est.")
        ax2.plot(rsys_xtrues[i][0], rsys_xtrues[i][1], color='m', marker='*', label="RSys True State")
        ax2.plot(rsys_conf_ellipse[:,0], rsys_conf_ellipse[:,1], color='r', label="RSys {}% Confidence Ellipse".format(quantile*100))
        ax2.set_xlabel("State 1")
        ax2.set_ylabel("State 2")
        leg = ax2.legend()
        leg.set_draggable(state=True)
        # Show
        plt.tight_layout()
        plt.show()
        foobar = 2
    print("Thats all folks!")



if __name__ == "__main__":
    #test_one_state()
    #test_two_state()
    #test_three_state()
    #test_relative_cpdf_of_2d_systems()
    test_relative_cpdf_of_3d_systems()
    