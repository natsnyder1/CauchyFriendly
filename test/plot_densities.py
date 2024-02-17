import numpy as np 
import ctypes 
import os 
import matplotlib.pyplot as plt 

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

if __name__ == "__main__":
    #test_one_state()
    #test_two_state()
    test_three_state()
    