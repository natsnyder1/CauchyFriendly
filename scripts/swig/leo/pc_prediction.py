import numpy as np 
import datetime 
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg',force=True)
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from datetime import datetime 
from datetime import timedelta 
from scipy.stats import chi2  
import copy 
import gmat_sat as gsat 


# Fancy Arrow Patch for MPL
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

def _create_unit_sphere():
    _u = np.linspace(0, 2 * np.pi, 50)
    _v = np.linspace(0, np.pi, 50)
    _x = np.outer(np.cos(_u), np.sin(_v))
    _y = np.outer(np.sin(_u), np.sin(_v))
    _z = np.outer(np.ones_like(_u), np.cos(_v))
    unitsphere = np.stack((_x, _y, _z), axis=-1)[..., None]
    return unitsphere

# Returns transformation as [along^T;cross^T,radial^T]
def get_along_cross_radial_transformation(x):
    # position and velocity 3-vector components
    rh = x[0:3]
    vh = x[3:6]
    rhn = np.linalg.norm(rh)
    vhn = np.linalg.norm(vh)
    # Radial Direction -- direction of position vector
    ur = rh / rhn # z-axis
    # Cross Track -- in the direction of the angular momentum vector (P cross V)
    uc = np.cross(rh, vh) # y-axis - cross track direction is radial direction cross_prod along track direction
    uc /= np.linalg.norm(uc)
    # Along Track -- will be coincident with the velocity vector for a perfectly circular orbit.    
    ua = np.cross(uc,ur)
    ua /= np.linalg.norm(ua)
    # Along, Cross, Radial
    R = np.vstack( (ua,uc,ur) )
    return R

def _cubic_poly_root_find(poly_fd, dfd_dt, d2fd_dt2):
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
            if (a1 + 2*a2*real_pr + 3*a3*real_pr**2) > 0:
                if good_root is None:
                    good_root = [real_pr]
                else:
                    good_root.append(real_pr)
    if type(good_root) == list:
        br = good_root[0] #best root
        br_val = poly_fd @ np.array([1,br,br**2,br**3])
        for i in range(1,len(good_root)):
            gr = good_root[i]
            gr_val = poly_fd @ np.array([1,gr,gr**2,gr**3])
            if br_val > gr_val:
                br_val = gr_val 
                br = gr 
        return br
    else:
        return None

def _cubic_poly(fd, dfd_dt):
    a0 = fd[0]
    a1 = dfd_dt[0]
    a2,a3 = np.linalg.inv([[1,1],[2,3]]) @ np.array([fd[1] - a0 - a1, dfd_dt[1] - a1])
    return np.array([a0,a1,a2,a3])

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
    dcord = 0
    for i in range(1,len(poly_coefs)):
        dcord += troot**(i-1) * poly_coefs[i] * i
    return cord, dcord

# Method from alfanso book
def closest_approach_info(tks, prim_tup, sec_tup):
    # Unpack the projected position, velocity and acceleration of the primary satellite
    p_pks, p_vks, p_aks = prim_tup 
    # Unpack the projected position, velocity and acceleration of the secondary satellite
    s_pks, s_vks, s_aks = sec_tup
    # Check sizings
    assert(tks.size == p_pks.shape[0] == s_pks.shape[0])
    assert(p_pks.shape == s_pks.shape)
    assert(p_vks.shape == s_vks.shape)
    assert(p_aks.shape == s_aks.shape)
    assert(s_pks.shape == p_vks.shape == p_aks.shape)
    assert(p_pks.shape[0] >  p_pks.shape[1])
    assert(p_vks.shape[0] >  p_vks.shape[1])
    assert(p_aks.shape[0] >  p_aks.shape[1])
    # Find the index of closest position between the two vehicles
    i_star = np.argmin( np.sum( (s_pks - p_pks)**2, axis=1) ) # eq 13

    # Return 
    # 1.) closest approach start index "i"
    # 2.) closest approach time localized on interval [i,i+1] in range [0,1]
    # 3.) closest approach time "t_c" in seconds
    # 4.) position of primary at closest approach "pp_c"
    # 5.) velocity of primary at closest approach "pv_c"
    # 6.) position of secondary at closest approach "sp_c"
    # 7.) velocity of secondary at closest approach "sv_c"

    if i_star == 0:
        return i_star, 0, tks[i_star], p_pks[i_star], p_vks[i_star], s_pks[i_star], s_vks[i_star],
    elif i_star == tks.size-1:
        return i_star, 1, tks[i_star], p_pks[i_star], p_vks[i_star], s_pks[i_star], s_vks[i_star],
    else:
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
        poly_fd = _cubic_poly(fd[0:2], dfd_dt[0:2])
        troot = _cubic_poly_root_find(poly_fd, dfd_dt[0:2], d2fd_dt2[0:2])
        # Repeat check for real positive roots between (0,1) on second interval -- eqns 20-23
        if troot is None:
            poly_fd = _cubic_poly(fd[1:3], dfd_dt[1:3])
            troot = _cubic_poly_root_find(poly_fd, dfd_dt[1:3], d2fd_dt2[1:3])
            # Error if nothing is found still
            if troot is None:
                print("ERROR PLZ DEBUG -- NO ROOT FOUND!")
                exit(1)
            else:
                # This is the closest approach time
                t_c = tks[idxs[1]] + (tks[idxs[2]]-tks[idxs[1]]) * troot #-- eqn 24
                second_interval = True
        else:
            # This is the closest approach time
            t_c = tks[idxs[0]] + (tks[idxs[1]] - tks[idxs[0]]) * troot #-- eqn 24
        
        # Now, find the point for the primary satellite at closest approach -- eqns 25-29
        i = idxs[0] + second_interval
        pp_c = np.zeros(3)
        pv_c = np.zeros(3)
        for j in range(3):
            pp_c[j], pv_c[j] = _quintic_poly_fit_and_eval(troot, p_pks[i:i+2,j], p_vks[i:i+2,j], p_aks[i:i+2,j])
        # Now, find the point for the secondary satellite at closest approach -- eqns 25-29
        sp_c = np.zeros(3)
        sv_c = np.zeros(3)
        for j in range(3):
            sp_c[j], sv_c[j] = _quintic_poly_fit_and_eval(troot, s_pks[i:i+2,j], s_vks[i:i+2,j], s_aks[i:i+2,j])

        # Return 
        # 1.) closest approach start index "i"
        # 2.) closest approach time localized on interval [i,i+1] in range [0,1]
        # 3.) closest approach time "t_c" in seconds
        # 4.) position of primary at closest approach "pp_c"
        # 5.) velocity of primary at closest approach "pv_c"
        # 6.) position of secondary at closest approach "sp_c"
        # 7.) velocity of secondary at closest approach "sv_c"
        return i, troot, t_c, pp_c, pv_c, sp_c, sv_c

# This seems to work really nicely, converging to the limit of gmats step
def iterative_time_closest_approach(dt, _t0, prim_tup, sec_tup, start_idx = 0, its = -1, with_plot=True):
    # For now this function assumes an integer time step
    assert(int(dt) == dt)
    assert(dt > 0)
    # Initial iteration
    p_pks,p_vks,p_aks = copy.deepcopy(prim_tup)
    s_pks,s_vks,s_aks = copy.deepcopy(sec_tup)
    t0 = copy.deepcopy(_t0)
    tks = np.arange(p_pks.shape[0]) * dt
    # Find the index of closest position between the two vehicles
    i = np.argmin( np.sum( (s_pks[start_idx:,:] - p_pks[start_idx:,:])**2, axis=1) ) # eq 13
    i += start_idx
    if (i == start_idx) or (i == s_pks[start_idx:,:].shape[0]):
        print("Index range is bad! Use the norm plotter to select a more appropriate index range!")
        return None,None,None,None,None,None,None
    
    # Idea is just invoke propagator smartly until last round. FIX PAST HERE
    
    # Left hand side of interval
    _dt = dt
    i_star_lhs = i
    t_lhs = tks[i]
    i -= 1 # start here and go two in front at new dt

    if with_plot:
        fig = plt.figure() 
        ax = fig.gca(projection='3d')
        plt.title("LEO Primary (red) vs. Secondary (blue) trajectory over time")
        ax.scatter(p_pks[start_idx:i+2,0], p_pks[start_idx:i+2,1], p_pks[start_idx:i+2,2], color = 'r')
        ax.scatter(s_pks[start_idx:i+2,0], s_pks[start_idx:i+2,1], s_pks[start_idx:i+2,2], color = 'b')
        ax.set_xlabel("x-axis (km)")
        ax.set_ylabel("y-axis (km)")
        ax.set_zlabel("z-axis (km)")
        # Plot relative difference in position
        fig2 = plt.figure()
        plt.title("Norm of Position Difference between Primary and Secondary")
        plt.plot(tks[start_idx:i+2], np.linalg.norm(p_pks[start_idx:i+2]-s_pks[start_idx:i+2], axis=1))
        plt.xlabel("Time (sec)")
        plt.ylabel("2-norm of position difference (km)")
        plt.show()
        foobar=5

    
    dts = [10,1,0.1,0.01,0.001]
    substeps = [int(2*dt/dts[0]),20,20,20,20]
    it_idx = 0 
    while dts[it_idx] > dt:
        it_idx += 1
    assert (dt % dts[it_idx]) == 0
    its = 5 if its == -1 else its 
    for it in range(it_idx, its):
        x0_prim = np.concatenate( (p_pks[i],p_vks[i]) )
        x0_sec  = np.concatenate( (s_pks[i],s_vks[i]) )
        dt = dts[it]
        t0 = t0 + timedelta( seconds = tks[i] )
        tks = tks[i] + np.arange(substeps[it]+1) * dt
        
        # Now run the primary over the subinterval i to i+1
        fermiSat = gsat.FermiSatelliteModel(t0, x0_prim, dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        p_pks = [] # Primary Pos 3-Vec
        p_vks = [] # Primary Vel 3-Vec
        p_aks = [] # Primary Acc 3-Vec
        # Propagate Primary and Store
        xk = x0_prim.copy()
        for i in range(substeps[it]+1):
            dxk_dt = fermiSat.get_state6_derivatives() 
            p_pks.append(xk[0:3])
            p_vks.append(xk[3:6])
            p_aks.append(dxk_dt[3:6])
            xk = fermiSat.step()
        p_pks = np.array(p_pks)
        p_vks = np.array(p_vks)
        p_aks = np.array(p_aks)
        fermiSat.clear_model()

        # Create Satellite Model for Secondary
        fermiSat = gsat.FermiSatelliteModel(t0, x0_sec, dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        s_pks = [] # Secondary Pos 3-Vec
        s_vks = [] # Secondary Vel 3-Vec
        s_aks = [] # Secondary Acc 3-Vec
        # Propagate Secondary and Store
        xk = x0_sec.copy()
        for i in range(substeps[it]+1):
            dxk_dt = fermiSat.get_state6_derivatives() 
            s_pks.append(xk[0:3])
            s_vks.append(xk[3:6])
            s_aks.append(dxk_dt[3:6])
            xk = fermiSat.step()
        s_pks = np.array(s_pks)
        s_vks = np.array(s_vks)
        s_aks = np.array(s_aks)
        fermiSat.clear_model()

        if it < 4:
            i = np.argmin( np.sum( (s_pks - p_pks)**2, axis=1) ) - 1
        else:
            # Now re-run the closest approach routine
            i, troot, t_c, pp_c, pv_c, sp_c, sv_c = closest_approach_info(tks, (p_pks,p_vks,p_aks), (s_pks,s_vks,s_aks))

            print("Iteration: ", it + 1, " (timestamp)")
            print("Step dt: ", dt, "(sec)")
            print("Tc: {} (sec), Idx of Tc: {}".format(t_c, i) )
            print("Primary at Tc: ", pp_c, "(km)")
            print("Secondary at Tc: ", sp_c, "(km)")
            print("Pos Diff is: ", pp_c-sp_c, "(km)")
            print("Pos Norm is: ", 1000*np.linalg.norm(pp_c-sp_c), "(m)")

        if with_plot:
            # Black points give found interval, # green point are +/- 1 buffers
            fig = plt.figure() 
            ax = fig.gca(projection='3d')
            plt.title("Leo Trajectory over Time")
            ax.scatter(p_pks[:i+2,0], p_pks[:i+2,1], p_pks[:i+2,2], color = 'r')
            ax.scatter(s_pks[:i+2,0], s_pks[:i+2,1], s_pks[:i+2,2], color = 'b')
            ax.scatter(p_pks[i,0], p_pks[i,1], p_pks[i,2], color = 'k')
            ax.scatter(p_pks[i+1,0], p_pks[i+1,1], p_pks[i+1,2], color = 'k')
            ax.scatter(s_pks[i,0], s_pks[i,1], s_pks[i,2], color = 'k')
            ax.scatter(s_pks[i+1,0], s_pks[i+1,1], s_pks[i+1,2], color = 'k')
            ax.set_xlabel("x-axis (km)")
            ax.set_ylabel("y-axis (km)")
            ax.set_zlabel("z-axis (km)")
            
            dist_norm = np.linalg.norm(p_pks[:i+5]-s_pks[:i+5], axis=1)
            fig2 = plt.figure()
            plt.title("Norm of Position Difference between Primary and Secondary")
            plt.plot(tks[:i+5], dist_norm)
            plt.scatter(tks[i-1], dist_norm[i-1], color='g')
            plt.scatter(tks[i], dist_norm[i], color='k')
            plt.scatter(tks[i+1], dist_norm[i+1], color='k')
            plt.scatter(tks[i+2], dist_norm[i+2], color='g')
            plt.xlabel("Time (sec)")
            plt.ylabel("2-norm of position difference (km)")
            plt.show()
            foobar=3
    
    if( t_c < t_lhs ):
        t_lhs -= _dt 
        i_star_lhs -= 1
    # i_star_lhs -> The nominal propagation index to stop at 
    # t_lhs -> The time at the nominal propagation index 
    # t_c -> The final time of closest approach 
    # pp_c -> The position of the primary at closest approach 
    # pv_c -> The velocity of the primary at closest approach 
    # sp_c -> The position of the secondary at closest approach
    # sv_c -> The velocity of the secondary at closest approach 
    return i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c

def draw_3d_encounter_plane(s_xtc, p_xtc, s_Ptc, p_Ptc, 
    scales = (0.0,0.0,0.0), 
    plot_var_3D=False, plot_var_2D=True, indiv_var=True, 
    mc_runs_prim = None, mc_runs_sec = None, 
    s_mce_Ptc = None, p_mce_Ptc = None):
    
    #scales = (0.0,0.000001,0.0) # for plot_var_3D
    fig = plt.figure() 
    ax = fig.gca(projection='3d') #fig.add_subplot(projection='3d')
    ax.set_title("Primary and Secondary at Time of Closest Approach:")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    p1 = p_xtc[0:3] # position vector of primary 
    v1 = p_xtc[3:6] # velocity vector of primary
    P1 = p_Ptc[0:3,0:3] # Covariance of primary
    p2 = s_xtc[0:3] # position vector of secondary 
    v2 = s_xtc[3:6] # velocity vector of secondary
    P2 = s_Ptc[0:3,0:3] # Covariance of secondary

    rp = p2 - p1 # relative position vector at tc 
    rv = v2 - v1 # relative velocity vector at tc 
    rP = P1 + P2 # relative position uncertainty
    
    # Draw Primary and its velocity vector direction
    ax.scatter(p1[0], p1[1], p1[2], s=2, color = 'r')
    suv1 = ( v1 / np.linalg.norm(v1) ) * np.min(np.abs(rp)) * 0.5 # scaled unit vector of primary
    ax.arrow3D(p1[0], p1[1], p1[2], suv1[0], suv1[1], suv1[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='r', label='PrimVelVec')

    # Draw Secondary and its velocity vector direction
    ax.scatter(p2[0], p2[1], p2[2], s=2, color = 'b')
    suv2 = ( v2 / np.linalg.norm(v2) ) * np.min(np.abs(rp)) * 0.5 # scaled unit vector of secondary
    ax.arrow3D(p2[0], p2[1], p2[2], suv2[0], suv2[1], suv2[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='b', label='SecVelVec')
    
    # Draw Encounter Plane is defined by (ep_c, ep_a)
    # ep_c is the center of the encounter plane (average of prim and sec positions)
    # ep_a is the normal of the hyperplane that defines the encounter plane (rel vel vec)
    # ep_b = ep_a^T @ ep_c is the offset for the hyperplane equation: ep_a^T @ (p - ep_c) == 0 --> ep_a^T @ p == ep_b
    ep_c = (p1 + p2) / 2.0
    ep_a = rv.copy()
    ep_b = ep_a @ ep_c

    ax.scatter(ep_c[0], ep_c[1], ep_c[2], s=80, color = 'g', marker = "*", label= 'Encounter Plane')
    suv3 = ( ep_a / np.linalg.norm(ep_a) ) * np.min(np.abs(rp)) * 0.5 # scaled unit vector of encounter plane
    ax.arrow3D(ep_c[0], ep_c[1], ep_c[2], suv3[0], suv3[1], suv3[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='g', label='EPVelVec')


    # Project the two points p1 and p2 exactly onto the encounter plane 
    #   Given point p and a hyperplane defined by (a,b) with a the normal and b the offset,
    #   The projection of point p following the line p + t*v with v the direction of travel of p onto (a,b) will be where a^T(p + t*v) = b
    #   Thus t_proj = (b - a^T @ p) / (a^T @ v) and where the two intersect is p_proj = p + t_proj * v 
    #   Note that the minimum distance between the HP and the point is achieved when the point is moved in direction v=a for p_proj and min_dist = norm(p-p_proj)
    t1 = (ep_b - ep_a @ p1) / (ep_a @ v1)
    ep_p1 = p1 + t1 * v1
    t2 = (ep_b - ep_a @ p2) / (ep_a @ v2)
    ep_p2 = p2 + t2 * v2

    ax.scatter(ep_p1[0], ep_p1[1], ep_p1[2], s=80, color = 'r', label= 'PrimSat')
    ax.scatter(ep_p2[0], ep_p2[1], ep_p2[2], s=80, color = 'b', label= 'SecSat')

    # Seek out two smallest coefficients in ep_a
    idxs = np.array([0,1,2])
    lci = np.argmax(np.abs(ep_a)) # large coefficient index
    smis = np.delete(idxs, lci) # smallest coefficient indices
    # Create a meshgrid in these two small coefficient coordinates using the plane points ep_p1, ep_p2
    sc1 = np.array(sorted([ ep_p1[smis[0]], ep_p2[smis[0]] ]))
    sc2 = np.array(sorted([ ep_p1[smis[1]], ep_p2[smis[1]] ]))
    if lci == 0:
        Y,Z = np.meshgrid(sc1, sc2)
        X = np.array([(ep_b - ep_a[1]*y - ep_a[2]*z ) / ep_a[0] for y,z in zip(Y,Z)]).reshape(Y.shape)
    elif lci == 1:
        X,Z = np.meshgrid(sc1, sc2)
        Y = np.array([(ep_b - ep_a[0]*x - ep_a[2]*z ) / ep_a[1] for x,z in zip(X,Z)]).reshape(X.shape)
    else:
        X,Y = np.meshgrid(sc1, sc2)
        Z = np.array([(ep_b - ep_a[0]*x - ep_a[1]*y ) / ep_a[2] for x,y in zip(X,Y)]).reshape(X.shape)
    ax.plot_surface(X,Y,Z, alpha=0.2, color='g')

    # Now plot the q_chosen covariance ellipsoids of the primary/secondary sat
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    q_chosen = quantiles[-1]
    s3 = chi2.ppf(q_chosen, 3) # s3 is the value for which e^T @ P^-1 @ e == s3 
    s2 = chi2.ppf(q_chosen, 3) # s3 is the value for which e^T @ P_2DProj^-1 @ e == s3 
    if indiv_var:
        _P = P1
    else: 
        _P = P1 + P2
    if plot_var_3D:
        D1, U1 = np.linalg.eig(_P)
        E1 = U1 @ np.diag(D1 * s3)**0.5 # Ellipse is the matrix square root of covariance
        #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
        unitsphere = _create_unit_sphere()
        ellipse_points1 = (E1 @ unitsphere).squeeze(-1) + p1
        leg_3dvar = 'PrimSatCovar' if indiv_var else 'PrimSecSatCovar'
        ax.plot_surface(*ellipse_points1.transpose(2, 0, 1), rstride=4, cstride=4, color='r', alpha=0.35, label=leg_3dvar)
        if indiv_var:
            # Now plot the 3D 99.99% covariance ellipsoids of the secondary sat
            D2, U2 = np.linalg.eig(P2)
            E2 = U2 @ np.diag(D2 * s3)**0.5 # Ellipse is the matrix square root of covariance
            #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
            unitsphere = _create_unit_sphere()
            ellipse_points2 = (E2 @ unitsphere).squeeze(-1) + p2
            leg_3dvar = 'SecSatCovar'
            ax.plot_surface(*ellipse_points2.transpose(2, 0, 1), rstride=4, cstride=4, color='b', alpha=0.35, label=leg_3dvar)
    if plot_var_2D:
        D1, U1 = np.linalg.eig(_P)
        E1 = U1 @ np.diag(D1 * s3)**0.5 # Ellipse is the matrix square root of covariance
        ep_aproj1 = ep_a.T @ E1 
        ep_bproj = np.linalg.inv(E1) @ (ep_p1 - ep_c)
        t1s = np.atleast_2d( np.array([ np.sin(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
        t2s = np.atleast_2d( np.array([ np.cos(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
        _,_,ep_aproj_orth1 = np.linalg.svd(ep_aproj1.reshape((1,3)))
        ep_aproj_orth1 = ep_aproj_orth1[1:]
        aorth1 = np.atleast_2d( ep_aproj_orth1[0] )
        aorth2 = np.atleast_2d( ep_aproj_orth1[1] )
        us1 = t1s * aorth1 + t2s * aorth2 #- ep_bproj
        ellipse_points_proj1 = (E1 @ us1.T).T + ep_p1
        leg_2dvar = 'PrimSatProj2DVar' if indiv_var else 'PrimSecComboSatProj2DVar'
        ax.plot(ellipse_points_proj1[:,0], ellipse_points_proj1[:,1], ellipse_points_proj1[:,2], color = 'r', label = leg_2dvar)
        if indiv_var:
            D2, U2 = np.linalg.eig(P2)
            E2 = U2 @ np.diag(D2 * s3)**0.5 # Ellipse is the matrix square root of covariance
            ep_aproj2 = ep_a.T @ E2 
            #ep_bproj = np.linalg.inv(E2) @ (ep_p2 - ep_c)
            _,_,ep_aproj_orth2 = np.linalg.svd(ep_aproj2.reshape((1,3)))
            ep_aproj_orth2 = ep_aproj_orth2[1:]
            aorth1 = np.atleast_2d( ep_aproj_orth2[0] )
            aorth2 = np.atleast_2d( ep_aproj_orth2[1] )
            us2 = t1s * aorth1 + t2s * aorth2 #- ep_bproj
            ellipse_points_proj2 = (E2 @ us2.T).T + ep_p2
            leg_2dvar = 'SecSatProj2DVar'
            ax.plot(ellipse_points_proj2[:,0], ellipse_points_proj2[:,1], ellipse_points_proj2[:,2], color = 'b', label = leg_2dvar)

        if p_mce_Ptc is not None:
            assert(s_mce_Ptc is not None)
            q70 = quantiles[0]
            s3_mce = chi2.ppf(q70, 3)
            P1_mce = p_mce_Ptc[0:3,0:3].copy()
            P2_mce = s_mce_Ptc[0:3,0:3].copy()
            if not indiv_var:
                P1_mce += P2_mce
            D1, U1 = np.linalg.eig(P1_mce)
            E1 = U1 @ np.diag(D1 * s3_mce)**0.5 # Ellipse is the matrix square root of covariance
            ep_aproj1 = ep_a.T @ E1 
            #ep_bproj = np.linalg.inv(E1) @ (ep_p1 - ep_c)
            _,_,ep_aproj_orth1 = np.linalg.svd(ep_aproj1.reshape((1,3)))
            ep_aproj_orth1 = ep_aproj_orth1[1:]
            aorth1 = np.atleast_2d( ep_aproj_orth1[0] )
            aorth2 = np.atleast_2d( ep_aproj_orth1[1] )
            us1 = t1s * aorth1 + t2s * aorth2 #- ep_bproj
            ellipse_points_proj1 = (E1 @ us1.T).T + ep_p1
            leg_2dvar = 'MCEPrimSatProj2DVar' if indiv_var else 'MCEPrimSecComboSatProj2DVar'
            ax.plot(ellipse_points_proj1[:,0], ellipse_points_proj1[:,1], ellipse_points_proj1[:,2], color = 'tab:purple', label = leg_2dvar)
            if indiv_var:
                D2, U2 = np.linalg.eig(P2_mce)
                E2 = U2 @ np.diag(D2 * s3_mce)**0.5 # Ellipse is the matrix square root of covariance
                ep_aproj2 = ep_a.T @ E2 
                _,_,ep_aproj_orth2 = np.linalg.svd(ep_aproj2.reshape((1,3)))
                ep_aproj_orth2 = ep_aproj_orth2[1:]
                aorth1 = np.atleast_2d( ep_aproj_orth2[0] )
                aorth2 = np.atleast_2d( ep_aproj_orth2[1] )
                us2 = t1s * aorth1 + t2s * aorth2 #- ep_bproj
                ellipse_points_proj2 = (E2 @ us2.T).T + ep_p2
                leg_2dvar = 'SecSatProj2DVar'
                ax.plot(ellipse_points_proj2[:,0], ellipse_points_proj2[:,1], ellipse_points_proj2[:,2], color = 'tab:cyan', label = leg_2dvar)
            
            
    # Should be a list of points of realizations
    if mc_runs_prim is not None:
        add_leg = True
        for mc_x in mc_runs_prim:
            # Point
            if add_leg:
                ax.scatter(mc_x[0], mc_x[1], mc_x[2], s=20, color = 'm', marker='^', label="MC of Prim at TCA")
            else:
                ax.scatter(mc_x[0], mc_x[1], mc_x[2], s=20, color = 'm', marker='^')
            # Projected point onto plane
            mcp = mc_x[0:3]
            mcv = mc_x[3:6]
            mct = (ep_b - ep_a @ mcp) / (ep_a @ mcv)
            ep_mcp = mcp + mct * mcv
            if add_leg:
                add_leg = False
                ax.scatter(ep_mcp[0], ep_mcp[1], ep_mcp[2], s=20, color = 'm', marker = "*", label="Proj MC of Prim at TCA on EP")
            else:
                ax.scatter(ep_mcp[0], ep_mcp[1], ep_mcp[2], s=20, color = 'm', marker = "*")
            mc_stacked = np.vstack((mcp, ep_mcp))
            ax.plot(mc_stacked[:,0], mc_stacked[:,1], mc_stacked[:,2], color = 'm')
    
    if mc_runs_sec is not None:
        add_leg = True
        for mc_x in mc_runs_sec:
            # Point
            if add_leg:
                ax.scatter(mc_x[0], mc_x[1], mc_x[2], s=20, color = 'tab:orange', marker='^', label="MC of Sec at TCA")
            else:
                ax.scatter(mc_x[0], mc_x[1], mc_x[2], s=20, color = 'tab:orange', marker='^')
            # Projected point onto plane
            mcp = mc_x[0:3]
            mcv = mc_x[3:6]
            mct = (ep_b - ep_a @ mcp) / (ep_a @ mcv)
            ep_mcp = mcp + mct * mcv
            if add_leg:
                add_leg = False
                ax.scatter(ep_mcp[0], ep_mcp[1], ep_mcp[2], s=20, color = 'tab:orange', marker = "*",label="Proj MC of Sec at TCA on EP")
            else:
                ax.scatter(ep_mcp[0], ep_mcp[1], ep_mcp[2], s=20, color = 'tab:orange', marker = "*")
            mc_stacked = np.vstack((mcp, ep_mcp))
            ax.plot(mc_stacked[:,0], mc_stacked[:,1], mc_stacked[:,2], color = 'tab:orange')

    # Set plot limits
    #'''
    ax.set_xlabel("x-axis (km)")
    ax.set_ylabel("y-axis (km)")
    ax.set_zlabel("z-axis (km)")
    leg = ax.legend()
    leg.set_draggable(state=True)
    xlim_low  = np.min([p1[0],p2[0]])
    xlim_high  = np.max([p1[0],p2[0]])
    ax.axes.set_xlim3d(left = xlim_low * (1.0 - scales[0] * np.sign(xlim_low)), 
                right = xlim_high * (1.0 + scales[0] * np.sign(xlim_high)))
    ylim_low  = np.min([p1[1],p2[1]])
    ylim_high  = np.max([p1[1],p2[1]])
    ax.axes.set_ylim3d(bottom = ylim_low * (1.0 - scales[1] * np.sign(ylim_low)), 
                top = ylim_high * (1.0 + scales[1] * np.sign(ylim_high)))
    zlim_low  = np.min([p1[2],p2[2]])
    zlim_high  = np.max([p1[2],p2[2]])
    ax.axes.set_zlim3d(bottom = zlim_low * (1.0 - scales[2] * np.sign(zlim_low)), 
                top = zlim_high * (1.0 + scales[2] * np.sign(zlim_high)))    
    #'''
    plt.show()
    foobar = 5

def draw_2d_projected_encounter_plane(quantile_kf,
    s_xtc, p_xtc, s_Ptc, p_Ptc, 
    mc_prim = None, mc_sec = None, 
    quantile_mce = None, s_mce_Ptc = None, p_mce_Ptc = None, 
    mc_sec_sample_tcas = None, mc_prim_sample_tcas = None):
    
    # Create relative quantities
    p_p = p_xtc[0:3] # position vector of primary 
    s_p = s_xtc[0:3] # position vector of secondary 
    rp = s_xtc[0:3] - p_xtc[0:3] # relative position vector
    rv = s_xtc[3:6] - p_xtc[3:6] # relative velocity vector
    # Transformation into (for now) coordinates orthogonal to the relative vel. vec
    _,_,T = np.linalg.svd(rv.reshape((1,3)))
    # Get expected location for the transformed (projected onto 2D plane) relative pos
    T2D = T[1:, :]
    rp2D = T2D @ rp # expected relative 2D position
    rP2D = T2D @ (s_Ptc[0:3,0:3] + p_Ptc[0:3,0:3]) @ T2D.T # expected relative 2D covariance
    # Encounter plane ep_a^T @ x = ep_b
    ep_c = (p_p + s_p) / 2.0
    ep_a = rv # normal to the encounter hyperplane is the relative vel vec direction
    ep_b = ep_a @ ep_c # offset 
    
    # Now plot the q_chosen covariance ellipsoids of the primary/secondary sat
    #quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    #q_chosen = quantiles[-1]
    assert quantile_kf < 1
    s2 = chi2.ppf(quantile_kf, 2) # s3 is the value for which e^T @ P_2DProj^-1 @ e == s2 
    t1s = np.atleast_2d( np.array([ np.sin(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    t2s = np.atleast_2d( np.array([ np.cos(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    unit_circle = np.hstack((t1s,t2s))
    D, U = np.linalg.eig(rP2D)
    E = U @ np.diag(D * s2)**0.5 # Ellipse is the matrix square root of covariance
    ell_points = (E @ unit_circle.T).T + rp2D 
    fig = plt.figure()
    plt.title("MC of Prim/Sec Positions at TCA, first projected onto \n(the expected) encounter plane, then made relative (sec - prim)")
    plt.plot(ell_points[:,0], ell_points[:,1], color='r', label='99.99% Proj 2D Cov. Ellipse for KF')
    plt.scatter(rp2D[0], rp2D[1], color='r', label='Expected relative projected position')
    plt.xlabel("Rel Vel Vec Orthog Direction 1 (km)")
    plt.ylabel("Rel Vel Vec Orthog Direction 2 (km)")

    # Now plot MCE if provided 
    if p_mce_Ptc is not None:
        assert s_mce_Ptc is not None
        mce_rP2D = T2D @ (s_mce_Ptc[0:3,0:3] + p_mce_Ptc[0:3,0:3]) @ T2D.T # expected relative 2D covariance
        assert quantile_mce is not None
        assert quantile_mce < 1
        s2 = chi2.ppf(quantile_mce, 2)
        D, U = np.linalg.eig(mce_rP2D)
        E = U @ np.diag(D * s2)**0.5 # Ellipse is the matrix square root of covariance
        ell_points = (E @ unit_circle.T).T + rp2D 
        plt.plot(ell_points[:,0], ell_points[:,1], color='g', label='70% Proj 2D Cov. Ellipse for MCE')

    # Plot Monte Carlo points enamating from x0 projected onto the encounter plane, if not None
    if (mc_prim is not None) and (mc_sec is not None):
        assert len(mc_prim) == len(mc_sec)
        proj_mcs = []
        for mcp, mcs in zip(mc_prim, mc_sec):
            # Project prim onto encounter plane 
            p1 = mcp[0:3]
            v1 = mcp[3:6]
            t1 = (ep_b - ep_a @ p1) / (ep_a @ v1)
            ep_p1 = p1 + t1 * v1
            # project sec onto encounter plane 
            p2 = mcs[0:3]
            v2 = mcs[3:6]
            t2 = (ep_b - ep_a @ p2) / (ep_a @ v2)
            ep_p2 = p2 + t2 * v2
            # subtract, project down
            mcr = ep_p2 - ep_p1
            proj_mc = T2D @ mcr
            proj_mcs.append(proj_mc)
        proj_mcs = np.array(proj_mcs)    
        plt.scatter(proj_mcs[0,0], proj_mcs[0,1], color='m', label = 'MC real. of (rel.) proj. pos. onto EP at TCA from x0')
        plt.scatter(proj_mcs[1:,0], proj_mcs[1:,1], color='m')

    # Plot Monte Carlo points enamating from N(x0,P0) projected onto the encounter plane, if not None
    if (mc_sec_sample_tcas is not None) and (mc_prim_sample_tcas is not None):
        assert len(mc_sec_sample_tcas) == len(mc_prim_sample_tcas)
        proj_mcs = []
        for mcp, mcs in zip(mc_prim_sample_tcas, mc_sec_sample_tcas):
            # Project prim onto encounter plane 
            p1 = mcp[0:3]
            v1 = mcp[3:6]
            t1 = (ep_b - ep_a @ p1) / (ep_a @ v1)
            ep_p1 = p1 + t1 * v1
            # project sec onto encounter plane 
            p2 = mcs[0:3]
            v2 = mcs[3:6]
            t2 = (ep_b - ep_a @ p2) / (ep_a @ v2)
            ep_p2 = p2 + t2 * v2
            # subtract, project down
            mcr = ep_p2 - ep_p1
            proj_mc = T2D @ mcr
            proj_mcs.append(proj_mc)
        proj_mcs = np.array(proj_mcs)
        plt.scatter(proj_mcs[0,0], proj_mcs[0,1], color='g', label = 'MC real. of (rel.) proj. pos. onto EP at TCA from x0 sampled')
        plt.scatter(proj_mcs[1:,0], proj_mcs[1:,1], color='g')
    plt.legend().set_draggable(True)
    plt.show()
    foobar=5

def analyze_3d_statistics(quantile_kf, quantile_mce,
    s_xtc, p_xtc, s_Ptc, p_Ptc,
    s_mce_Ptc, p_mce_Ptc, 
    mc_prim_tcas, mc_sec_tcas,
    mc_sec_sample_tcas, mc_prim_sample_tcas):
    
    s3_quant_kf = chi2.ppf(quantile_kf, 3)
    s3_quant_mce = chi2.ppf(quantile_mce, 3)

    # KF and MCE Points
    x1 = p_xtc[0:3]
    v1 = p_xtc[3:6]
    sv1 = v1 / np.linalg.norm(v1)
    Pkf1 = p_Ptc[0:3,0:3]
    Pkf1_I = np.linalg.inv(Pkf1)
    Pce1 = p_mce_Ptc[0:3,0:3]
    Pce1_I = np.linalg.inv(Pce1)
    x2 = s_xtc[0:3]
    v2 = s_xtc[3:6]
    sv2 = v2 / np.linalg.norm(v2)
    Pkf2 = s_Ptc[0:3,0:3]
    Pkf2_I = np.linalg.inv(Pkf2)
    Pce2 = s_mce_Ptc[0:3,0:3]
    Pce2_I = np.linalg.inv(Pce2)
    # Monte Carlo points emanating from x0
    mc_prim_x0s = np.array(mc_prim_tcas)[:,0:3] if mc_prim_tcas is not None else None
    mc_sec_x0s = np.array(mc_sec_tcas)[:,0:3] if mc_sec_tcas is not None else None
    # Monte Carlo points sampled from x0,P0_kf
    mc_prim_sampled = np.array(mc_prim_sample_tcas)[:,0:3] if mc_prim_sample_tcas is not None else None
    mc_sec_sampled = np.array(mc_sec_sample_tcas)[:,0:3] if mc_sec_sample_tcas is not None else None

    # =============
    # First subplot -- KF of PRIM
    # =============
    # set up the axes for the first plot
    fig = plt.figure() 
    ax11 = fig.gca(projection='3d') 
    # Plot The KF Covariance Ellipsoid 
    D, U = np.linalg.eig(Pkf1)
    E = U @ np.diag(D * s3_quant_kf)**0.5 # Ellipse is the matrix square root of covariance
    #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
    unitsphere = _create_unit_sphere()
    ellipse_points = (E @ unitsphere).squeeze(-1) + x1
    ax11.plot(x1[0],x1[1],x1[2], color='r', label='KF Primary Satellite ' + str(100*quantile_kf) + "% Ellipse")
    ax11.plot_surface(*ellipse_points.transpose(2, 0, 1), rstride=4, cstride=4, color='r', alpha=0.35)
    ax11.arrow3D(x1[0], x1[1], x1[2], sv1[0], sv1[1], sv1[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='b', label='Primary satellite velocity vector direction')
    
    mc_points = np.zeros((0,3))
    with_mc_points = False
    kf_counts = 0
    len_points = 0
    if mc_prim_x0s is not None:
        with_mc_points = True
        ax11.scatter(mc_prim_x0s[:,0], mc_prim_x0s[:,1], mc_prim_x0s[:,2], color='m', label='MC realizations forced by SaS=1.3 atms. density changes, I.C is x0')
        mc_points = np.vstack((mc_points, mc_prim_x0s)) 
    if mc_prim_sampled is not None:
        with_mc_points = True
        ax11.scatter(mc_prim_sampled[:,0], mc_prim_sampled[:,1], mc_prim_sampled[:,2], color='g', label='MC realizations forced by SaS=1.3 atms. density changes, I.C is N(x0,P0_kf)')
        mc_points = np.vstack((mc_points,mc_prim_sampled)) 
    if with_mc_points:
        min_bounds = x1 - np.min(mc_points, axis = 0)
        max_bounds = np.max(mc_points, axis = 0) - x1
        eq_bounds = np.max( np.vstack((min_bounds,max_bounds)), axis = 0 )
        ax11.set_xbound(x1[0]-eq_bounds[0], x1[0]+eq_bounds[0])
        ax11.set_ybound(x1[1]-eq_bounds[1], x1[1]+eq_bounds[1])
        ax11.set_zbound(x1[2]-eq_bounds[2], x1[2]+eq_bounds[2])
        len_points = mc_points.shape[0]
        for mcp in mc_points:
            is_in_kf_prim = (mcp-x1) @ Pkf1_I @ (mcp-x1) < s3_quant_mce
            kf_counts += is_in_kf_prim
    ax11.set_title('KF 3D Position Covariance ' + str(quantile_kf*100) + "% Ellipsoid for the 7-day Forward Projected Primary Satellite\n Plotted against monte carlo realizations of satellite location (green/magenta)\n# Points inside Ellipsoid={}/{} points total".format(kf_counts, len_points) )
    ax11.set_xlabel("x-axis (km)")
    ax11.set_ylabel("y-axis (km)")
    ax11.set_zlabel("z-axis (km)")
    leg = ax11.legend()
    leg.set_draggable(state=True)
    plt.show()

    # =============
    # Second subplot -- KF of SEC
    # =============
    # set up the axes for the first plot
    fig = plt.figure() 
    ax12 = fig.gca(projection='3d') 
    ax12.set_title('KF of Sec Sat')
    # Plot The KF Covariance Ellipsoid 
    D, U = np.linalg.eig(Pkf2)
    E = U @ np.diag(D * s3_quant_kf)**0.5 # Ellipse is the matrix square root of covariance
    #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
    unitsphere = _create_unit_sphere()
    ellipse_points = (E @ unitsphere).squeeze(-1) + x2
    ax12.plot(x2[0],x2[1],x2[2], color='r', label='KF Secondary Satellite ' + str(100*quantile_kf) + "% Ellipse")
    ax12.plot_surface(*ellipse_points.transpose(2, 0, 1), rstride=4, cstride=4, color='r', alpha=0.35)
    ax12.arrow3D(x2[0], x2[1], x2[2], sv2[0], sv2[1], sv2[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='b', label='Secondary satellite velocity vector direction')
    
    mc_points = np.zeros((0,3))
    with_mc_points = False
    kf_counts = 0
    len_points = 0
    if mc_sec_x0s is not None:
        with_mc_points = True
        ax12.scatter(mc_sec_x0s[:,0], mc_sec_x0s[:,1], mc_sec_x0s[:,2], color='m',label='MC realizations forced by SaS=1.3 atms. density changes, I.C is x0')
        mc_points = np.vstack((mc_points, mc_sec_x0s)) 
    if mc_sec_sampled is not None:
        with_mc_points = True
        ax12.scatter(mc_sec_sampled[:,0], mc_sec_sampled[:,1], mc_sec_sampled[:,2], color='g', label='MC realizations forced by SaS=1.3 atms. density changes, I.C is N(x0,P0_kf)')
        mc_points = np.vstack((mc_points, mc_sec_sampled)) 
    if with_mc_points:
        min_bounds = x2 - np.min(mc_points, axis = 0)
        max_bounds = np.max(mc_points, axis = 0) - x2
        eq_bounds = np.max( np.vstack((min_bounds,max_bounds)), axis = 0 )
        ax12.set_xbound(x2[0]-eq_bounds[0], x2[0]+eq_bounds[0])
        ax12.set_ybound(x2[1]-eq_bounds[1], x2[1]+eq_bounds[1])
        ax12.set_zbound(x2[2]-eq_bounds[2], x2[2]+eq_bounds[2])
        len_points = mc_points.shape[0]
        for mcp in mc_points:
            is_in_kf_sec = (mcp-x2) @ Pkf2_I @ (mcp-x2) < s3_quant_mce
            kf_counts += is_in_kf_sec
    ax12.set_title('KF 3D Position Covariance ' + str(quantile_kf*100) + "% Ellipsoid for the 7-day Forward Projected Secondary Satellite\n Plotted against monte carlo realizations of satellite location (green/magenta)\n# Points inside Ellipsoid={}/{} points total".format(kf_counts, len_points) )
    leg = ax12.legend()
    ax12.set_xlabel("x-axis (km)")
    ax12.set_ylabel("y-axis (km)")
    ax12.set_zlabel("z-axis (km)")
    leg.set_draggable(state=True)
    plt.show()

    # =============
    # Third subplot -- MCE of PRIM
    # =============
    # set up the axes for the first plot
    fig = plt.figure() 
    ax21 = fig.gca(projection='3d') 
    ax21.set_title('MCE of Prim Sat')
    # Plot The MCE Covariance Ellipsoid 
    D, U = np.linalg.eig(Pce1)
    E = U @ np.diag(D * s3_quant_kf)**0.5 # Ellipse is the matrix square root of covariance
    #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
    unitsphere = _create_unit_sphere()
    ellipse_points = (E @ unitsphere).squeeze(-1) + x1
    ax21.plot(x1[0],x1[1],x1[2], color='r', label='MCE Primary Satellite ' + str(100*quantile_mce) + "% Ellipse")
    ax21.plot_surface(*ellipse_points.transpose(2, 0, 1), rstride=4, cstride=4, color='r', alpha=0.35)
    ax21.arrow3D(x1[0], x1[1], x1[2], sv1[0], sv1[1], sv1[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='b', label='Primary satellite velocity vector direction')
    
    mc_points = np.zeros((0,3))
    with_mc_points = False
    mce_counts = 0
    len_points = 0
    if mc_prim_x0s is not None:
        with_mc_points = True
        ax21.scatter(mc_prim_x0s[:,0], mc_prim_x0s[:,1], mc_prim_x0s[:,2], color='m',label='MC realizations forced by SaS=1.3 atms. density changes. I.C is x0')
        mc_points = np.vstack((mc_points,mc_prim_x0s)) 
    if mc_prim_sampled is not None:
        with_mc_points = True
        ax21.scatter(mc_prim_sampled[:,0], mc_prim_sampled[:,1], mc_prim_sampled[:,2], color='g',label='MC realizations forced by SaS=1.3 atms. density changes, I.C is N(x0,P0_kf)')
        mc_points = np.vstack((mc_points,mc_prim_sampled)) 
    if with_mc_points:
        min_bounds = x1 - np.min(mc_points, axis = 0)
        max_bounds = np.max(mc_points, axis = 0) - x1
        eq_bounds = np.max( np.vstack((min_bounds,max_bounds)), axis = 0 )
        ax21.set_xbound(x1[0]-eq_bounds[0], x1[0]+eq_bounds[0])
        ax21.set_ybound(x1[1]-eq_bounds[1], x1[1]+eq_bounds[1])
        ax21.set_zbound(x1[2]-eq_bounds[2], x1[2]+eq_bounds[2])
        len_points = mc_points.shape[0]
        for mcp in mc_points:
            is_in_mce_prim = (mcp-x1) @ Pce1_I @ (mcp-x1) < s3_quant_mce
            mce_counts += is_in_mce_prim
    ax21.set_title('MCE 3D Position Covariance ' + str(quantile_mce*100) + "% Ellipsoid for the 7-day Forward Projected Primary Satellite\n Plotted against monte carlo realizations of satellite location (green/magenta)\n# Points inside Ellipsoid={}/{} points total".format(mce_counts, len_points) )
    ax21.set_xlabel("x-axis (km)")
    ax21.set_ylabel("y-axis (km)")
    ax21.set_zlabel("z-axis (km)")
    leg = ax21.legend()
    leg.set_draggable(state=True)
    plt.show()

    # =============
    # Fourth subplot -- MCE of SEC
    # =============
    # set up the axes for the first plot
    fig = plt.figure() 
    ax22 = fig.gca(projection='3d') 
    # Plot The MCE Covariance Ellipsoid 
    D, U = np.linalg.eig(Pce2)
    E = U @ np.diag(D * s3_quant_kf)**0.5 # Ellipse is the matrix square root of covariance
    #unitsphere = create_unit_sphere(15 * np.pi/180, 15 * np.pi/180)
    unitsphere = _create_unit_sphere()
    ellipse_points = (E @ unitsphere).squeeze(-1) + x2
    ax22.plot(x2[0],x2[1],x2[2], color='r', label='MCE Secondary Satellite ' + str(100*quantile_mce) + "% Ellipse")
    ax22.plot_surface(*ellipse_points.transpose(2, 0, 1), rstride=4, cstride=4, color='r', alpha=0.35)
    ax22.arrow3D(x2[0], x2[1], x2[2], sv2[0], sv2[1], sv2[2], arrowstyle="-|>", mutation_scale=20, lw=3, color='b', label='Secondary satellite velocity vector direction')
    
    mc_points = np.zeros((0,3))
    with_mc_points = False
    mce_counts = 0
    len_points = 0
    if mc_sec_x0s:
        with_mc_points = True
        ax22.scatter(mc_sec_x0s[:,0], mc_sec_x0s[:,1], mc_sec_x0s[:,2], color='m',label='MC realizations forced by SaS=1.3 atms. density changes, I.C is x0')
        mc_points = np.vstack((mc_points,mc_sec_x0s))
    if mc_sec_sampled is not None:
        with_mc_points = True
        ax22.scatter(mc_sec_sampled[:,0], mc_sec_sampled[:,1], mc_sec_sampled[:,2], color='g',label='MC realizations forced by SaS=1.3 atms. density changes, I.C is N(x0,P0_kf)')
        mc_points = np.vstack((mc_points,mc_sec_sampled))
    if with_mc_points:
        mc_points = np.vstack((mc_sec_x0s,mc_sec_sampled)) 
        min_bounds = x2 - np.min(mc_points, axis = 0)
        max_bounds = np.max(mc_points, axis = 0) - x2
        eq_bounds = np.max( np.vstack((min_bounds,max_bounds)), axis = 0 )
        ax22.set_xbound(x2[0]-eq_bounds[0], x2[0]+eq_bounds[0])
        ax22.set_ybound(x2[1]-eq_bounds[1], x2[1]+eq_bounds[1])
        ax22.set_zbound(x2[2]-eq_bounds[2], x2[2]+eq_bounds[2])
        len_points = mc_points.shape[0]
        for mcp in mc_points:
            is_in_mce_sec = (mcp-x2) @ Pce2_I @ (mcp-x2) < s3_quant_mce
            mce_counts += is_in_mce_sec
    ax22.set_title('MCE 3D Position Covariance ' + str(quantile_mce*100) + "% Ellipsoid for the 7-day Forward Projected Secondary Satellite\n Plotted against monte carlo realizations of satellite location (green/magenta)\n# Points inside Ellipsoid={}/{} points total".format(mce_counts, mc_points.shape[0]) )
    ax22.set_xlabel("x-axis (km)")
    ax22.set_ylabel("y-axis (km)")
    ax22.set_zlabel("z-axis (km)")
    leg = ax22.legend()
    leg.set_draggable(state=True)
    plt.show()
    foobar = 3


'''
def old_iterative_time_closest_approach(dt, _t0, prim_tup, sec_tup, start_idx = 0, its = -1, with_plot=True):
    # For now this function assumes an integer time step
    assert(int(dt) == dt)
    # Initial iteration
    p_pks,p_vks,p_aks = copy.deepcopy(prim_tup)
    s_pks,s_vks,s_aks = copy.deepcopy(sec_tup)
    t0 = copy.deepcopy(_t0)
    tks = np.arange(p_pks.shape[0]) * dt
    i, troot, t_c, pp_c, pv_c, sp_c, sv_c = closest_approach_info(tks[start_idx:], 
        (p_pks[start_idx:,:],p_vks[start_idx:,:],p_aks[start_idx:,:]), 
        (s_pks[start_idx:,:],s_vks[start_idx:,:],s_aks[start_idx:,:]))
    i += start_idx
    # Left hand side of interval
    i_star_lhs = i
    t_lhs = tks[i]

    if with_plot:
        fig = plt.figure() 
        ax = fig.gca(projection='3d')
        plt.title("LEO Primary (red) vs. Secondary (blue) trajectory over time")
        ax.scatter(p_pks[start_idx:i+2,0], p_pks[start_idx:i+2,1], p_pks[start_idx:i+2,2], color = 'r')
        ax.scatter(s_pks[start_idx:i+2,0], s_pks[start_idx:i+2,1], s_pks[start_idx:i+2,2], color = 'b')
        ax.set_xlabel("x-axis (km)")
        ax.set_ylabel("y-axis (km)")
        ax.set_zlabel("z-axis (km)")
        # Plot relative difference in position
        fig2 = plt.figure()
        plt.title("Norm of Position Difference between Primary and Secondary")
        plt.plot(tks[start_idx:i+2], np.linalg.norm(p_pks[start_idx:i+2]-s_pks[start_idx:i+2], axis=1))
        plt.xlabel("Time (sec)")
        plt.ylabel("2-norm of position difference (km)")
        plt.show()

    print("Iteration: 1")
    print("t0: ", t0, "(timestamp)")
    print("Step dt: ", dt, "(sec)")
    print("Tc: {} (sec), Idx of Tc: {}".format(t_c, i) )
    print("Primary at Tc: ", pp_c, "(km)")
    print("Secondary at Tc: ", sp_c, "(km)")
    print("Pos Diff is: ", pp_c-sp_c, "(km)")
    print("Pos Norm is: ", 1000*np.linalg.norm(pp_c-sp_c), "(m)")

    # Now we know minimum is somewhere over [i,i+1]
    # Know the start time is now t0 + tks[i]
    substeps = [int(dt),30,30,30]
    its = len(substeps) if its == -1 else its
    for it in range(its):
        if it == 0:
            x0_prim = np.concatenate( (p_pks[i],p_vks[i]) )
            x0_sec  = np.concatenate( (s_pks[i],s_vks[i]) )
            t0 = t0 + timedelta( seconds = tks[i] )
            dt = dt / substeps[it]
            tks = tks[i] + np.arange(substeps[it]+1) * dt
        else:
            if (i > 0) and (i+2 < (substeps[it-1]+1) ):
                j = i-1
                scale = 3
            elif i == 0:
                print("Hit LOWER BOUNDARY i == 0")
                j = i 
                scale = 2
                substeps[it] = 20
            elif (i+2) == (substeps[it-1]+1):
                print("Hit UPPER BOUNDARY i+2 ==",i+2)
                j = i-1
                scale = 2
                substeps[it] = 20
            x0_prim = np.concatenate( (p_pks[j],p_vks[j]) )
            x0_sec  = np.concatenate( (s_pks[j],s_vks[j]) )
            t0 = t0 + timedelta( seconds = tks[j] )
            dt = scale*dt / substeps[it]
            tks = tks[j] + np.arange(substeps[it]+1) * dt
        
        # Now run the primary over the subinterval i to i+1
        fermiSat = gsat.FermiSatelliteModel(t0, x0_prim, dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        p_pks = [] # Primary Pos 3-Vec
        p_vks = [] # Primary Vel 3-Vec
        p_aks = [] # Primary Acc 3-Vec
        # Propagate Primary and Store
        xk = x0_prim.copy()
        for i in range(substeps[it]+1):
            dxk_dt = fermiSat.get_state6_derivatives() 
            p_pks.append(xk[0:3])
            p_vks.append(xk[3:6])
            p_aks.append(dxk_dt[3:6])
            xk = fermiSat.step()
        p_pks = np.array(p_pks)
        p_vks = np.array(p_vks)
        p_aks = np.array(p_aks)
        fermiSat.clear_model()

        # Create Satellite Model for Secondary
        fermiSat = gsat.FermiSatelliteModel(t0, x0_sec, dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        s_pks = [] # Secondary Pos 3-Vec
        s_vks = [] # Secondary Vel 3-Vec
        s_aks = [] # Secondary Acc 3-Vec
        # Propagate Secondary and Store
        xk = x0_sec.copy()
        for i in range(substeps[it]+1):
            dxk_dt = fermiSat.get_state6_derivatives() 
            s_pks.append(xk[0:3])
            s_vks.append(xk[3:6])
            s_aks.append(dxk_dt[3:6])
            xk = fermiSat.step()
        s_pks = np.array(s_pks)
        s_vks = np.array(s_vks)
        s_aks = np.array(s_aks)
        fermiSat.clear_model()

        # Now re-run the closest approach routine
        i, troot, t_c, pp_c, pv_c, sp_c, sv_c = closest_approach_info(tks, (p_pks,p_vks,p_aks), (s_pks,s_vks,s_aks))

        print("Iteration: ", it + 2, " (timestamp)")
        print("Step dt: ", dt, "(sec)")
        print("Tc: {} (sec), Idx of Tc: {}".format(t_c, i) )
        print("Primary at Tc: ", pp_c, "(km)")
        print("Secondary at Tc: ", sp_c, "(km)")
        print("Pos Diff is: ", pp_c-sp_c, "(km)")
        print("Pos Norm is: ", 1000*np.linalg.norm(pp_c-sp_c), "(m)")

        if with_plot:
            # Black points give found interval, # green point are +/- 1 buffers
            fig = plt.figure() 
            ax = fig.gca(projection='3d')
            plt.title("Leo Trajectory over Time")
            ax.scatter(p_pks[:i+2,0], p_pks[:i+2,1], p_pks[:i+2,2], color = 'r')
            ax.scatter(s_pks[:i+2,0], s_pks[:i+2,1], s_pks[:i+2,2], color = 'b')
            ax.scatter(p_pks[i,0], p_pks[i,1], p_pks[i,2], color = 'k')
            ax.scatter(s_pks[i,0], s_pks[i,1], s_pks[i,2], color = 'k')
            if (i+1) < s_pks.shape[0]:
                ax.scatter(p_pks[i+1,0], p_pks[i+1,1], p_pks[i+1,2], color = 'k')
                ax.scatter(s_pks[i+1,0], s_pks[i+1,1], s_pks[i+1,2], color = 'k')
            ax.set_xlabel("x-axis (km)")
            ax.set_ylabel("y-axis (km)")
            ax.set_zlabel("z-axis (km)")
            
            dist_norm = np.linalg.norm(p_pks[:i+5]-s_pks[:i+5], axis=1)
            fig2 = plt.figure()
            plt.title("Norm of Position Difference between Primary and Secondary")
            plt.plot(tks[:i+5], dist_norm)
            plt.scatter(tks[i-1], dist_norm[i-1], color='g')
            plt.scatter(tks[i], dist_norm[i], color='k')
            if (i+1) < s_pks.shape[0]:
                plt.scatter(tks[i+1], dist_norm[i+1], color='k')
            if (i+2) < s_pks.shape[0]:
                plt.scatter(tks[i+2], dist_norm[i+2], color='g')
            plt.xlabel("Time (sec)")
            plt.ylabel("2-norm of position difference (km)")
            plt.show()
            foobar=3


    
    # i_star_lhs -> The nominal propagation index to stop at 
    # t_lhs -> The time at the nominal propagation index 
    # t_c -> The final time of closest approach 
    # pp_c -> The position of the primary at closest approach 
    # pv_c -> The velocity of the primary at closest approach 
    # sp_c -> The position of the secondary at closest approach
    # sv_c -> The velocity of the secondary at closest approach 
    return i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c



def old_draw_2d_projected_encounter_plane(s_xtc, p_xtc, s_Ptc, p_Ptc, mc_prim, mc_sec, with_ep_proj=True):
    # Create relative quantities
    p_p = p_xtc[0:3] # position vector of primary 
    s_p = s_xtc[0:3] # position vector of secondary 
    rp = s_xtc[0:3] - p_xtc[0:3] # relative position vector
    # Transformation into along, cross and radial track coordinates 
    T = get_along_cross_radial_transformation(s_xtc)
    # Get expected location for the cross and radial track directions only 
    T2D = T[1:, :]
    rp2D = T2D @ rp # expected relative 2D position for cross and radial track directions
    rP2D = T2D @ (s_Ptc[0:3,0:3] + p_Ptc[0:3,0:3]) @ T2D.T # expected relative 2D covariance
    # Encounter plane ep_a^T @ x = ep_b
    ep_c = (p_p + s_p) / 2.0
    ep_a = T[0] # normal to the encounter hyperplane is the along track direction
    ep_b = ep_a @ ep_c # offset 
    
    # Now plot the q_chosen covariance ellipsoids of the primary/secondary sat
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    q_chosen = quantiles[-1]
    s2 = chi2.ppf(q_chosen, 2) # s3 is the value for which e^T @ P_2DProj^-1 @ e == s2 
    t1s = np.atleast_2d( np.array([ np.sin(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    t2s = np.atleast_2d( np.array([ np.cos(2*np.pi*t) for t in np.linspace(0,1,100)]) ).T
    unit_circle = np.hstack((t1s,t2s))
    D, U = np.linalg.eig(rP2D)
    E = U @ np.diag(D * s2)**0.5 # Ellipse is the matrix square root of covariance
    fig = plt.figure()
    ell_points = (E @ unit_circle.T).T + rp2D 
    plt.title("MC of Prim/Sec Positions at TCA, first projected onto \n(the expected) encounter plane, then made relative (sec - prim)")
    plt.plot(ell_points[:,0], ell_points[:,1], color='r', label='99.99\% Proj 2D Covariance Ellipsoid')
    plt.scatter(rp2D[0], rp2D[1], color='r', label=r'Expected projected and relative position $r^{k+N|k}=\mathbb{E}[p_s^{k+N|k} - p^{k+N|k}_p]$')
    plt.xlabel("Cross-Track Direction (km)")
    plt.ylabel("Radial-Track Direction (km)")

    leg_mc = None
    for mcp,mcs in zip(mc_prim, mc_sec):
        # Project prim onto encounter plane 
        p1 = mcp[0:3]
        v1 = mcp[3:6]
        t1 = (ep_b - ep_a @ p1) / (ep_a @ v1)
        ep_p1 = p1 + t1 * v1
        # project sec onto encounter plane 
        p2 = mcs[0:3]
        v2 = mcs[3:6]
        t2 = (ep_b - ep_a @ p2) / (ep_a @ v2)
        ep_p2 = p2 + t2 * v2
        # subtract, project down
        mcr = ep_p2 - ep_p1
        proj_mc = T2D @ mcr
        if leg_mc is None:
            leg_mc = True
            plt.scatter(proj_mc[0], proj_mc[1], color='m', label = 'MC real. of (rel.) proj. pos. onto EP at TCA')
        else:
            plt.scatter(proj_mc[0], proj_mc[1], color='m')
    plt.show()
    foobar=5

'''