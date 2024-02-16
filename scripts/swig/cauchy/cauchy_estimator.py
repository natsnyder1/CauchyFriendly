from cmath import log
import numpy as np 
import ctypes as ct
import pycauchy
import matplotlib.pyplot as plt 
import math
import os

CAUCHY_TO_GAUSSIAN_NOISE = 1.3898
GAUSSIAN_TO_CAUCHY_NOISE = 1.0 / CAUCHY_TO_GAUSSIAN_NOISE

# Used to initialize communication with the underlying shared library
# Can run in 3 modes
# lti -- Dynamics matrices constant
# ltv -- Dynamics matrices non-constant
# nonlin -- Nonlinear system, uses extended cauchy

# Helper Functions
def set_tr_search_idxs_ordering(ordering):
    if type(ordering) == np.ndarray:
        _ordering = ordering.copy().astype(np.int32)
    else:
        _ordering = np.array(ordering).astype(np.int32)
    return pycauchy.pycauchy_set_tr_search_idxs_ordering(_ordering)

def speyers_window_init(x1_hat, Var, H, gamma, z):
    assert x1_hat.size == Var.shape[0] == Var.shape[1]
    assert x1_hat.size == H.size    
    n = x1_hat.size
    _x1_hat = x1_hat.astype(np.float64).reshape(-1)
    _Var = Var.astype(np.float64).reshape(-1)
    _H = H.astype(np.float64).reshape(-1)
    _gamma = float(gamma)
    _z = float(z)
    _A0, p0, b0 = pycauchy.pycauchy_speyers_window_init(_x1_hat, _Var, _H, _gamma, _z)
    A0 = _A0.reshape((n,n))
    return A0, p0, b0

 # Dynamic Update Callback Containers for Underlying LTV/Nonlin run-modes
class C_CauchyDynamicsUpdateContainer(ct.Structure):
    _fields_ = [
                ("x", ct.POINTER(ct.c_double)),
                ("u", ct.POINTER(ct.c_double)),
                ("dt", ct.c_double),
                ("step", ct.c_int),
                ("n", ct.c_int),
                ("cmcc", ct.c_int),
                ("pncc", ct.c_int),
                ("p", ct.c_int),
                ("Phi", ct.POINTER(ct.c_double)),
                ("Gamma", ct.POINTER(ct.c_double)),
                ("B", ct.POINTER(ct.c_double)),
                ("H", ct.POINTER(ct.c_double)),
                ("beta", ct.POINTER(ct.c_double)),
                ("gamma", ct.POINTER(ct.c_double)),
                ("is_xbar_set_for_ece", ct.c_bool),
                ("other_stuff", ct.c_void_p)
                ]

class Py_CauchyDynamicsUpdateContainer():
    def __init__(self, cduc):
        self.cduc = cduc
        self.n = self.cduc.contents.n
        self.pncc = self.cduc.contents.pncc 
        self.cmcc = self.cduc.contents.cmcc 
        self.p =  self.cduc.contents.p 
    
    # The cget/cset methods set or get the underline C pointers
    def cget_step(self):
        return self.cduc.contents.step
    
    def cget_dt(self):
        return self.cduc.contents.dt

    # The inputs to these functions are numpy vectors / matrices, which then sets the raw C-Pointers
    def cget_x(self):
        x = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            x[i] = self.cduc.contents.x[i]
        return x

    def cset_x(self, x):
        assert(x.ndim == 1)
        assert(x.size == self.n)
        for i in range(self.n):
            self.cduc.contents.x[i] = x[i]
    
    def cget_u(self):
        u = np.zeros(self.cmcc, dtype=np.float64)
        for i in range(self.cmcc):
            u[i] = self.cduc.contents.u[i]
        return u 

    def cget_Phi(self):
        size_Phi = self.n * self.n
        Phi = np.zeros(size_Phi)
        for i in range(size_Phi):
            Phi[i] = self.cduc.contents.Phi[i]
        return Phi.reshape((self.n , self.n))

    def cset_Phi(self, Phi):
        size_Phi = self.n * self.n
        assert(Phi.size == size_Phi)
        _Phi = Phi.reshape(-1)
        for i in range(size_Phi):
            self.cduc.contents.Phi[i] = _Phi[i]

    def cget_Gamma(self):
        size_Gamma = self.n*self.pncc
        Gamma = np.zeros(size_Gamma)
        for i in range(size_Gamma):
            Gamma[i] = self.cduc.contents.Gamma[i]
        return Gamma.reshape((self.n, self.pncc))

    def cset_Gamma(self, Gamma):
        size_Gamma = self.n*self.pncc
        assert(Gamma.size == size_Gamma)
        _Gamma = Gamma.reshape(-1)
        for i in range(size_Gamma):
            self.cduc.contents.Gamma[i] = _Gamma[i]

    def cget_B(self):
        size_B = self.n*self.cmcc
        B = np.zeros(size_B)
        for i in range(size_B):
            B[i] = self.cduc.contents.B[i]
        return B.reshape((self.n, self.cmcc))
    
    def cset_B(self, B):
        size_B = self.n*self.cmcc
        assert(B.size == size_B)
        _B = B.reshape(-1)
        for i in range(size_B):
            self.cduc.contents.B[i] = _B[i]

    def cset_beta(self, beta):
        size_beta = self.pncc
        assert(beta.size == size_beta)
        _beta = beta.reshape(-1)
        for i in range(size_beta):
            self.cduc.contents.beta[i] = _beta[i]

    def cget_H(self):
        size_H = self.n * self.p
        H = np.zeros(size_H)
        for i in range(size_H):
            H[i] = self.cduc.contents.H[i]
        return H.reshape((self.p , self.n))

    def cset_H(self, H):
        size_H = self.p*self.n
        assert(H.size == size_H)
        _H = H.reshape(-1)
        for i in range(size_H):
            self.cduc.contents.H[i] = _H[i]

    def cget_gamma(self):
        size_gamma = self.p
        gamma = np.zeros(size_gamma)
        for i in range(size_gamma):
            gamma[i] = self.cduc.contents.gamma[i]
        return gamma
    
    def cset_gamma(self, gamma):
        size_gamma = self.p
        assert(gamma.size == size_gamma)
        _gamma = gamma.reshape(-1)
        for i in range(size_gamma):
            self.cduc.contents.gamma[i] = _gamma[i]

    def cset_is_xbar_set_for_ece(self):
        self.cduc.contents.is_xbar_set_for_ece = True

    def cset_zbar(self, c_zbar, zbar):
        if self.p == 1:
            if type(zbar) != np.ndarray:
                zbar = np.array([zbar], dtype=np.float64).reshape(-1)
        _zbar = zbar.reshape(-1)
        assert(_zbar.size == self.p)
        size_zbar = _zbar.size
        for i in range(size_zbar):
            c_zbar[i] = _zbar[i]

# Template Functions. Give directions to produce python callbacks compatible with C program
def template_dynamics_update_callback(c_duc):
    # In all callbacks, unless you really know what you are doing, call this first
    # This creates an object which returns the C data arrays thorught get/set methods as nice numpy matrices / vectors
    pyduc = Py_CauchyDynamicsUpdateContainer(c_duc)
    
    # get stuff from C
    # like get the current state:
    # x_k = pyduc.cget_x()

    # do stuff 
    # ...
    # If doing a nonlinear estimation problem, make sure to do the following at minimum
    #   x_k = pyduc.cget_x()
    #   u_k = pyduc.cget_u() (if we have a control)
    #   x_k+1 = f(x_k)
    #   Phi <- compute using jacobian_f(x_k, u_k)
    #   Gamma <- compute using jacobian_f(x_k, u_k)
    #   B <- compute using jacobian_f(x_k, u_k)
    #   beta <- possibly set if time varyinig 
    #   (call) pyduc.set_is_xbar_set_for_ece() (this tells the C library you are absolutely sure that you have updated the system state)
    # ...
    # end doing stuff

    # return stuff to C
    # call pyduc.cset_"x/Phi/Gamma/B/beta" depending on what you have updated
    # for example, to set Phi, the transition matrix (newly updated by you)
    # pyduc.cset_Phi(Phi), which sets the Phi matrix on the c-side
    
    # once this function ends, the modified data is available to the C program
    foo = 5

def template_nonlinear_model(c_duc, c_zbar):
    # In all callbacks, unless you really know what you are doing, call this first
    # This creates an object which returns the C data arrays through get/set methods as nice numpy matrices / vectors
    pyduc = Py_CauchyDynamicsUpdateContainer(c_duc)

    # here we use \bar{x}_{k+1|k} to form 
    # \bar{z}_{k+1|k} = h(\bar{x}_{k+1|k})
    
    # py_zbar = h(pyduc.cget_x())

    # You must make sure to set zbar for the C program
    # pyduc.set_zbar(py_zbar, zbar)

    # once this function ends, the modified data is available to the C program
    foo = 5

def template_extended_msmt_update_callback(c_duc):
    # In all callbacks, unless you really know what you are doing, call this first
    # This creates an object which returns the C data arrays through get/set methods as nice numpy matrices / vectors
    pyduc = Py_CauchyDynamicsUpdateContainer(c_duc)
    
    # get stuff from C
    # like get the current state:
    # \bar{x}_{k+1} = pyduc.cget_x()

    # do stuff 
    # ...
    # end doing stuff
    # If doing a nonlinear estimation problem, make sure to do the following at minimum
    #   \bar{x}_{k+1} = pyduc.cget_x()
    #   H <- compute using jacobian_h(\bar{x}_{k+1})
    #   gamma <- possibly set if time varying
    # return stuff to C
    # call pyduc.cset_"H/gamma" depending on what you have updated
    # for example, to set H, the measurement matrix (newly updated by you)
    # pyduc.cset_H(H), which sets the H matrix on the c-side
    
    # once this function ends, the modified data is available to the C program
    foo = 5

# The Python Wrapper to interact with the C-side Sliding Window Manager
class PySlidingWindowManager_CPP():

    def __init__(self, mode, num_windows, num_sim_steps, log_dir = None, debug_print = True, log_seq = False, log_full = False):
        self.num_windows = int(num_windows)
        assert(num_windows < 20)
        self.num_sim_steps = int(num_sim_steps)
        self.modes = ["lti", "ltv", "nonlin"]
        if(mode.lower() not in self.modes):
            print("[Error PySlidingWindowManager:] chosen mode {} invalid. Please choose one of the following: {}".format(mode, self.modes))
        else:
            self.mode = mode.lower()
            print("Set Sliding Window Manager Mode to:", self.mode)
        self.is_initialized = False
        self.moment_info = {"x" : [], "P" :[], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "win_idx" : [], "err_code" : []} # Full window means
        self.debug_print = bool(debug_print)
        self.log_seq = bool(log_seq)
        self.log_full = bool(log_full) 
        if log_dir is None:
            self.log_dir = None 
        elif len(log_dir) == 0:
            self.log_dir = None 
        else:
            self.log_dir = log_dir
        assert(self.log_dir != "")

    def _init_params_checker(self, A0, p0, b0):
        assert(A0.shape[0] == A0.shape[1])
        assert(A0.shape[0] == p0.size)
        assert(A0.shape[0] == b0.size)
        n = A0.shape[0]
        if(n == 1):
            print("Cannot use the sliding window manager for 1-dimentional systems! You can use the PyCauchyEstimator class to run the estimator indefinately for 1D systems!")
            assert(n > 1)
        assert(np.linalg.matrix_rank(A0) == n)
        assert(np.all(p0 >= 0))
        return n
    
    def _ndim_input_checker(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma):
        
        n = self._init_params_checker(A0, p0, b0)
        assert(Phi.shape[0] == Phi.shape[1])
        assert(Phi.shape[0] == n)
        # Full rank test
        assert(np.linalg.matrix_rank(Phi) == n)
        assert(Gamma.shape[0] == n)
        assert(beta.ndim == 1)
        pncc = 0
        if Gamma is not None:
            assert(Gamma.ndim <= 2)
            if Gamma.ndim == 1:
                assert(beta.size == 1)
            else:
                assert(beta.size == Gamma.shape[1])
            
            assert(beta.ndim == 1)
            pncc = beta.size
        else:
            assert(beta is None)
        cmcc = 0
        if B is not None:
            assert(B.ndim <= 2)
            assert(B.shape[0] == n)
            if(B.ndim == 1):
                cmcc = 1
            else:
                cmcc = B.shape[1]
        p = 0
        if(H.ndim == 1):
            assert(H.ndim <= 2)
            assert(H.shape[0] == n)
            assert(gamma.size == 1)
            p = 1
        else:
            assert(H.shape[1] == n)
            assert(gamma.size == H.shape[0])
            p = gamma.size

        if( np.abs(np.any(H @ Gamma)) < 1e-12 ):
            print("Warning PySlidingWindowManager: | H @ Gamma | < eps for some input / output channels. This may result in undefined moments!")
        self.n = n
        self.pncc = pncc
        self.cmcc = cmcc
        self.p = p
        
    def _msmts_controls_checker(self, msmts, controls):
        _msmts = None
        _controls = None
        if(self.p == 1):
            if(type(msmts) != np.ndarray):
                _msmts = np.array([msmts]).reshape(-1).astype(np.float64)
                assert(_msmts.size == self.p)
            else:
                _msmts = msmts.copy().reshape(-1).astype(np.float64)
                assert(_msmts.size == self.p)
        else:
            _msmts = np.array([msmts]).reshape(-1).astype(np.float64)
            assert(_msmts.size == self.p)
        if(self.cmcc == 0):
            assert(controls is None)
            _controls = np.array([], dtype=np.float64) 
        else:
            assert(controls is not None)
            if(type(controls) != np.ndarray):
                _controls = np.array([controls]).reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
            else:
                _controls = controls.copy().reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
        return _msmts, _controls
    
    def initialize_lti(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dt=0, step=0, win_var_boost = None):
        if self.mode != "lti":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTI initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._Phi = Phi.reshape(-1).astype(np.float64)
        self._Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        self._H = H.reshape(-1).astype(np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        self._win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
        
        pycauchy.pycauchy_initialize_lti_window_manager(self.num_windows, self.num_sim_steps, self._A0, self._p0, self._b0, self._Phi, self._Gamma, self._B, self._beta, self._H, self._gamma, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, self._win_var_boost)
        self.is_initialized = True
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )

    def initialize_ltv(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, dt=0, step=0, win_var_boost = None):
        if(self.mode != "ltv"):
            print("Attempting to call initialize_ltv method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTV initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._Phi = Phi.reshape(-1).astype(np.float64)
        self._Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        self._H = H.reshape(-1).astype(np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        self._win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
                # create the dynamics_update_callback ctypes callback function
        py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        f_duc = py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(f_duc, ct.c_void_p).value

        pycauchy.pycauchy_initialize_ltv_window_manager(self.num_windows, self.num_sim_steps, self._A0, self._p0, self._b0, self._Phi, self._Gamma, self._B, self._beta, self._H, self._Gamma, self.f_duc_ptr1, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, self._win_var_boost)
        self.is_initialized = True
        print("LTV initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )
    
    def initialize_nonlin(self, x0, A0, p0, b0, beta, gamma, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, cmcc, dt=0, step=0, win_var_boost = None):
        if self.mode != "nonlin":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("NonLin initialization not successful!")
            return
        self.n = self._init_params_checker(A0, p0, b0)
        if(beta is None):
            self.pncc = 0
        else:
            assert(beta.ndim == 1)
            self.pncc = beta.size
        assert(gamma.ndim == 1)
        self.p = gamma.size
        self.cmcc = int(cmcc)
        assert(x0.size == self.n)
        # create the dynamics_update_callback ctypes callback function
        py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        f_duc = py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(f_duc, ct.c_void_p).value

        # create the nonlinear_msmt_model ctypes callback function 
        py_callback_type2 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer), ct.POINTER(ct.c_double))
        f_duc = py_callback_type2(nonlinear_msmt_model)
        self.f_duc_ptr2 = ct.cast(f_duc, ct.c_void_p).value

        # create the extended_msmt_update_callback ctypes callback function 
        py_callback_type3 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        f_duc = py_callback_type3(extended_msmt_update_callback)
        self.f_duc_ptr3 = ct.cast(f_duc, ct.c_void_p).value

        self._x0 = x0.reshape(-1).astype(np.float64)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        _cmcc = self.cmcc
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        self._win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
        
        pycauchy.pycauchy_initialize_nonlin_window_manager(self.num_windows, self.num_sim_steps, self._x0, self._A0, self._p0, self._b0, self._beta, self._gamma, self.f_duc_ptr1, self.f_duc_ptr2, self.f_duc_ptr3, _cmcc, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, self._win_var_boost)
        self.is_initialized = True
        print("Nonlin initialization successful! You can use the step(msmts, controls) method now to run the Sliding Window Manager!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )

    def step(self, msmts, controls):
        if(self.is_initialized == False):
            print("Estimator is not initialized yet. Mode set to {}. Please call method initialize_{} before running step()!".format(self.mode, self.mode))
            print("Not stepping! Please call correct method / fix mode!")
            return
        _msmts, _controls = self._msmts_controls_checker(msmts, controls)
        fz, x, P, cerr_fz, cerr_x, cerr_P, win_idx, err_code = pycauchy.pycauchy_step(_msmts, _controls)
        self.moment_info["fz"].append(fz)
        self.moment_info["x"].append(x)
        self.moment_info["P"].append(P.reshape((self.n, self.n)))
        self.moment_info["cerr_x"].append(cerr_x)
        self.moment_info["cerr_P"].append(cerr_P)
        self.moment_info["cerr_fz"].append(cerr_fz)
        self.moment_info["win_idx"] = win_idx
        self.moment_info["err_code"].append(err_code)
        
    # Shuts down sliding window manager
    def shutdown(self):
        if(self.is_initialized == False):
            print("Cannot shutdown Sliding Window Manager before it has been initialized!")
            return
        pycauchy.pycauchy_shutdown()
        print("Sliding Window Manager backend C data structure has been shutdown!")
        self.is_initialized = False
    
    def __del__(self):
        if self.is_initialized:
            self.shutdown()
            self.is_initialized = False

# The Python Wrapper to interact with the C-side Cauchy Estimator
class PyCauchyEstimator():
    def __init__(self, mode, num_steps, debug_print = True):
        self.modes = ["lti", "ltv", "nonlin"]
        if(mode.lower() not in self.modes):
            print("[Error PyCauchyEstimator:] chosen mode {} invalid. Please choose one of the following: {}".format(mode, self.modes))
        else:
            self.mode = mode.lower()
            print("Set Cauchy Estimator Mode to:", self.mode)
        self.num_steps = int(num_steps)
        assert(self.num_steps > 0)
        self.is_initialized = False
        self.moment_info = {"x" : [], "P" : [], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "err_code" : []} # Full (or best) window means
        self.debug_print = bool(debug_print)
        self.step_count = 0

    def _init_params_checker(self, A0, p0, b0):
        assert(A0.shape[0] == A0.shape[1])
        assert(A0.shape[0] == p0.size)
        assert(A0.shape[0] == b0.size)
        n = A0.shape[0]
        assert(np.linalg.matrix_rank(A0) == n)
        assert(np.all(p0 >= 0))
        return n
    
    def _ndim_input_checker(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma):
        
        n = self._init_params_checker(A0, p0, b0)
        assert(Phi.shape[0] == Phi.shape[1])
        assert(Phi.shape[0] == n)
        # Full rank test
        assert(np.linalg.matrix_rank(Phi) == n)
        assert(Gamma.shape[0] == n)
        assert(beta.ndim == 1)
        pncc = 0
        if Gamma is not None:
            assert(Gamma.ndim <= 2)
            if Gamma.ndim == 1:
                assert(beta.size == 1)
            else:
                assert(beta.size == Gamma.shape[1])
            
            assert(beta.ndim == 1)
            pncc = beta.size
        else:
            assert(beta is None)
        cmcc = 0
        if B is not None:
            assert(B.ndim <= 2)
            assert(B.shape[0] == n)
            if(B.ndim == 1):
                cmcc = 1
            else:
                cmcc = B.shape[1]
        p = 0
        if(H.ndim == 1):
            assert(H.ndim <= 2)
            assert(H.shape[0] == n)
            assert(gamma.size == 1)
            p = 1
        else:
            assert(H.shape[1] == n)
            assert(gamma.size == H.shape[0])
            p = gamma.size

        if( np.abs(np.any(H @ Gamma)) < 1e-12 ):
            print("Warning PyCauchyEstimator: | H @ Gamma | < eps for some input / output channels. This may result in undefined moments!")
        self.n = n
        self.pncc = pncc
        self.cmcc = cmcc
        self.p = p
        
    def _msmts_controls_checker(self, msmts, controls):
        _msmts = None
        _controls = None
        if(self.p == 1):
            if(type(msmts) != np.ndarray):
                _msmts = np.array([msmts]).reshape(-1).astype(np.float64)
                assert(_msmts.size == self.p)
            else:
                _msmts = msmts.copy().reshape(-1).astype(np.float64)
                assert(_msmts.size == self.p)
        else:
            _msmts = np.array([msmts]).reshape(-1).astype(np.float64)
            assert(_msmts.size == self.p)
        if(self.cmcc == 0):
            assert(controls is None)
            _controls = np.array([], dtype=np.float64) 
        else:
            assert(controls is not None)
            if(type(controls) != np.ndarray):
                _controls = np.array([controls]).reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
            else:
                _controls = controls.copy().reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
        return _msmts, _controls
    
    def _call_step(self, _msmts, _controls, full_info):
        # Estimator returns all info for all p measurements
        # the prefix '_' to variable names is used to indicate this was returned by the step function
        # for mode LTI, the dynamics are unchanging and are not returned
        if self.mode == "lti":
            _, _, _, _, _, _, \
            self._fz, self._x, self._P, \
            self._cerr_fz, self._cerr_x, self._cerr_P, self._err_code = pycauchy.pycauchy_single_step_ltiv(self.py_handle, _msmts, _controls)
        elif self.mode == "ltv":
            self._Phi, self._Gamma, self._B, self._H, self._beta, self._gamma, \
            self._fz, self._x, self._P, \
            self._cerr_fz, self._cerr_x, self._cerr_P, self._err_code = pycauchy.pycauchy_single_step_ltiv(self.py_handle, _msmts, _controls)
        else:
            self._Phi, self._Gamma, self._B, self._H, self._beta, self._gamma, \
            self._fz, self._x, self._P, self._xbar, self._zbar, \
            self._cerr_fz, self._cerr_x, self._cerr_P, self._err_code = pycauchy.pycauchy_single_step_nonlin(self.py_handle, _msmts, _controls, self.step_count != 0)

        if full_info:
            xs = []
            Ps = []
            for i in range(self.p):
                xs.append( self._x[i*self.n:(i+1)*self.n].copy() )
                Ps.append( self._P[i*self.n*self.n:(i+1)*self.n*self.n].copy().reshape((self.n, self.n)) )
        else:
            xs = self._x[-self.n:].copy()
            Ps = self._P[-self.n*self.n:].copy().reshape((self.n,self.n))

        fz = self._fz[-1]
        x = self._x[-self.n:].copy()
        P = self._P[-self.n*self.n:].copy()
        cerr_fz = self._cerr_fz[-1]
        cerr_x = self._cerr_x[-1]
        cerr_P = self._cerr_P[-1]
        err_code = self._err_code[-1]

        self.moment_info["fz"].append(fz)
        self.moment_info["x"].append(x)
        self.moment_info["P"].append(P.reshape((self.n, self.n)))
        self.moment_info["cerr_x"].append(cerr_x)
        self.moment_info["cerr_P"].append(cerr_P)
        self.moment_info["cerr_fz"].append(cerr_fz)
        self.moment_info["err_code"].append(err_code)

        self.step_count += 1
        return xs, Ps
    
    def initialize_lti(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, init_step=0, dt=0):
        if self.mode != "lti":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTI initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._Phi = Phi.reshape(-1).astype(np.float64)
        self._Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        self._H = H.reshape(-1).astype(np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)
        _init_step = int(init_step)
        _dt = float(dt)
        
        self.py_handle = pycauchy.pycauchy_initialize_lti(self.num_steps, self._A0, self._p0, self._b0, self._Phi, self._Gamma, self._B, self._beta, self._H, self._gamma, _dt, _init_step, self.debug_print)
        self.is_initialized = True
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: You can call the step function {} time-steps, {} measurements per step == {} total times!".format(self.num_steps, self.p, self.num_steps * self.p) )

    def initialize_ltv(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, init_step=0, dt=0):
        if(self.mode != "ltv"):
            print("Attempting to call initialize_ltv method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTV initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._Phi = Phi.reshape(-1).astype(np.float64)
        self._Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        self._H = H.reshape(-1).astype(np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)

        # create the dynamics_update_callback ctypes callback function
        self.py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        self.f_duc = self.py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(self.f_duc, ct.c_void_p).value
        _init_step = int(init_step)
        _dt = float(dt)

        self.py_handle = pycauchy.pycauchy_initialize_ltv(self.num_steps, self._A0, self._p0, self._b0, self._Phi, self._Gamma, self._B, self._beta, self._H, self._gamma, self.f_duc_ptr1, _dt, _init_step, self.debug_print)
        self.is_initialized = True
        print("LTV initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: You can call the step function {} time-steps, {} measurements per step == {} total times!".format(self.num_steps, self.p, self.num_steps * self.p) )
    
    def initialize_nonlin(self, x0, A0, p0, b0, beta, gamma, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, cmcc, dt=0, step=0):
        if self.mode != "nonlin":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("NonLin initialization not successful!")
            return
        self.n = self._init_params_checker(A0, p0, b0)
        if(beta is None):
            self.pncc = 0
        else:
            assert(beta.ndim == 1)
            self.pncc = beta.size
        assert(gamma.ndim == 1)
        self.p = gamma.size
        self.cmcc = int(cmcc)
        assert(x0.size == self.n)
        # create the dynamics_update_callback ctypes callback function
        self.py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        self.f_duc1 = self.py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(self.f_duc1, ct.c_void_p).value

        # create the nonlinear_msmt_model ctypes callback function 
        self.py_callback_type2 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer), ct.POINTER(ct.c_double))
        self.f_duc2 = self.py_callback_type2(nonlinear_msmt_model)
        self.f_duc_ptr2 = ct.cast(self.f_duc2, ct.c_void_p).value

        # create the extended_msmt_update_callback ctypes callback function 
        self.py_callback_type3 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        self.f_duc3 = self.py_callback_type3(extended_msmt_update_callback)
        self.f_duc_ptr3 = ct.cast(self.f_duc3, ct.c_void_p).value

        self._x0 = x0.reshape(-1).astype(np.float64)
        self._A0 = A0.reshape(-1).astype(np.float64)
        self._p0 = p0.reshape(-1).astype(np.float64)
        self._b0 = b0.reshape(-1).astype(np.float64)
        self._beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        self._gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        _cmcc = self.cmcc
        
        self.py_handle = pycauchy.pycauchy_initialize_nonlin(self.num_steps, self._x0, self._A0, self._p0, self._b0, self._beta, self._gamma, self.f_duc_ptr1, self.f_duc_ptr2, self.f_duc_ptr3, _cmcc, _dt, _step, self.debug_print)
        self.is_initialized = True
        print("Nonlin initialization successful! You can use the step(msmts, controls) method now to run the Cauchy Estimtor!")
        print("Note: You can call the step function {} time-steps, {} measurements per step == {} total times!".format(self.num_steps, self.p, self.num_steps * self.p) )

    # full_info=True returns list of moments for each measurement processed
    # full_info=False returns moments after processing all measurements
    def step(self, msmts, controls = None, full_info=False):
        full_info = bool(full_info)

        if(self.is_initialized == False):
            print("Estimator is not initialized yet. Mode set to {}. Please call method initialize_{} before running step()!".format(self.mode, self.mode))
            print("Not stepping! Please call correct method / fix mode!")
            return
        if(self.step_count == self.num_steps):
            print("[Error:] Cannot step estimator again, you have already stepped the estimator the initialized number of steps")
            print("Not stepping! Please shut estimator down or reset it!")
            return
        _msmts, _controls = self._msmts_controls_checker(msmts, controls)
        self._msmts = _msmts.copy()
        self._controls = _controls.copy()
        return self._call_step(_msmts, _controls, full_info)

    # NOT FINISHED -- Should be used as a way to either
    # 1.) Step only time propagation routine
    # 2.) Step only measurement update routine
    # 3.) Step measurements asynchronously as they are recieved
    # Applies controls (time propagates) only if its the first measurement of the time-step
    def step_asynchronous(self, msmts, controls = None):
        '''
        # If its the first measurement at the current time step
        if( (self.step_count % self.p) == 0 ):
            self._msmts = np.array([], dtype = np.float64)
            # check to see whether a control should be given
            if self.cmcc > 0:
                # if a control should be provided
                assert(controls is not None)
                # check to make sure the control vector size is correct
                _controls = np.array([controls]).copy().reshape(-1)
                # check to make sure the control vector size is correct
                assert(_controls.size == self.cmcc)
            else:
                assert(controls is None)
                _controls = np.array([], dtype=np.float64)
        # If its not the first measurement at the current time step
        else:
            if self.cmcc > 0:
                if controls is not None:
                    print("Warning step_single_msmt: This is not the first msmt update at the current step! Your controls will not be applied!")
            else:
                if controls is not None:
                    print("Error step_single_msmt: Cannot apply a control vector if you specify cmcc=", self.cmcc)
                    assert(controls is None)
            _controls = np.array([], dtype=np.float64)
        _msmt = np.array([msmt]).copy().reshape(-1)
        assert(_msmt.size == 1)
        '''
        print("WARN step_scalar_msmt: this function has not been fully implemented! Please do so! Exiting!")
        exit(1)

    def get_last_mean_cov(self):
        return self.moment_info["x"][-1], self.moment_info["P"][-1]

    # find reinitialization parameters for the msmt_idx-th measurement just computed
    # if msmt_idx not specified, uses the last measurement
    def get_reinitialization_statistics(self, msmt_idx = -1):
        if( (self.step_count == 0) or (self.is_initialized == False) ):
            print("[Error get_reinitialization_statistics]: Cannot find reinitialization stats of an estimator not initialized, or that has not processed at least one measurement! Please correct!")
            return None, None, None
        if( (msmt_idx >= self.p) or (msmt_idx < -self.p) ):
            print("[Error get_reinitialization_statistics]: Cannot find reinitialization stats for msmt_idx={}. The index is out of range -{} <= msmt_idx < {}...(max index is p-1={})! Please correct!".format(msmt_idx, -self.p, self.p-1, self.p-1))
            return None, None, None

        msmt_idx = int(msmt_idx)
        if(msmt_idx < 0):
            msmt_idx += self.p
        
        reinit_msmt = self._msmts[msmt_idx]
        reinit_xhat = self._x[msmt_idx*self.n : (msmt_idx+1)*self.n].copy()
        reinit_Phat = self._P[msmt_idx*self.n*self.n : (msmt_idx+1)*self.n*self.n].copy()
        reinit_H = self._H[msmt_idx*self.n : (msmt_idx+1)*self.n].copy()
        reinit_gamma = self._gamma[msmt_idx]
        if self.mode != "nonlin":
            A0, p0, b0 = pycauchy.pycauchy_get_reinitialization_statistics(self.py_handle, reinit_msmt, reinit_xhat, reinit_Phat, reinit_H, reinit_gamma)
            A0 = A0.reshape( (self.n, self.n) )
            return A0, p0, b0
        else:
            reinit_xbar = self._xbar[msmt_idx*self.n:(msmt_idx+1)*self.n].copy()
            reinit_zbar = self._zbar[msmt_idx]
            dx = reinit_xhat - reinit_xbar
            dz = reinit_msmt - reinit_zbar
            A0, p0, b0 = pycauchy.pycauchy_get_reinitialization_statistics(self.py_handle, dz, dx, reinit_Phat, reinit_H, reinit_gamma)
            A0 = A0.reshape( (self.n, self.n) )
            return A0, p0, b0, reinit_xbar
    
    # provide optional arguments A0, p0, b0, xbar if you'd like these to change upon reset
    def reset(self, A0 = None, p0 = None, b0 = None, xbar = None):
        if(self.is_initialized == False):
            print("Cannot reset estimator before it has been initialized (or after shutdown has been called)!")
            return
        self.step_count = 0
        if A0 is not None:
            assert(A0.size == self.n*self.n)
        if p0 is not None:
            assert(p0.size == self.n)
        if b0 is not None:
            assert(b0.size == self.n)
        if xbar is not None:
            assert(xbar.size == self.n)
        if (self.mode != "nonlin") and (xbar is not None):
            print("Note to user: Setting xbar for any mode besides 'nonlinear' will have no effect!")
        self._A0 = A0.copy().reshape(-1) if A0 is not None else np.array([], dtype=np.float64)
        self._p0 = p0.copy().reshape(-1) if p0 is not None else np.array([], dtype=np.float64)
        self._b0 = b0.copy().reshape(-1) if b0 is not None else np.array([], dtype=np.float64)
        self._xbar = xbar.copy().reshape(-1) if xbar is not None else np.array([], dtype=np.float64)
        pycauchy.pycauchy_single_step_reset(self.py_handle, self._A0, self._p0, self._b0, self._xbar)

    # Used to process a single measurement after reinitializing estimator
    # Resets this estimator about the msmt_idx-th measurement the other estimator has just processed
    def reset_about_estimator(self, other_estimator, msmt_idx = -1):
        if self.is_initialized == False:
            print("[Error reset_about_estimator:] This estimator is not initialized! Must initialize the estimator before using this function!")
        if other_estimator.is_initialized == False:
            print("[Error reset_about_estimator:] Other estimator is not initialized! Must initialize the other estimator (and step it) before using this function!")
        if other_estimator.step_count == 0:
            print("[Error reset_about_estimator:] Other estimator has step_count == 0 (step_count must be > 0). The inputted estimator must be stepped before using this function!")
            assert(False)
        if id(self) == id(other_estimator):
            print("[Error reset_about_estimator:] Other estimator cannot be this estimator itself!")
            assert(False)
        if(self.p != other_estimator.p):
            print("[Error reset_about_estimator:] Both estimators must process the same number of measurements! this={}, other={}".format(self.p, other_estimator.p))
            assert(False)
        if(self.mode != other_estimator.mode):
            print("[Error reset_about_estimator:] Both estimators must have same mode! this={}, other={}".format(self.mode, other_estimator.mode))
            assert(False)
        if( (msmt_idx >= self.p) or (msmt_idx < -self.p) ):
            print("[Error reset_about_estimator:] Specified msmt_idx={}. The index is out of range -{} <= msmt_idx < {}...(max index is p-1={})! Please correct!".format(msmt_idx, -self.p, self.p-1, self.p-1))
            assert(False)
        msmt_idx = int(msmt_idx)
        if(msmt_idx < 0):
            msmt_idx += self.p
        _msmts = other_estimator._msmts[msmt_idx:].copy()
        self._msmts = _msmts.copy()
        if self.mode != "nonlin":
            A0, p0, b0 = other_estimator.get_reinitialization_statistics(msmt_idx)
            self.reset(A0, p0, b0)
        else:
            A0, p0, b0, xbar = other_estimator.get_reinitialization_statistics(msmt_idx)
            self.reset(A0, p0, b0, xbar)
        xs, Ps = self._call_step(_msmts, np.array([], dtype=np.float64), False)
        pycauchy.pycauchy_single_step_set_master_step(self.py_handle, self.p)
        return xs, Ps
    
    def reset_with_last_measurement(self, z_scalar, A0, p0, b0, xbar):
        _z_scalar = np.array(z_scalar).copy().reshape(-1)
        if self.mode != "nonlin":
            self.reset(A0, p0, b0)
        else:
            self.reset(A0, p0, b0, xbar)
        xs, Ps = self._call_step(_z_scalar, np.array([], dtype=np.float64), False)
        pycauchy.pycauchy_single_step_set_master_step(self.py_handle, self.p)
        return xs, Ps

    def set_window_number(self, win_num):
        win_num = int(win_num)
        pycauchy.pycauchy_single_step_set_window_number(self.py_handle, win_num)
    
    # Shuts down estimator
    def shutdown(self):
        if(self.is_initialized == False):
            print("Cannot shutdown Cauchy Estimator before it has been initialized!")
            return
        pycauchy.pycauchy_single_step_shutdown(self.py_handle)
        self.py_handle = None
        print("Cauchy estimator backend C data structure has been shutdown!")
        self.is_initialized = False
     
    def get_pyduc(self):
        self._c_pyduc_voidp = pycauchy.pycauchy_single_step_get_duc(self.py_handle)
        self.c_pyduc_type = ct.POINTER(C_CauchyDynamicsUpdateContainer)
        self.c_pyduc = ct.cast(int(self._c_pyduc_voidp), self.c_pyduc_type)
        self.pyduc = Py_CauchyDynamicsUpdateContainer(self.c_pyduc)
        return self.pyduc

    def get_marginal_2D_pointwise_cpdf(self, marg_idx1, marg_idx2, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir = None):
        if(self.is_initialized == False):
            print("Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!")
            return None,None,None
        if(self.step_count < 1):
            print("Cannot evaluate Cauchy Estimator Marginal 2D CPDF before it has been stepped!")
            return None,None,None
        _marg_idx1 = int(marg_idx1)
        _marg_idx2 = int(marg_idx2)
        _gridx_low = float(gridx_low)
        _gridx_high = float(gridx_high)
        _gridx_resolution = float(gridx_resolution)
        _gridy_low = float(gridy_low)
        _gridy_high = float(gridy_high)
        _gridy_resolution = float(gridy_resolution)
        assert(_gridx_high > _gridx_low)
        assert(_gridy_high > _gridy_low)
        assert(_gridx_resolution > 0)
        assert(_gridy_resolution > 0)
        assert((_marg_idx1 > -1) and (_marg_idx1 < _marg_idx2) and (_marg_idx2 < self.n))
        if log_dir is None:
            _log_dir = None
        else:
            _log_dir = str(log_dir)
            if _log_dir == "":
                _log_dir = None
            elif _log_dir[-1] == "/":
                _log_dir = log_dir[:-1]

        cpdf_points, num_gridx, num_gridy = pycauchy.pycauchy_get_marginal_2D_pointwise_cpdf(self.py_handle, _marg_idx1, _marg_idx2, _gridx_low, _gridx_high, _gridx_resolution, _gridy_low, _gridy_high, _gridy_resolution, _log_dir)
        cpdf_points = cpdf_points.reshape(num_gridx*num_gridy, 3)
        X = cpdf_points[:,0].reshape( (num_gridy, num_gridx) )
        Y = cpdf_points[:,1].reshape( (num_gridy, num_gridx) )
        Z = cpdf_points[:,2].reshape( (num_gridy, num_gridx) )
        return X, Y, Z

    def get_2D_pointwise_cpdf(self, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir = None):
        if(self.is_initialized == False):
            print("Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!")
            return None,None,None
        if(self.n != 2):
            print("Cannot evaluate Cauchy Estimator 2D CPDF for a {}-state system!".format(self.n))
            return None,None,None
        if(self.step_count < 1):
            print("Cannot evaluate Cauchy Estimator 2D CPDF before it has been stepped!")
            return None,None,None
        return self.get_marginal_2D_pointwise_cpdf(0, 1, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir)
    
    def get_marginal_1D_pointwise_cpdf(self, marg_idx, gridx_low, gridx_high, gridx_resolution, log_dir = None):
        if(self.is_initialized == False):
            print("Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!")
            return None,None
        if(self.step_count < 1):
            print("Cannot evaluate Cauchy Estimator 1D Marginal CPDF before it has been stepped!")
            return None,None

        _marg_idx = int(marg_idx)
        _gridx_low = float(gridx_low)
        _gridx_high = float(gridx_high)
        _gridx_resolution = float(gridx_resolution)
        assert(_gridx_high > _gridx_low)
        assert(_gridx_resolution > 0)
        assert((_marg_idx > -1) and (_marg_idx < self.n))
        if log_dir is None:
            _log_dir = None
        else:
            _log_dir = str(log_dir)
            if _log_dir == "":
                _log_dir = None
            elif _log_dir[-1] == "/":
                _log_dir = log_dir[:-1]

        cpdf_points, num_gridx = pycauchy.pycauchy_get_marginal_1D_pointwise_cpdf(self.py_handle, _marg_idx, _gridx_low, _gridx_high, _gridx_resolution, _log_dir)
        cpdf_points = cpdf_points.reshape(num_gridx, 2)
        X = cpdf_points[:,0]
        Y = cpdf_points[:,1]
        return X, Y

    def get_1D_pointwise_cpdf(self, gridx_low, gridx_high, gridx_resolution, log_dir = None):
        if(self.is_initialized == False):
            print("Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!")
            return None,None
        if(self.n != 1):
            print("Cannot evaluate Cauchy Estimator 1D CPDF for a {}-state system!".format(self.n))
            return None,None
        if(self.step_count < 1):
            print("Cannot evaluate Cauchy Estimator 1D CPDF before it has been stepped!")
            return None,None
        return self.get_marginal_1D_pointwise_cpdf(0, gridx_low, gridx_high, gridx_resolution, log_dir)
 
    def plot_2D_pointwise_cpdf(self, X, Y, Z, state_labels = (1,2)):
        GRID_HEIGHT = 8
        GRID_WIDTH = 2
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        #plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath},{amssymb}}'

        fig = plt.figure(figsize=(15,11))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-1,:],projection='3d')
        #ax.set_box_aspect((1, 1, 1), zoom= 1.1)
        plt.tight_layout()
        
        _ = ax.plot_wireframe(X, Y, Z, color='b', label=r"Cauchy Estimator's CPDF: \text{  }$f_{X_k|Y_k}(x_k|y_k)$")
        #surf = ax.plot_surface(X,Y,Z, color='b',linewidth=0)

        # Add State Markers
        z_height = 0
        cauchy_state = self.moment_info["x"][-1]
        #ax.scatter(true_state[0], true_state[1], z_height, label=r"True State of Dynamic System: $x_k$", color='r', marker='^', s=100)
        ax.scatter(cauchy_state[0], cauchy_state[1], z_height, label=r"Cauchy Estimator's State Hypothesis: $\hat{x}_k$", color='b', marker='^', s=100)
        #ax.scatter(kf_state[0], kf_state[1], z_height, color='g', label=r"Kalman Filter's State Hypothesis: $\hat{x}_k$", marker='^', s=100)
        
        #ell_kf = ellipse_points(kf_state, kf_covar, 1)
        #ell_cauchy = ellipse_points(cauchy_state, cauchy_covar, 1)
        #ax.plot(ell_kf[:,0], ell_kf[:,1], zs=z_height, zdir='z', color='m', label="Kalman Filter's 70\\% Confidence Ellipsoid", linewidth = 4)
        #ax.plot(ell_cauchy[:,0], ell_cauchy[:,1], zs=z_height, zdir='z', color='tab:orange', label="Cauchy Estimator's 70\\% Confidence Ellipsoid", linewidth = 4)
        
        ax.set_xlabel("x-axis (State-{})".format(state_labels[0]), fontsize=14)
        ax.set_ylabel("y-axis (State-{})".format(state_labels[1]), fontsize=14)
        ax.set_zlabel("z-axis (CPDF Probability)", rotation=180, fontsize=14)
        #ax.legend(loc=2, bbox_to_anchor=(-.52, 1), fontsize=14)
        plt.show()

    def plot_1D_pointwise_cpdf(self, x, y, state_label=1):
        plt.plot(x,y)
        plt.xlabel("State-{}".format(state_label))
        plt.ylabel("CPDF Probability")
        plt.show()
    
    def __del__(self):
        if self.is_initialized:
            self.shutdown()
            self.is_initialized = False

# The Python Wrapper to interact with a Sliding Window Manager Object without the pipes and forking on the C-side
# This is slower than PySlidingWindowManager_CPP but doesnt come with all the uneeded baggage for debugging
class PySlidingWindowManager():

    def __init__(self, mode, num_windows, swm_debug_print = True, win_debug_print = False):
        self.num_windows = int(num_windows)
        assert(num_windows < 20)
        self.modes = ["lti", "ltv", "nonlin"]
        if(mode.lower() not in self.modes):
            print("[Error PySlidingWindowManager:] chosen mode {} invalid. Please choose one of the following: {}".format(mode, self.modes))
        else:
            self.mode = mode.lower()
            print("Set Sliding Window Manager Mode to:", self.mode)
        self.is_initialized = False
        self.moment_info = {"x" : [], "P" :[], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "win_idx" : [], "err_code" : []} # Full window means
        self.avg_moment_info = {"x" : [], "P" :[], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "win_idx" : [], "err_code" : []} # Full window means
        self.debug_print = bool(swm_debug_print)
        self.step_idx = 0
        self.win_idxs = np.arange(self.num_windows)
        self.win_counts = np.zeros(self.num_windows, dtype=np.int64)
        self.cauchyEsts = [PyCauchyEstimator(self.mode, self.num_windows, bool(win_debug_print) ) for _ in range(self.num_windows)]

    def _set_dimensions(self):
        self.n = self.cauchyEsts[0].n
        self.cmcc = self.cauchyEsts[0].cmcc
        self.pncc = self.cauchyEsts[0].pncc
        self.p = self.cauchyEsts[0].p

    def initialize_lti(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dt=0, step=0, reinit_func = None):
        if p0.size == 1:
            print("[Error PySlidingWindowManager:] Do not use this class for scalar systems! This is only for systems of dimension >1 Use the PyCauchyEstimator class instead!")
            exit(1)
        if self.mode != "lti":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("NonLin initialization not successful!")
            return
        self.reinit_func = reinit_func
        _step = int(step)
        _dt = float(dt)
        for i in range(self.num_windows):
            self.cauchyEsts[i].initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, _step + i, _dt)
            self.cauchyEsts[i].set_window_number(i+1)
        self.is_initialized = True
        self._set_dimensions()
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )

    def initialize_ltv(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, dt=0, step=0, reinit_func = None):
        if p0.size == 1:
            print("[Error PySlidingWindowManager:] Do not use this class for scalar systems! This is only for systems of dimension >1 Use the PyCauchyEstimator class instead!")
            exit(1)
        if self.mode != "ltv":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("NonLin initialization not successful!")
            return
        self.reinit_func = reinit_func
        _dt = float(dt)
        _step = int(step)
        for i in range(self.num_windows):
            self.cauchyEsts[i].initialize_ltv(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, _step + i, _dt)
            self.cauchyEsts[i].set_window_number(i+1)        

        self.is_initialized = True
        self._set_dimensions()
        print("LTV initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )
    
    def initialize_nonlin(self, x0, A0, p0, b0, beta, gamma, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, cmcc, dt=0, step=0, reinit_func = None):
        if p0.size == 1:
            print("[Error PySlidingWindowManager:] Do not use this class for scalar systems! This is only for systems of dimension >1 Use the PyCauchyEstimator class instead!")
            exit(1)
        if self.mode != "nonlin":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("NonLin initialization not successful!")
            return
        self.reinit_func = reinit_func
        _dt = float(dt)
        _step = int(step)
        for i in range(self.num_windows):
            self.cauchyEsts[i].initialize_nonlin(x0, A0, p0, b0, beta, gamma, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, cmcc, _dt, _step + i)
            self.cauchyEsts[i].set_window_number(i+1)     

        self.is_initialized = True
        self._set_dimensions()
        print("Nonlin initialization successful! You can use the step(msmts, controls) method now to run the Sliding Window Manager!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )

    def _best_window_est(self):
        W = self.num_windows
        okays = np.zeros(W, dtype=np.bool8)
        idxs = []
        check_idx = self.p-1
        COV_UNSTABLE = 2
        COV_DNE = 8
        for i in range(W):
            if(self.win_counts[i] > 0):
                err = self.cauchyEsts[i]._err_code[check_idx]
                if (err & COV_UNSTABLE) or (err & COV_DNE):
                    pass
                else:
                    idxs.append((i, self.win_counts[i]))
                    okays[i] = True
        if self.step_idx == 0:
            best_idx = 0
            okays[0] = True
        else:
            if(len(idxs) == 0):
                print("No window is available without an error code!")
                exit(1)
            sorted_idxs = list(reversed(sorted(idxs, key = lambda x : x[1])))
            best_idx = sorted_idxs[0][0]

        n = self.n
        best_estm = self.cauchyEsts[best_idx]
        self.moment_info["fz"].append(best_estm._fz[check_idx])
        self.moment_info["x"].append(best_estm._x[check_idx*n:].copy())
        self.moment_info["P"].append(best_estm._P[check_idx*n*n:].copy().reshape((n,n)))
        self.moment_info["cerr_x"].append(best_estm._cerr_x[check_idx])
        self.moment_info["cerr_P"].append(best_estm._cerr_P[check_idx])
        self.moment_info["cerr_fz"].append(best_estm._cerr_fz[check_idx])
        self.moment_info["win_idx"].append(best_idx)
        self.moment_info["err_code"].append(best_estm._err_code[check_idx])
        return best_idx, okays
    
    def _weighted_average_win_est(self, usable_wins):
        n = self.n
        last_idx = self.p-1
        win_avg_mean = np.zeros(n)
        win_avg_cov = np.zeros((n,n))
        win_avg_fz = 0
        win_avg_cerr_fz = 0 
        win_avg_cerr_x = 0
        win_avg_cerr_P = 0
        win_avg_err_code = 0
        win_norm_fac = 0.0
        for i in range(self.num_windows):
            win_count = self.win_counts[i]
            if self.win_counts[i] > 0:
                win_okay = usable_wins[i]
                if win_okay:
                    est = self.cauchyEsts[i]
                    norm_fac = win_count / self.num_windows
                    win_norm_fac += norm_fac
                    x, P = est.get_last_mean_cov()
                    win_avg_mean += x * norm_fac
                    win_avg_cov += P * norm_fac
                    win_avg_fz += est._fz[last_idx] * norm_fac 
                    win_avg_cerr_fz += est._cerr_fz[last_idx] * norm_fac
                    win_avg_cerr_x += est._cerr_x[last_idx] * norm_fac
                    win_avg_cerr_P += est._cerr_P[last_idx] * norm_fac
                    win_avg_err_code |= est._err_code[last_idx]
        win_avg_mean /= win_norm_fac
        win_avg_cov /= win_norm_fac
        win_avg_fz /= win_norm_fac
        win_avg_cerr_fz /= win_norm_fac
        win_avg_cerr_x /= win_norm_fac
        win_avg_cerr_P /= win_norm_fac
        self.avg_moment_info["fz"].append(win_avg_fz)
        self.avg_moment_info["x"].append(win_avg_mean.copy())
        self.avg_moment_info["P"].append(win_avg_cov.copy())
        self.avg_moment_info["cerr_x"].append(win_avg_cerr_x)
        self.avg_moment_info["cerr_P"].append(win_avg_cerr_P)
        self.avg_moment_info["cerr_fz"].append(win_avg_cerr_fz)
        self.avg_moment_info["win_idx"].append(-1)
        self.avg_moment_info["err_code"].append(win_avg_err_code)
        return win_avg_mean, win_avg_cov

    def step(self, msmts, controls = None, reinit_args = None):
        if(self.is_initialized == False):
            print("Estimator is not initialized yet. Mode set to {}. Please call method initialize_{} before running step()!".format(self.mode, self.mode))
            print("Not stepping! Please call correct method / fix mode!")
            return
        if self.debug_print:
            print("SWM: Step ", self.step_idx)
        # If we are on first estimation step, step the first window
        if self.step_idx == 0:
            if self.debug_print:
                print("  Window {} is on step {}/{}".format(1, 1, self.num_windows) )
            self.cauchyEsts[0].step(msmts, controls)
            self.win_counts[0] += 1
        else:
            # Step all windows that are not uninitialized
            idx_max = np.argmax(self.win_counts)
            idx_min = np.argmin(self.win_counts)
            for win_idx, win_count in zip(self.win_idxs, self.win_counts):
                if(win_count > 0):
                    if self.debug_print:
                        print("  Window {} is on step {}/{}".format(win_idx+1, win_count+1, self.num_windows) )
                    self.cauchyEsts[win_idx].step(msmts, controls)
                    self.win_counts[win_idx] += 1
        
        # Find best window, its estimates, and the weighted average of the estimation results
        best_idx, usable_wins = self._best_window_est()
        wavg_xhat, wavg_Phat = self._weighted_average_win_est(usable_wins)
        xhat, Phat = self.cauchyEsts[best_idx].get_last_mean_cov()
        
        # Reinitialize the empty window
        if self.step_idx > 0:
            if self.reinit_func is not None:
                self.reinit_func(self.cauchyEsts, best_idx, usable_wins, self.win_counts.copy(), reinit_args)
            else:
                if reinit_args is not None:
                    print("  [Warn PySlidingWindowManager:] Providing reinit_args with no reinit_func given will do nothing!")
                # using speyer's start method if no user reinit function provided
                speyer_restart_idx = self.p-1
                self.cauchyEsts[idx_min].reset_about_estimator(self.cauchyEsts[best_idx], msmt_idx = speyer_restart_idx)
                if self.debug_print:
                    print("  Window {} reinitializes Window {}".format(best_idx+1, idx_min+1))
            # reset full estimator, increment reinitialized estimator count
            self.win_counts[idx_min] += 1
            if(self.win_counts[idx_max] == self.num_windows):
                self.cauchyEsts[idx_max].reset()
                self.win_counts[idx_max] = 0
        # Increment step index
        self.step_idx += 1
        # return the best estimate as well as the averaged conditional estimates 
        return xhat, Phat, wavg_xhat, wavg_Phat
        
    # Shuts down sliding window manager
    def shutdown(self):
        if(self.is_initialized == False):
            print("Cannot shutdown Sliding Window Manager before it has been initialized!")
            return
        for i in range(self.num_windows):
            self.cauchyEsts[i].shutdown()
        self.win_counts = np.zeros(self.num_windows, dtype=np.int64)
        print("Sliding Window Manager has been shutdown!")
        self.is_initialized = False
        self.step_idx = 0
    
    def __del__(self):
        if self.is_initialized:
            self.shutdown()
            self.is_initialized = False


# Reads and parses window data 
def load_window_data(f_win_data):
    file = open(f_win_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(":")[1].split(" ")] for line in lines])

# Reads and parses Kalman Filter Data
def load_data(f_data):
    file = open(f_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(" ")] for line in lines])

def load_cauchy_log_folder(log_dir, with_win_logging = True):
    log_dir = log_dir + "/" if log_dir[-1] != "/" else log_dir
    f_means = "cond_means.txt"
    f_covars = "cond_covars.txt"
    f_err_means = "cerr_cond_means.txt"
    f_err_covars = "cerr_cond_covars.txt"
    f_err_norm_factors = "cerr_norm_factors.txt"

    # Load in the data points
    means = load_window_data(log_dir + f_means) if with_win_logging else load_data(log_dir + f_means)
    print("Means: ", means.shape)

    covars = load_window_data(log_dir + f_covars) if with_win_logging else load_data(log_dir + f_covars)
    print("Covars: ", covars.shape)
    n = int(np.sqrt(covars.shape[1]))
    covars = covars.reshape((covars.shape[0], n,n))
    print("Covars after Reshaping: ", covars.shape)

    cerr_means = load_window_data(log_dir + f_err_means) if with_win_logging else load_data(log_dir + f_err_means)
    print("Cerr Means: ", cerr_means.shape)
    cerr_covars = load_window_data(log_dir + f_err_covars) if with_win_logging else load_data(log_dir + f_err_covars)
    print("Cerr Covar: ", cerr_covars.shape)
    cerr_norm_factors = load_window_data(log_dir + f_err_norm_factors) if with_win_logging else load_data(log_dir + f_err_norm_factors)
    print("Cerr Means: ", cerr_norm_factors.shape)

    return {"x": means, "P": covars, "cerr_x": cerr_means, "cerr_P": cerr_covars, "cerr_fz": cerr_norm_factors}

def load_kalman_log_folder(log_dir):
    log_dir = log_dir + "/" if log_dir[-1] != "/" else log_dir
    f_means = "kf_cond_means.txt"
    f_covars = "kf_cond_covars.txt"
    xs_kf = load_data(log_dir + f_means)
    Ps_kf = load_data(log_dir + f_covars)
    T = Ps_kf.shape[0]
    n = int( np.sqrt(Ps_kf[0].size) + 0.99 )
    return (xs_kf, Ps_kf.reshape((T,n,n)))

def load_sim_truth_log_folder(log_dir):
    log_dir = log_dir + "/" if log_dir[-1] != "/" else log_dir
    f_msmt_noises = "msmt_noises.txt"
    f_proc_noises = "proc_noises.txt"
    f_true_states = "true_states.txt"
    f_msmts = "msmts.txt"
    true_states = np.loadtxt(log_dir + f_true_states)
    if true_states.ndim == 1:
        true_states = true_states.reshape((true_states.size,1))
    print("True States: ", true_states.shape)

    msmts = np.loadtxt(log_dir + f_msmts)
    msmt_noises = np.loadtxt(log_dir + f_msmt_noises)
    proc_noises = np.loadtxt(log_dir + f_proc_noises)
    if(msmts.ndim == 1):
        msmts = msmts.reshape((msmts.size,1))
    if(proc_noises.ndim == 1):
        proc_noises = proc_noises.reshape((proc_noises.size,1))
    if(msmt_noises.ndim == 1):
        msmt_noises = msmt_noises.reshape((msmt_noises.size,1))
    print("Msmts: ", msmts.shape)
    print("Msmt Noises: ", msmt_noises.shape)
    print("Proc Noises: ", proc_noises.shape)
    return true_states, msmts, proc_noises, msmt_noises

def log_sim_truth(log_dir, xs, zs, ws, vs):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if log_dir[-1] != "/":
        log_dir += "/"
    np.savetxt(log_dir+"true_states.txt", xs, delimiter= " ")
    np.savetxt(log_dir+"msmts.txt", zs, delimiter= " ")
    np.savetxt(log_dir+"proc_noises.txt", ws, delimiter= " ")
    np.savetxt(log_dir+"msmt_noises.txt", vs,  delimiter= " ")

def log_kalman(log_dir, xs_kf, Ps_kf):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if log_dir[-1] != "/":
        log_dir += "/"
    T = xs_kf.shape[0]
    n = xs_kf.shape[1]
    np.savetxt(log_dir+"kf_cond_means.txt", xs_kf, delimiter= " ")
    np.savetxt(log_dir+"kf_cond_covars.txt", Ps_kf.reshape(T, n*n), delimiter= " ")

def log_cauchy(log_dir, moment_dic):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if log_dir[-1] != "/":
        log_dir += "/"
    T = len(moment_dic["x"])
    n = moment_dic["x"][0].size
    np.savetxt(log_dir+"cond_means.txt", np.array(moment_dic["x"]), delimiter= " ")
    np.savetxt(log_dir+"cond_covars.txt", np.array(moment_dic["P"].reshape(T,n*n)), delimiter= " ")
    np.savetxt(log_dir+"norm_factors.txt", np.array(moment_dic["fz"]), delimiter= " ")
    np.savetxt(log_dir+"cerr_cond_means.txt", np.array(moment_dic["cerr_x"]), delimiter= " ")
    np.savetxt(log_dir+"cerr_cond_covars.txt", np.array(moment_dic["cerr_P"]), delimiter= " ")
    np.savetxt(log_dir+"cerr_norm_factors.txt", np.array(moment_dic["cerr_fz"]), delimiter= " ")
    np.savetxt(log_dir+"numeric_error_codes.txt", np.array(moment_dic["err_code"]), delimiter= " ")
    
def simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt = True, dynamics_update_callback = None, other_params = None):
    if(B is not None):
        assert(us is not None)
        assert(us.shape[0] == num_steps)
        if(B.size == x0_truth.size):
            us = us.reshape((num_steps,1))
            B = B.reshape((x0_truth.size, 1))
        else:
            assert(us.shape[1] == B.shape[1])
    else:
        assert(us is None)
        B = np.zeros((x0_truth.size,1))
        us = np.zeros((num_steps,1))
    assert(Gamma is not None)
    assert(beta is not None)
    Gamma = Gamma.reshape((x0_truth.size, beta.size))

    xs = [x0_truth]
    zs = [] 
    ws = [] 
    vs = []

    xk = x0_truth.copy()
    if(with_zeroth_step_msmt):
        v0 = np.array([np.random.standard_cauchy() * gam for gam in gamma])
        zs.append( H @ xk + v0 )
        vs.append( v0 )
    
    for i in range(num_steps):
        if dynamics_update_callback is not None:
            dynamics_update_callback(Phi, B, Gamma, H, beta, gamma, i, other_params)
        uk = us[i]
        wk = np.array([np.random.standard_cauchy() * bet for bet in beta])
        xk = Phi @ xk + B @ uk + Gamma @ wk 
        xs.append(xk)
        ws.append(wk)
        vk = np.array([np.random.standard_cauchy() * gam for gam in gamma])
        zs.append( H @ xk + vk )
        vs.append( vk )
    return ( np.array(xs), np.array(zs), np.array(ws), np.array(vs) )

def simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt = True, dynamics_update_callback = None, other_params = None):
    if(B is not None):
        assert(us is not None)
        assert(us.shape[0] == num_steps)
        if(B.size == x0_truth.size):
            us = us.reshape((num_steps,1))
        else:
            assert(us.shape[1] == B.shape[1])
    else:
        assert(us is None)
        B = np.zeros((x0_truth.size,1))
        us = np.zeros((num_steps,1))
    if(W.size == 1):
        W = W.reshape((1,1))
    else:
        assert(W.shape[0] == W.shape[1])
    if(V.size == 1):
        V = V.reshape((1,1))
    else:
        assert(V.shape[0] == V.shape[1])
    assert(Gamma is not None)
    assert(W is not None)
    Gamma = Gamma.reshape((x0_truth.size, W.shape[0]))
    v_zero_vec = np.zeros(V.shape[0])
    w_zero_vec = np.zeros(W.shape[0])
    xs = [x0_truth]
    zs = [] 
    ws = [] 
    vs = []

    xk = x0_truth.copy()
    if(with_zeroth_step_msmt):
        v0 = np.random.multivariate_normal(v_zero_vec, V)
        zs.append( H @ xk + v0 )
        vs.append( v0 )
    
    for i in range(num_steps):
        if dynamics_update_callback is not None:
            dynamics_update_callback(Phi, B, Gamma, H, W, V, i, other_params)
        uk = us[i]
        wk = np.random.multivariate_normal(w_zero_vec, W)
        xk = Phi @ xk + B @ uk + Gamma @ wk 
        xs.append(xk)
        ws.append(wk)
        vk = np.random.multivariate_normal(v_zero_vec, V)
        zs.append( H @ xk + vk )
        vs.append( vk )
    return ( np.array(xs), np.array(zs), np.array(ws), np.array(vs) )

def plot_simulation_history(cauchy_moment_info, simulation_history, kf_history, with_partial_plot=False, with_cauchy_delay=False, scale=1):
    
    with_sim = simulation_history is not None
    with_kf = kf_history is not None
    with_ce = cauchy_moment_info is not None

    if with_sim:
        n = simulation_history[0].shape[1]
        T = np.arange(0, simulation_history[1].shape[0])
    elif with_kf:
        n = kf_history[0].shape[1]
        T = np.arange(0, kf_history[0].shape[0])
    elif with_ce:
        n = cauchy_moment_info["x"][0].size
        T = len(cauchy_moment_info["x"]) + with_cauchy_delay
    else:
        print("Must provide simulation data, kalman filter data or cauchy estimator data!\nExiting function with no plotting (Nothing Given!)")
        return 

    # Simulation history
    if with_sim:
        true_states = simulation_history[0]
        msmts = simulation_history[1]
        proc_noises = simulation_history[2]
        msmt_noises = simulation_history[3]

    # Cauchy Estimator
    if with_ce:
        means = np.array(cauchy_moment_info["x"])
        covars = np.array(cauchy_moment_info["P"])
        cerr_norm_factors = np.array(cauchy_moment_info["cerr_fz"])
        cerr_means = np.array(cauchy_moment_info["cerr_x"])
        cerr_covars = np.array(cauchy_moment_info["cerr_P"])
        n = means.shape[1]
    
    # Kalman filter 
    with_kf = kf_history is not None
    if with_kf:
        kf_cond_means = kf_history[0]
        kf_cond_covars = kf_history[1] 
        


    # Check array lengths, cauchy_delay, partial plot parameters
    if with_ce:
        cd = with_cauchy_delay 
        #plot_len variable has been introduced so that runs which fail can still be partially plotted
        if(not with_partial_plot and with_cauchy_delay):
            plot_len = covars.shape[0] + cd
            if(plot_len != T.size):
                print("[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths")
                print("Cauchy Covars size: ", covars.shape)
                print("with_cauchy_delay: ", with_cauchy_delay)
                print("T size: ", T.size)
                print("Please fix appropriately!")
                assert(False)
        elif(with_partial_plot or with_cauchy_delay):
            plot_len = covars.shape[0] + cd
            if(plot_len > T.size):
                print("[ERROR]: covars.shape[0] + with_cauchy_delay > T.size. You have mismatch in array lengths")
                print("Cauchy Covars size: ", covars.shape)
                print("with_cauchy_delay: ", with_cauchy_delay)
                print("T size: ", T.size)
                print("Please fix appropriately!")
                assert(False)
        else:
            if(covars.shape[0] + cd != T.size):
                print("[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths")
                print("Cauchy Covars size: ", covars.shape)
                print("with_cauchy_delay: ", with_cauchy_delay)
                print("T size: ", T.size)
                print("Please toggle on 'p' option for partial plotting or set 'd' to lag cauchy estimator appropriately")
                assert(False)
            plot_len = T.size
    else:
        plot_len = T.size

    # 1.) Plot the true state history vs the conditional mean estimate  
    # 2.) Plot the state error and one-sigma bound of the covariance 
    # 3.) Plot the msmts, and the msmt and process noise 
    # 4.) Plot the max complex error in the mean/covar and norm factor 
    fig = plt.figure()
    #if with_kf:
    fig.suptitle("True States (r) vs Cauchy (b) vs Kalman (g--)")
    #else:
    #   fig.suptitle("True States (r) vs Cauchy Estimates (b)")
    for i in range(n):
        plt.subplot(int(str(n) + "1" + str(i+1)))
        if with_sim:
            plt.plot(T[:plot_len], true_states[:plot_len,i], 'r')
        if with_ce:
            plt.plot(T[cd:plot_len], means[:,i], 'b')
        if with_kf:
            plt.plot(T[:plot_len], kf_cond_means[:plot_len,i], 'g--')

    if with_kf or with_ce:
        fig = plt.figure()
        #if with_kf:
        fig.suptitle("Cauchy 1-Sig (b/r) vs Kalman 1-Sig (g-/m-)")
        #else:
        #    fig.suptitle("State Error (b) vs One Sigma Bound (r)")
        for i in range(n):
            plt.subplot(int(str(n) + "1" + str(i+1)))
            if with_ce:
                plt.plot(T[cd:plot_len], true_states[cd:plot_len,i] - means[:,i], 'b')
                plt.plot(T[cd:plot_len], scale*np.sqrt(covars[:,i,i]), 'r')
                plt.plot(T[cd:plot_len], -scale*np.sqrt(covars[:,i,i]), 'r')
            if with_kf:
                plt.plot(T[:plot_len], true_states[:plot_len,i] - kf_cond_means[:plot_len,i], 'g--')
                plt.plot(T[:plot_len], scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'm--')
                plt.plot(T[:plot_len], -scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'm--')

    if with_sim:
        line_types = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        fig = plt.figure()
        fig.suptitle("Msmts (m), Msmt Noise (g), Proc Noise (b)")
        m = 3 #proc_noises.shape[1] + msmt_noises.shape[1] + msmts.shape[1]
        count = 1
        plt.subplot(int(str(m) + "1" + str(count)))
        for i in range(msmts.shape[1]):
            plt.plot(T[:plot_len], msmts[:plot_len,i], "m" + line_types[i])
        count += 1
        plt.subplot(int(str(m) + "1" + str(count)))
        for i in range(msmt_noises.shape[1]):
            plt.plot(T[:plot_len], msmt_noises[:plot_len,i], "m" + line_types[i])
        count += 1
        plt.subplot(int(str(m) + "1" + str(count)))
        for i in range(proc_noises.shape[1]):
            plt.plot(T[1:plot_len], proc_noises[:plot_len-1,i], "k" + line_types[i])
    if with_ce:
        fig = plt.figure()
        fig.suptitle("Complex Errors (mean,covar,norm factor) in Semi-Log")
        plt.subplot(311)
        plt.semilogy(T[cd:plot_len], cerr_means, 'g')
        plt.subplot(312)
        plt.semilogy(T[cd:plot_len], cerr_covars, 'g')
        plt.subplot(313)
        plt.semilogy(T[cd:plot_len], cerr_norm_factors, 'g')
    plt.show()

def plot_2D_pointwise_cpdfs(XYZ_list, cond_means_list, colors):
        GRID_HEIGHT = 8
        GRID_WIDTH = 2
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')
        #plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath},{amssymb}}'

        fig = plt.figure(figsize=(15,11))
        gs = fig.add_gridspec(GRID_HEIGHT,GRID_WIDTH)
        ax = fig.add_subplot(gs[0:GRID_HEIGHT-1,:],projection='3d')
        plt.tight_layout()
        
        for i in range(len(XYZ_list)):
            X,Y,Z = XYZ_list[i]
            cmean = cond_means_list[i]
            color = colors[i]
            #ax.plot_wireframe(X, Y, Z, color=color, label=r"Cauchy Estimator's CPDF: \text{  }$f_{X_k|Y_k}(x_k|y_k)$")
            ax.plot_surface(X,Y,Z, color=color,linewidth=0)

            ax.scatter(cmean[0], cmean[1], 0, label=r"Cauchy Estimator's State Hypothesis: $\hat{x}_k$", color=color, marker='^', s=100)

        ax.set_xlabel("x-axis (State-1)", fontsize=14)
        ax.set_ylabel("y-axis (State-2)", fontsize=14)
        ax.set_zlabel("z-axis (CPDF Probability)", rotation=180, fontsize=14)
        #ax.legend(loc=2, bbox_to_anchor=(-.52, 1), fontsize=14)
        plt.show()

# runge kutta integrator
def runge_kutta4(f, x, dt):
    k1 = f(x)
    k2 = f(x + dt*k1/2.0)
    k3 = f(x + dt*k2/2.0)
    k4 = f(x + dt*k3)
    x_new = x + 1.0 / 6.0 * (k1 + 2*k2 + 2*k3 + k4) * dt 
    return x_new

# returns Central Difference Gradient of vector f, the matrix Jacobian, 4th Order expansion
def cd4_gvf(x, f, other_params=None):
    # numerical gradient 
    n = x.size
    if other_params is None:
        m = f(x).size
    else:
        m = f(x, *other_params).size
    ep = 1e-5
    G = np.zeros((m,n))
    zr = np.zeros(n)
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        if other_params is None:
            G[:,i] = (-1.0 * f(x + 2.0*ep*ei) + 8.0*f(x + ep*ei) - 8.0 * f(x - ep*ei) + f(x - 2.0*ep*ei) ) / (12.0*ep) 
        else:
            G[:,i] = (-1.0 * f(x + 2.0*ep*ei, *other_params) + 8.0*f(x + ep*ei, *other_params) - 8.0 * f(x - ep*ei, *other_params) + f(x - 2.0*ep*ei, *other_params) ) / (12.0*ep) 
    return G

# input: jacobian of nonliinear dynamics matrix f(x,u) w.r.t x, continous time control matrix G, power spectral density Q of ctime process, change in time dt (time of step k to k+1), order of taylor approximation
# returns: discrete state transition matrix, discrete control matrix, and discrete process noise matrix, given the gradient of f(x,u) w.r.t x
# This function essentially gives you the process model parameter matrices required for the EKF
def discretize_nl_sys(JacA, G, Q, dt, order, with_Gamk = True, with_Wk = True):
    assert(JacA.ndim == 2)
    assert(G.ndim == 2)
    if with_Wk: 
        assert(Q.ndim == 2)
    assert(dt > 0)
    assert(order > 0)

    n = JacA.shape[0]
    Phi_k = np.zeros((n,n))
    Gam_k = np.zeros_like(G)
    W_k = np.zeros((n,n))

    # Form Discrete Time State Transition Matrices Phi_k and Control Gain Matrix Gam_k
    for i in range(order+1):
        Phi_k += np.linalg.matrix_power(JacA, i) * dt**i / math.factorial(i)
        if with_Gamk:
            Gam_k += np.linalg.matrix_power(JacA, i) @ G * dt**(i+1) / math.factorial(i+1)

    if with_Wk:
        # Form Discrete Time Noise Matrix Qk
        for i in range(order+1):
            for j in range(order+1):
                tmp_i = np.linalg.matrix_power(JacA, i) / math.factorial(i) @ G
                tmp_j = np.linalg.matrix_power(JacA, j) / math.factorial(j) @ G
                Tk_coef = dt**(i+j+1) / (i+j+1)
                W_k += tmp_i @ Q @ tmp_j.T * Tk_coef
    if not with_Gamk and not with_Wk:
        return Phi_k
    elif not with_Gamk and with_Wk:
        return Phi_k, W_k
    elif with_Gamk and not with_Wk:
        return Phi_k, Gam_k
    else:
        return Phi_k, Gam_k, W_k
    
    

def discretize_nl_sys_proccess_noise(JacA, G, Q, dt, order):
    assert(JacA.ndim == 2)
    assert(G.ndim == 2)
    assert(Q.ndim == 2)
    assert(dt > 0)
    assert(order > 0)

    n = JacA.shape[0]
    Q_k = np.zeros((n,n))
    # Form Discrete Time Noise Matrix Qk
    for i in range(order+1):
        for j in range(order+1):
            tmp_i = np.linalg.matrix_power(JacA, i) / math.factorial(i) @ G
            tmp_j = np.linalg.matrix_power(JacA, j) / math.factorial(j) @ G
            Tk_coef = dt**(i+j+1) / (i+j+1)
            Q_k += tmp_i @ Q @ tmp_j.T * Tk_coef
    return Q_k


# input: Continous time dynamics A, continous time PSD of full system state, dt 
# Q should be positive semidefinite
def discretize_ctime_process_noise_lyap(A, Q, dt, order):
    assert(A.shape == Q.shape)
    import scipy.linalg as la 
    # P is the solution to the continous time lyapunov equation 
    P = la.solve_continuous_lyapunov(A, -Q)
    # Can find discrete time process noise by manipulating steady state discrete time expression:
    # P = Phi @ P @ Phi^T + Qk
    n = A.shape[0]
    Phi = np.zeros((n,n))
    # Form State Transition Matrices
    for i in range(order+1):
        Phi += np.linalg.matrix_power(A, i) * dt**i / math.factorial(i)
    Qk = P - Phi @ P @ Phi.T
    return Qk

# Draw a random exponential variable of the pdf \lambda * exp(-\lambda * x)
#@nb.njit(cache=True)
def random_exponential(lam):
    EPS = 1e-16
    ALMOST_ONE = 1.0 - EPS
    # Draw a random uniform variable on the open interval (0,1)
    U = np.random.uniform(EPS, ALMOST_ONE)
    return  -np.log( U ) / lam

# Draw a random alpha stable variable 
# the parameters are: 
# 1.) alpha \in (0,2] -- this is the stability param (2=Gaussian, 1 = Cauchy, 0.5 = Levy)
# 2.) beta \in [-1,1] -- this is the skewness param 
# 3.) c \in (0, inf] -- this is the scale param (standard deviation for Gaussian)
# 4.) mu \in [-inf,inf] -- this is the location parameter
# Note: For any value of alpha less than or equal to 2, the variance is undefined 
# This implements the Chambers, Mallows, and Stuck (CMS) method from their seminal paper in 1976
#@nb.njit(cache=True)
def random_alpha_stable(alpha, beta, c, mu):
    EPS = 1e-16
    ALMOST_ONE = 1.0 - EPS
    #Generate a random variable on the open interval (-pi/2, pi/2)
    U = np.random.uniform(-np.pi/2.0, np.pi/2.0) * ALMOST_ONE
    #Generate a random exponential variable with mean of 1.0
    W = random_exponential(1.0)
    zeta = -beta * np.tan(np.pi * alpha / 2.0)
    xi = np.pi / 2.0 if alpha == 1.0 else 1.0 / alpha * np.arctan(-zeta)
    X = 0.0 # ~ S_\alpha(\beta,1,0)
    if(alpha == 1.0):
        X =  1.0 / xi * ( (np.pi / 2.0 + beta * U) * np.tan(U) - beta * np.log((np.pi/2.0 * W * np.cos(U)) / (np.pi/2.0 + beta * U)) )
    else:
        X = (1.0 + zeta**2)**(1.0/(2.0*alpha)) * np.sin(alpha*(U+xi)) / (np.cos(U)**(1.0/alpha)) * ((np.cos(U - alpha*(U + xi))) / W)**( (1.0 - alpha) / alpha )
    # Now scale and locate the random variable depending on alpha == 1.0 or not 
    Y = 0.0
    if(alpha == 1.0):
        Y = c*X + (2.0/np.pi)*beta*c*np.log(c) + mu
    else:
        Y = c*X + mu 
    return Y

# Draw a random alpha stable variable 
# This function assumes that the beta (skewness parameter) for the random alpha stable method is zero 
# the parameters are: 
# 1.) alpha \in (0,2] -- this is the stability param (2=Gaussian, 1 = Cauchy, 0.5 = Levy)
# 2.) c \in (0, inf] -- this is the scale param (standard deviation for Gaussian)
# 3.) mu \in [-inf,inf] -- this is the location parameter
# Note: For any value of alpha less than or equal to 2, the variance is undefined 
# This implements the Chambers, Mallows, and Stuck (CMS) method from their seminal paper in 1976
#@nb.njit(cache=True)
def random_symmetric_alpha_stable(alpha, c, mu):
    EPS = 1e-16
    ALMOST_ONE = 1.0 - EPS
    #Generate a random variable on interval (-pi/2, pi/2)
    U = np.random.uniform(-np.pi/2.0, np.pi/2.0) * ALMOST_ONE
    #Generate a random exponential variable with mean of 1.0
    W = random_exponential(1.0)
    xi = np.pi / 2.0 if alpha == 1.0 else 0.0
    X = 0.0 # ~ S_\alpha(\beta,1,0)
    if(alpha == 1.0):
        X = np.tan(U)
    else:
        X = np.sin(alpha*(U+xi)) / (np.cos(U)**(1.0/alpha)) * ((np.cos(U - alpha*(U + xi))) / W)**( (1.0 - alpha) / alpha )
    # Now scale and locate the random variable
    Y = c*X + mu
    return Y

