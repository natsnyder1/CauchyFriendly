import numpy as np 
import ctypes as ct
import pycauchy
import matplotlib.pyplot as plt 
import math

CAUCHY_TO_GAUSSIAN_NOISE = 1.3898
GAUSSIAN_TO_CAUCHY_NOISE = 1.0 / CAUCHY_TO_GAUSSIAN_NOISE

# Used to initialize communication with the underlying shared library
# Can run in 3 modes
# lti -- Dynamics matrices constant
# ltv -- Dynamics matrices non-constant
# nonlin -- Nonlinear system, uses extended cauchy

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
    # The inputs to these functions are numpy vectors / matrices, which then sets the raw C-Pointers
    def cget_x(self):
        x = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            x[i] = self.cduc.contents.x[i]
        return x
    
    def cget_step(self):
        return self.cduc.contents.step

    def cset_x(self, x):
        assert(x.ndim == 1)
        assert(x.size == self.n)
        for i in range(self.n):
            self.cduc.contents.x[i] = x[i]

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

    def cset_Gamma(self, Gamma):
        size_Gamma = self.n*self.pncc
        assert(Gamma.size == size_Gamma)
        _Gamma = Gamma.reshape(-1)
        for i in range(size_Gamma):
            self.cduc.contents.Gamma[i] = _Gamma[i]

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

    def cset_gamma(self, gamma):
        size_gamma = self.cmcc
        assert(gamma.size == size_gamma)
        _gamma = gamma.reshape(-1)
        for i in range(size_gamma):
            self.cduc.contents.gamma[i] = _gamma[i]

    def cset_is_xbar_set_for_ece(self):
        self.cduc.contents.is_xbar_set_for_ece = True

    def cset_zbar(self, c_zbar, zbar):
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
    #   H <- compute using jacobian_h(x_k+1)
    #   gamma <- possibly set if time varying
    #   (call) pyduc.set_is_xbar_set_for_ece() (this tells the C library you are absolutely sure that you have updated the system state)
    # ...
    # end doing stuff

    # return stuff to C
    # call pyduc.cset_"x/Phi/Gamma/B/H/beta/gamma" depending on what you have updated
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


class PySlidingWindowManager():

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
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _Phi = Phi.reshape(-1).astype(np.float64)
        _Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        _H = H.reshape(-1).astype(np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        _win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
        
        pycauchy.pycauchy_initialize_lti_window_manager(self.num_windows, self.num_sim_steps, _A0, _p0, _b0, _Phi, _Gamma, _B, _beta, _H, _gamma, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, _win_var_boost)
        self.is_initialized = True
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: Conditional Mean/Variance will be a function of the last {} time-steps, {} measurements per step == {} total!".format(self.num_windows, self.p, self.p * self.num_windows) )

    def initialize_ltv(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, dt=0, step=0, win_var_boost = None):
        if(self.mode != "ltv"):
            print("Attempting to call initialize_ltv method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTV initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _Phi = Phi.reshape(-1).astype(np.float64)
        _Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        _H = H.reshape(-1).astype(np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        _win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
                # create the dynamics_update_callback ctypes callback function
        py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        f_duc = py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(f_duc, ct.c_void_p).value

        pycauchy.pycauchy_initialize_ltv_window_manager(self.num_windows, self.num_sim_steps, _A0, _p0, _b0, _Phi, _Gamma, _B, _beta, _H, _gamma, self.f_duc_ptr1, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, _win_var_boost)
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

        _x0 = x0.reshape(-1).astype(np.float64)
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        _cmcc = self.cmcc
        if(win_var_boost is not None):
            assert(type(win_var_boost) == np.ndarray)
            assert(win_var_boost.size == self.n)
        _win_var_boost = win_var_boost.reshape(-1).astype(np.float64) if win_var_boost is not None else np.array([], dtype = np.float64)
        
        pycauchy.pycauchy_initialize_nonlin_window_manager(self.num_windows, self.num_sim_steps, _x0, _A0, _p0, _b0, _beta, _gamma, self.f_duc_ptr1, self.f_duc_ptr2, self.f_duc_ptr3, _cmcc, self.debug_print, self.log_seq, self.log_full, self.log_dir, _dt, _step, _win_var_boost)
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
        self.moment_info = {"x" : [], "P" :[], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "err_code" : []} # Full (or best) window means
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
    
    def initialize_lti(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, init_step=0, dt=0):
        if self.mode != "lti":
            print("Attempting to call initialize_lti method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTI initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _Phi = Phi.reshape(-1).astype(np.float64)
        _Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        _H = H.reshape(-1).astype(np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)
        _init_step = int(init_step)
        _dt = float(dt)
        
        self.py_handle = pycauchy.pycauchy_initialize_lti(self.num_steps, _A0, _p0, _b0, _Phi, _Gamma, _B, _beta, _H, _gamma, _dt, _init_step, self.debug_print)
        self.is_initialized = True
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")
        print("Note: You can call the step function {} time-steps, {} measurements per step == {} total times!".format(self.num_steps, self.p, self.num_steps * self.p) )

    def initialize_ltv(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, init_step=0, dt=0):
        if(self.mode != "ltv"):
            print("Attempting to call initialize_ltv method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTV initialization not successful!")
            return
        self._ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _Phi = Phi.reshape(-1).astype(np.float64)
        _Gamma = Gamma.reshape(-1).astype(np.float64) if Gamma is not None else np.array([], dtype = np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _B = B.reshape(-1).astype(np.float64) if B is not None else np.array([], dtype = np.float64)
        _H = H.reshape(-1).astype(np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)

        # create the dynamics_update_callback ctypes callback function
        py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(C_CauchyDynamicsUpdateContainer))
        f_duc = py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(f_duc, ct.c_void_p).value
        _init_step = int(init_step)
        _dt = float(dt)

        self.py_handle = pycauchy.pycauchy_initialize_ltv(self.num_steps, _A0, _p0, _b0, _Phi, _Gamma, _B, _beta, _H, _gamma, self.f_duc_ptr1, _dt, _init_step, self.debug_print)
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

        _x0 = x0.reshape(-1).astype(np.float64)
        _A0 = A0.reshape(-1).astype(np.float64)
        _p0 = p0.reshape(-1).astype(np.float64)
        _b0 = b0.reshape(-1).astype(np.float64)
        _beta = beta.reshape(-1).astype(np.float64) if beta is not None else np.array([], dtype = np.float64)
        _gamma = gamma.reshape(-1).astype(np.float64)
        _dt = float(dt)
        _step = int(step)
        _cmcc = self.cmcc
        
        self.py_handle = pycauchy.pycauchy_initialize_nonlin(self.num_steps, _x0, _A0, _p0, _b0, _beta, _gamma, self.f_duc_ptr1, self.f_duc_ptr2, self.f_duc_ptr3, _cmcc, _dt, _step, self.debug_print)
        self.is_initialized = True
        print("Nonlin initialization successful! You can use the step(msmts, controls) method now to run the Cauchy Estimtor!")
        print("Note: You can call the step function {} time-steps, {} measurements per step == {} total times!".format(self.num_steps, self.p, self.num_steps * self.p) )

    def step(self, msmts, controls):
        if(self.is_initialized == False):
            print("Estimator is not initialized yet. Mode set to {}. Please call method initialize_{} before running step()!".format(self.mode, self.mode))
            print("Not stepping! Please call correct method / fix mode!")
            return
        if(self.step_count == self.num_steps):
            print("[Error:] Cannot step estimator again, you have already stepped the estimator the initialized number of steps")
            print("Not stepping! Please shut estimator down!")
            return
        _msmts, _controls = self._msmts_controls_checker(msmts, controls)
        self.last_msmt = _msmts[-1]

        if self.mode != "nonlin":
            fz, x, P, cerr_fz, cerr_x, cerr_P, err_code = pycauchy.pycauchy_single_step_ltiv(self.py_handle, _msmts, _controls)
        else:
            fz, x, P, cerr_fz, cerr_x, cerr_P, err_code = pycauchy.pycauchy_single_step_nonlin(self.py_handle, _msmts, _controls, self.step_count != 0)

        self.moment_info["fz"].append(fz)
        self.moment_info["x"].append(x)
        self.moment_info["P"].append(P.reshape((self.n, self.n)))
        self.moment_info["cerr_x"].append(cerr_x)
        self.moment_info["cerr_P"].append(cerr_P)
        self.moment_info["cerr_fz"].append(cerr_fz)
        self.moment_info["err_code"].append(err_code)
        self.step_count += 1

        # Shuts down sliding window manager
    def shutdown(self):
        if(self.is_initialized == False):
            print("Cannot shutdown Cauchy Estimator before it has been initialized!")
            return
        pycauchy.pycauchy_single_step_shutdown(self.py_handle)
        self.py_handle = None
        print("Cauchy estimator backend C data structure has been shutdown!")
        self.is_initialized = False
    
    def get_2D_pointwise_cpdf(self, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir = None):
        if(self.is_initialized == False):
            print("Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!")
            return
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
        if log_dir is None:
            _log_dir = None
        else:
            _log_dir = str(log_dir)
            if _log_dir == "":
                _log_dir = None
            elif _log_dir[-1] == "/":
                _log_dir = log_dir[:-1]

        cpdf_points, num_gridx, num_gridy = pycauchy.pycauchy_get_2D_pointwise_cpdf(self.py_handle, _gridx_low, _gridx_high, _gridx_resolution, _gridy_low, _gridy_high, _gridy_resolution, _log_dir)
        cpdf_points = cpdf_points.reshape(num_gridx*num_gridy, 3)
        X = cpdf_points[:,0].reshape( (num_gridy, num_gridx) )
        Y = cpdf_points[:,1].reshape( (num_gridy, num_gridx) )
        Z = cpdf_points[:,2].reshape( (num_gridy, num_gridx) )
        return X, Y, Z
    
    def get_reinitialization_statistics(self):
        if( (self.step_count == 0) or (self.is_initialized == False) ):
            print("[Error get_reinitialization_statistics]: Cannot find reinitialization stats of an estimator not initialized, or that has not processed at least one measurement! Please correct!")
            return None, None, None
        if self.mode != "nonlin":
            A0, p0, b0 = pycauchy.pycauchy_get_reinitialization_statistics(self.py_handle, self.last_msmt)
            A0 = A0.reshape( (self.n, self.n) )
            return A0, p0, b0
        else:
            print("Mode Nonlinear Not implemented yet!")
            return None, None, None
    def get_last_mean_cov(self):
        return self.moment_info["x"][-1], self.moment_info["P"][-1]
        
    def plot_2D_pointwise_cpdf(self, X,Y,Z):
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
        
        ax.set_xlabel("x-axis (State-1)", fontsize=14)
        ax.set_ylabel("y-axis (State-2)", fontsize=14)
        ax.set_zlabel("z-axis (CPDF Probability)", rotation=180, fontsize=14)
        #ax.legend(loc=2, bbox_to_anchor=(-.52, 1), fontsize=14)
        plt.show()


    def __del__(self):
        if self.is_initialized:
            self.shutdown()
            self.is_initialized = False


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
    
    # Sim
    true_states = simulation_history[0]
    msmts = simulation_history[1]
    proc_noises = simulation_history[2]
    msmt_noises = simulation_history[3]

    # Cauchy 
    means = np.array(cauchy_moment_info["x"])
    covars = np.array(cauchy_moment_info["P"])
    cerr_norm_factors = np.array(cauchy_moment_info["cerr_fz"])
    cerr_means = np.array(cauchy_moment_info["cerr_x"])
    cerr_covars = np.array(cauchy_moment_info["cerr_P"])
    
    # Kalman filter 
    with_kf = kf_history is not None
    if with_kf:
        kf_cond_means = kf_history[0]
        kf_cond_covars = kf_history[1] 

    # Check array lengths, cauchy_delay, partial plot parameters
    n = means.shape[1]
    T = np.arange(0, msmts.shape[0])
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
    
    # 1.) Plot the true state history vs the conditional mean estimate  
    # 2.) Plot the state error and one-sigma bound of the covariance 
    # 3.) Plot the msmts, and the msmt and process noise 
    # 4.) Plot the max complex error in the mean/covar and norm factor 
    fig = plt.figure(1)
    if with_kf:
        fig.suptitle("True States (r) vs Cauchy (b) vs Kalman (g--)")
    else:
        fig.suptitle("True States (r) vs Cauchy Estimates (b)")
    for i in range(covars.shape[1]):
        plt.subplot(int(str(n) + "1" + str(i+1)))
        plt.plot(T[:plot_len], true_states[:plot_len,i], 'r')
        plt.plot(T[cd:plot_len], means[:,i], 'b')
        if with_kf:
            plt.plot(T[:plot_len], kf_cond_means[:plot_len,i], 'g--')

    fig = plt.figure(2)
    if with_kf:
        fig.suptitle("Cauchy 1-Sig (b/r) vs Kalman 1-Sig (g-/m-)")
    else:
        fig.suptitle("State Error (b) vs One Sigma Bound (r)")
    for i in range(covars.shape[1]):
        plt.subplot(int(str(n) + "1" + str(i+1)))
        plt.plot(T[cd:plot_len], true_states[cd:plot_len,i] - means[:,i], 'b')
        plt.plot(T[cd:plot_len], scale*np.sqrt(covars[:,i,i]), 'b--')
        plt.plot(T[cd:plot_len], -scale*np.sqrt(covars[:,i,i]), 'b--')
        if with_kf:
            plt.plot(T[:plot_len], true_states[:plot_len,i] - kf_cond_means[:plot_len,i], 'g--')
            plt.plot(T[:plot_len], scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'g--')
            plt.plot(T[:plot_len], -scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'g--')

    line_types = ['-', '--', '-.', ':', '-']
    fig = plt.figure(3)
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

    fig = plt.figure(4)
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
def cd4_gvf(x, f):
    # numerical gradient 
    n = x.size
    m = f(x).size
    ep = 1e-5
    G = np.zeros((m,n))
    zr = np.zeros(n)
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        G[:,i] = (-1.0 * f(x + 2.0*ep*ei) + 8.0*f(x + ep*ei) - 8.0 * f(x - ep*ei) + f(x - 2.0*ep*ei) ) / (12.0*ep) 
    return G

# input: jacobian of nonliinear dynamics matrix f(x,u) w.r.t x, continous time control matrix G, power spectral density Q of ctime process, change in time dt (time of step k to k+1), order of taylor approximation
# returns: discrete state transition matrix, discrete control matrix, and discrete process noise matrix, given the gradient of f(x,u) w.r.t x
# This function essentially gives you the process model parameter matrices required for the EKF
def discretize_nl_sys(JacA, G, Q, dt, order):
    assert(JacA.ndim == 2)
    assert(G.ndim == 2)
    assert(Q.ndim == 2)
    assert(dt > 0)
    assert(order > 0)

    n = JacA.shape[0]
    Phi_k = np.zeros((n,n))
    Gam_k = np.zeros_like(G)
    Q_k = np.zeros((n,n))

    # Form Discrete Time State Transition Matrices Phi_k and Control Gain Matrix Gam_k
    for i in range(order+1):
        Phi_k += np.linalg.matrix_power(JacA, i) * dt**i / math.factorial(i)
        Gam_k += np.linalg.matrix_power(JacA, i) @ G * dt**(i+1) / math.factorial(i+1)

    # Form Discrete Time Noise Matrix Qk
    for i in range(order+1):
        for j in range(order+1):
            tmp_i = np.linalg.matrix_power(JacA, i) / math.factorial(i) @ G
            tmp_j = np.linalg.matrix_power(JacA, j) / math.factorial(j) @ G
            Tk_coef = dt**(i+j+1) / (i+j+1)
            Q_k += tmp_i @ Q @ tmp_j.T * Tk_coef
    return Phi_k, Gam_k, Q_k

# input: Continous time dynamics A, continous time PSD of full system state, dt 
# Q should be positive semidefinite
def discretize_ctime_process_noise(A, Q, dt, order):
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