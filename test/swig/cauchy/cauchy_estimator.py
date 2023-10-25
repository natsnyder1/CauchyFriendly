import numpy as np 
import ctypes as ct
#import pycauchy as cauchy


# Need a user interface for the user


# Used to initialize communication with the underlying shared library
# Can run in 3 modes
# lti -- Dynamics matrices constant
# ltv -- Dynamics matrices non-constant
# nonlin -- Nonlinear system, uses extended cauchy

class CDynamicsUpdateContainer(ct.Structure):
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

class PyDynamicsUpdateContainer():
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
            x[i] = self.cduc.contains.x[i]
        return x
    
    def cget_step(self):
        return self.cduc.contains.step

    def cset_x(self, x):
        assert(x.ndim == 1)
        assert(x.size == self.n)
        for i in range(self.n):
            self.cduc.contains.x[i] = x[i]

    def cset_Phi(self, Phi):
        size_Phi = self.n * self.n
        assert(Phi.size == size_Phi)
        _Phi = Phi.reshape(-1)
        for i in range(size_Phi):
            self.cduc.contains.Phi[i] = _Phi[i]
    def cset_Gamma(self, Gamma):
        size_Gamma = self.n*self.pncc
        assert(Gamma.size == size_Gamma)
        _Gamma = Gamma.reshape(-1)
        for i in range(size_Gamma):
            self.cduc.contains.Gamma[i] = _Gamma[i]
    def cset_B(self, B):
        size_B = self.n*self.cmcc
        assert(B.size == size_B)
        _B = B.reshape(-1)
        for i in range(size_B):
            self.cduc.contains.B[i] = _B[i]

    def cset_beta(self, beta):
        size_beta = self.pncc
        assert(beta.size == size_beta)
        _beta = beta.reshape(-1)
        for i in range(size_beta):
            self.cduc.contains.beta[i] = _beta[i]

    def cset_H(self, H):
        size_H = self.p*self.n
        assert(H.size == size_H)
        _H = H.reshape(-1)
        for i in range(size_H):
                self.cduc.contains.H[i] = _H[i]

    def cset_gamma(self, gamma):
        size_gamma = self.cmcc
        assert(gamma.size == size_gamma)
        _gamma = gamma.reshape(-1)
        for i in range(size_gamma):
            self.cduc.contains.gamma[i] = _gamma[i]

    def set_is_xbar_set_for_ece(self):
        self.cduc.contains.is_xbar_set_for_ece = True

def template_dynamics_update_callback(cduc):
    # In all callbacks, unless you really know what you are doing, call this first
    # This creates an object which returns the C data arrays thorught get/set methods as nice numpy matrices / vectors
    pyduc = PyDynamicsUpdateContainer(cduc)
    
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

def template_extended_msmt_update_callback(cduc):
    # In all callbacks, unless you really know what you are doing, call this first
    # This creates an object which returns the C data arrays thorught get/set methods as nice numpy matrices / vectors
    pyduc = PyDynamicsUpdateContainer(cduc)
    
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


class PyCauchyEstimator():

    def __init__(self, mode, num_windows, num_sim_steps, log_dir = None, debug_print = True, log_seq = False, log_all = False):
        self.num_windows = num_windows
        self.num_sim_steps = num_sim_steps
        self.modes = ["lti", "ltv", "nonlin"]
        if(mode.lower() not in self.modes):
            print("[Error PyCauchyEstimator:] chosen mode {} invalid. Please choose one of the following: {}".format(mode, self.modes))
        else:
            self.mode = mode.lower()
            print("Set Cauchy Estimator Mode to:", self.mode)
        self.is_initialized = False
        self.moment_info = {"x" : [], "P" :[], "cerr_x" : [], "cerr_P" : [], "fz" :[], "cerr_fz" : [], "err_code" : []} # Full window means
        self.debug_print = debug_print
        self.log_seq = log_seq
        self.log_all = log_all 
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
                cmcc == B.shape[1]
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
            if(type(controls) != np.ndarray):
                _controls = np.array([controls]).reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
            else:
                _controls = controls.copy().reshape(-1).astype(np.float64)
                assert(_controls.size == self.cmcc)
        return _msmts, _controls
    
    def initialize_lti(self, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dt=0, step=0):
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
        cauchy.pycauchy_initialize_lti_window_manager(self.num_windows, self.num_sim_steps, _A0, _p0, _b0, _Phi, _Gamma, _B, _beta, _H, _gamma, self.debug_print, self.log_seq, self.log_all, self.log_dir, _dt, _step)
        self.is_initialized = True
        print("LTI initialization successful! You can use the step(msmts, controls) method to run the estimtor now!")

    def initialize_ltv(self, A0, p0, b0, Phi, Gamma, B, beta, H, gamma, dyn_update_callback):
        if(self.mode != "ltv"):
            print("Attempting to call initialize_ltv method when mode was set to {} is not allowed! You must call initialize_{} ... or reset the mode altogether!".format(self.mode, self.mode))
            print("LTV initialization not successful!")
            return
        print("LTV NOT IMPLEMENTED YET")
        exit(1)
    
    def initialize_nonlin(self, x0, A0, p0, b0, beta, gamma, cmcc, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, dt=0, step=0):
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
        assert(cmcc >= 0)
        self.cmcc = int(cmcc)
        assert(x0.size == self.n)
        # create the dynamics_update_callback ctypes callback function
        py_callback_type1 = ct.CFUNCTYPE(None, ct.POINTER(CDynamicsUpdateContainer))
        f_duc = py_callback_type1(dynamics_update_callback)
        self.f_duc_ptr1 = ct.cast(f_duc, ct.c_void_p).value

        # create the nonlinear_msmt_model ctypes callback function 
        py_callback_type2 = ct.CFUNCTYPE(None, ct.POINTER(CDynamicsUpdateContainer), ct.POINTER(ct.c_double))
        f_duc = py_callback_type2(nonlinear_msmt_model)
        self.f_duc_ptr2 = ct.cast(f_duc, ct.c_void_p).value

        # create the extended_msmt_update_callback ctypes callback function 
        py_callback_type3 = ct.CFUNCTYPE(None, ct.POINTER(CDynamicsUpdateContainer))
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
        cauchy.pycauchy_initialize_nonlin_window_manager(self.num_windows, self.num_sim_steps, _x0, _A0, _p0, _b0, _beta, _gamma, _cmcc, self.f_duc_ptr1, self.f_duc_ptr2, self.f_duc_ptr3, self.debug_print, self.log_seq, self.log_all, self.log_dir, _dt, _step)
        self.is_initialized = True
        print("Nonlin initialization successful! You can use the step(msmts, controls) method now to run the cauchy estimtor!")

    def step(self, msmts, controls):
        if(self.is_initialized == False):
            print("Estimator is not initialized yet. Mode set to {}. Please call method initialize_{} before running step()!".format(self.mode, self.mode))
            print("Not stepping! Please call correct method / fix mode!")
            return
        _msmts, _controls = self._msmts_controls_checker(msmts, controls)
        fz, x, P, cerr_x, cerr_P, cerr_fz, err_code = cauchy.pycauchy_step(_msmts, _controls)
        self.moment_info["fz"].append(fz)
        self.moment_info["x"].append(x)
        self.moment_info["P"].append(P)
        self.moment_info["cerr_x"].append(cerr_x)
        self.moment_info["cerr_P"].append(cerr_P)
        self.moment_info["cerr_fz"].append(cerr_fz)
        self.moment_info["err_code"].append(err_code)

    # Shuts down sliding window manager
    def shutdown(self):
        if(self.is_initialized == False):
            print("Cannot shutdown Cauchy Estimator before it has been initialized!")
            return
        cauchy.pycauchy_shutdown()
        print("Cauchy estimator backend C data structure has been shutdown!")
        self.is_initialized = False
    
    def __del__(self):
        if self.is_initialized:
            self.shutdown()
            self.is_initialized = False