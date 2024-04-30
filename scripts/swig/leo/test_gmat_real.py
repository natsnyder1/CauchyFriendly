import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import os, sys 
import cauchy_estimator as ce 
import math 
import copy 
import time

file_dir = os.path.dirname(os.path.abspath(__file__))
gmat_root_dir = '/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/GMAT/application'
gmat_data_dir = file_dir + "/gmat_data/gps_2_11_23/"
assert( os.path.isdir(gmat_data_dir) )
gmat_bin_dir = gmat_root_dir + '/bin'
gmat_api_dir = gmat_root_dir + '/api'
gmat_py_dir = gmat_bin_dir + '/gmatpy'
gmat_startup_file = gmat_bin_dir + '/api_startup_file.txt'
sys.path.append(gmat_bin_dir)
sys.path.append(gmat_api_dir)
sys.path.append(gmat_py_dir)
if os.path.exists(gmat_startup_file):
   import gmat_py as gmat
   gmat.Setup(gmat_startup_file)
from datetime import datetime 
from datetime import timedelta 

import pickle 
import pc_prediction as pc 


MonthDic = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
MonthDic2 = {v:k for k,v in MonthDic.items()}


# Process Noise Model 
def leo6_process_noise_model(dt):
    q = 8e-15; # Process noise ncertainty in the process position and velocity
    W = np.zeros((6,6))
    W[0:3,0:3] = q * np.eye(3) * dt**3 / 3 
    W[0:3,3:6] = q * np.eye(3) * dt**2 / 2 
    W[3:6,0:3] = q * np.eye(3) * dt**2 / 2 
    W[3:6,3:6] = q * np.eye(3) * dt 
    return W

# Process Noise Model 
def leo6_process_noise_model2(dt):
    q = 8e-21; # Process noise ncertainty in the process position and velocity
    W = np.zeros((6,6))
    W[0:3,0:3] = q * np.eye(3) * dt**3 / 3 
    W[0:3,3:6] = q * np.eye(3) * dt**2 / 2 
    W[3:6,0:3] = q * np.eye(3) * dt**2 / 2 
    W[3:6,3:6] = q * np.eye(3) * dt 
    return W

# Initial state given in distance units kilometers
class FermiSatelliteModel():
    def __init__(self, t0, x0, dt):
        if type(t0) == str:
            self.t0 = t0
        elif type(t0) == datetime:
            #self.t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
            self.t0 = self.format_datetime(t0)
        else:
            print("Unrecognized format for starting time t0!")
            exit(1)
        self.x0 = x0.copy()
        self.dt = dt 
        # Solve for list 
        self.solve_for_states = [] 
        self.solve_for_taus = []
        self.solve_for_fields = [] 
        self.solve_for_dists = [] 
        self.solve_for_scales = [] 
        self.solve_for_nominals = []
        self.solve_for_alphas = []
        self.solve_for_fields_acceptable = ["Cd", "Cr"]
        self.solve_for_dists_acceptable = ["gauss", "sas"]
        self.is_model_constructed = False

    def create_model(self, with_jacchia=True, with_SRP=True):
        assert self.is_model_constructed == False
        # Solar System Properties -- Newly Added
        mod = gmat.Moderator.Instance()
        self.ss = mod.GetDefaultSolarSystem()
        self.ss.SetField("EphemerisSource", "DE421")
        self.earth = self.ss.GetBody('Earth')
        eop_file_path = gmat_data_dir + "eop_file.txt"
        self.earth.SetField('EopFileName', eop_file_path)

        # Create Fermi Model
        self.sat = gmat.Construct("Spacecraft", "Fermi")
        self.sat.SetField("DateFormat", "UTCGregorian") #EpochFormat Form Box
        self.sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
        self.sat.SetField("DisplayStateType","Cartesian") #StateType Form Box
    
        self.sat.SetField("Epoch", self.t0) # 19:31:54:000") 
        self.sat.SetField("DryMass", 3995.6)
        self.sat.SetField("Cd", 2.1)
        self.sat.SetField("Cr", 0.75)
        #self.sat.SetField("CrSigma", 0.1)
        self.sat.SetField("DragArea", 14.18)
        self.sat.SetField("SRPArea", 14.18)
        self.sat.SetField("Id", '2525')
        self.sat.SetField("X", self.x0[0])
        self.sat.SetField("Y", self.x0[1])
        self.sat.SetField("Z", self.x0[2])
        self.sat.SetField("VX", self.x0[3])
        self.sat.SetField("VY", self.x0[4])
        self.sat.SetField("VZ", self.x0[5])

        #sat.SetField("SolveFors", 'CartesianState, FogmCd, FogmAtmosDensityScaleFactor')
        self.fueltank = gmat.Construct("ChemicalTank", "FuelTank")
        self.fueltank.SetField("FuelMass", 359.9) #FuelMass = 359.9
        self.fueltank.Help()
        self.sat.SetField("Tanks", "FuelTank") # ??? does this add the fueltank to satellite?
        self.sat.Help()
        print(self.sat.GetGeneratingString(0))

        # Not sure if this is necessary
        #cordSysFermi = gmat.Construct("CoordinateSystem", "FermiVNB")
        #cordSysFermi.SetField("Origin", "Fermi")
        #cordSysFermi.SetField("Axes", "ObjectReferenced")
        #cordSysFermi.SetField("XAxis", "V") # Gives Error Message
        #cordSysFermi.SetField("YAxis", "N") # Gives Error Message
        #cordSysFermi.SetField("Primary", "Earth") # Gives Error Message
        #cordSysFermi.SetField("Secondary", "Fermi") # Gives Error Message

        # Create Force Model 
        self.fm = gmat.Construct("ForceModel", "TheForces")
        self.fm.SetField("ErrorControl", "None")
        # A 70x70 EGM96 Gravity Model
        self.earthgrav = gmat.Construct("GravityField")
        self.earthgrav.SetField("BodyName","Earth")
        self.earthgrav.SetField("Degree",70)
        self.earthgrav.SetField("Order",70)
        self.earthgrav.SetField("PotentialFile","EGM96.cof")
        self.earthgrav.SetField("TideModel", "SolidAndPole")
        # The Point Masses
        self.moongrav = gmat.Construct("PointMassForce")
        self.moongrav.SetField("BodyName","Luna")
        self.sungrav = gmat.Construct("PointMassForce")
        self.sungrav.SetField("BodyName","Sun")
        # Solar Radiation Pressure
        if with_SRP:
            self.srp = gmat.Construct("SolarRadiationPressure")
            #srp.SetField("SRPModel", "Spherical")
            self.srp.SetField("Flux", 1370.052)
        # Drag Model
        if with_jacchia:
            self.jrdrag = gmat.Construct("DragForce")
            self.jrdrag.SetField("AtmosphereModel","JacchiaRoberts")
            self.jrdrag.SetField("HistoricWeatherSource", 'CSSISpaceWeatherFile')
            self.jrdrag.SetField("CSSISpaceWeatherFile", gmat_data_dir+"/SpaceWeather-v1.2.txt")
            # Build and set the atmosphere for the model
            self.atmos = gmat.Construct("JacchiaRoberts")
            self.jrdrag.SetReference(self.atmos)

        self.fm.AddForce(self.earthgrav)
        self.fm.AddForce(self.moongrav)
        self.fm.AddForce(self.sungrav)
        if with_jacchia:
            self.fm.AddForce(self.jrdrag)
        if with_SRP:
            self.fm.AddForce(self.srp)
        self.fm.Help()
        print(self.fm.GetGeneratingString(0))

        # Build Integrator
        self.gator = gmat.Construct("RungeKutta89", "Gator")
        # Build the propagation container that connect the integrator, force model, and spacecraft together
        self.pdprop = gmat.Construct("Propagator","PDProp")  
        # Create and assign a numerical integrator for use in the propagation
        self.pdprop.SetReference(self.gator)
        # Set some of the fields for the integration
        self.pdprop.SetField("InitialStepSize", self.dt)
        self.pdprop.SetField("Accuracy", 1.0e-13)
        self.pdprop.SetField("MinStep", 0.0)
        self.pdprop.SetField("MaxStep", self.dt)
        self.pdprop.SetField("MaxStepAttempts", 50)

        # Assign the force model to the propagator
        self.pdprop.SetReference(self.fm)
        # It also needs to know the object that is propagated
        self.pdprop.AddPropObject(self.sat)
        # Setup the state vector used for the force, connecting the spacecraft
        self.psm = gmat.PropagationStateManager()
        self.psm.SetObject(self.sat)
        self.psm.SetProperty("AMatrix")
        #psm.SetProperty("STM") #increases state size, but gives undefined data for STM
        self.psm.BuildState()
        # Finish the object connection
        self.fm.SetPropStateManager(self.psm)
        self.fm.SetState(self.psm.GetState())

        # Before Initialization, make two coordinate converters to help convert MJ2000Eq Frame to ECF Frame
        self.ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
        self.eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
        self.csConverter = gmat.CoordinateConverter()

        # Perform top level initialization
        gmat.Initialize()

        # Finish force model setup:
        ##  Map the spacecraft state into the model
        self.fm.BuildModelFromMap()
        ##  Load the physical parameters needed for the forces
        self.fm.UpdateInitialData()

        # Perform the integation subsysem initialization
        self.pdprop.PrepareInternals()
        # Refresh the integrator reference
        self.gator = self.pdprop.GetPropagator()
        self.is_model_constructed = True

    def format_datetime(self,t):
        t_format = "{} {} {} ".format(t.day, MonthDic2[t.month], t.year)
        if t.hour < 10:
            t_format += "0"
        t_format += str(t.hour) + ":"
        if t.minute < 10:
            t_format += "0"
        t_format += str(t.minute) + ":"
        if t.second < 10:
            t_format += "0"
        t_format += str(t.second) + "."
        millisec = str(np.round(t.microsecond / 1e6, 3)).split(".")[1]
        if millisec == "0":
            millisec = "000"
        t_format += millisec
        return t_format 

    def clear_model(self):
        gmat.Clear()
        self.is_model_constructed = False

    def reset_state(self, x, iter):
        assert x.size == (6 + len(self.solve_for_states))
        for j in range(len(self.solve_for_states)):
            self.solve_for_states[j] = x[6+j]
            val = self.solve_for_nominals[j] * ( 1 + self.solve_for_states[j] )
            self.sat.SetField(self.solve_for_fields[j], val)
        self.sat.SetState(*x[0:6])
        self.fm.BuildModelFromMap()
        self.fm.UpdateInitialData()
        self.pdprop.PrepareInternals()
        self.gator = self.pdprop.GetPropagator() # refresh integrator
        self.gator.SetTime(iter * self.dt)

    def reset_initial_state(self, x):
        self.x0 = x[0:6].copy()
        self.reset_state(x, 0)

    def transform_Earth_MJ2000Eq_2_BodyFixed(self, time_a1mjd, state_mj2000):
        state_in = gmat.Rvector6(*list(state_mj2000[0:6])) # State in MJ2000Eq
        state_out = gmat.Rvector6() # State in Earth Body Fixed Coordinates
        self.csConverter.Convert(time_a1mjd, state_in, self.eci, state_out, self.ecf)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array # ECF (Earth Coordinates Fixed)
    
    def transform_Earth_BodyFixed_2_MJ2000Eq(self, time_a1mjd, state_earth_bf):
        state_in = gmat.Rvector6(*list(state_earth_bf[0:6])) # State in Earth Body Fixed Coordinates
        state_out = gmat.Rvector6() # State in MJ2000Eq
        self.csConverter.Convert(time_a1mjd, state_in, self.ecf, state_out, self.eci)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array # Earth Coordinates Inertial
    
    def solve_for_state_jacobians(self, Jac, dv_dt):
        # Nominal dv_dt w/out parameter changes is inputted
        pstate = self.gator.GetState()
        eps = 0.0005
        for j in range(len(self.solve_for_states)):
            val = float( self.sat.GetField(self.solve_for_fields[j]) )
            # dv_dt with small parameter change
            self.sat.SetField(self.solve_for_fields[j], val+eps)
            self.fm.GetDerivatives(pstate, dt=self.dt, order=1)
            dv_eps_dt = np.array(self.fm.GetDerivativeArray()[3:6])
            Jac_sfs = (dv_eps_dt - dv_dt) / (eps/self.solve_for_nominals[j]) # Simple Model -- Can make derivative more robust
            Jac[3:6, 6+j] = Jac_sfs
            Jac[6+j, 6+j] = -1.0 / self.solve_for_taus[j]
            # reset nominal field
            self.sat.SetField(self.solve_for_fields[j], val)
        return Jac

    def get_jacobian_matrix(self):
        pstate = self.gator.GetState() #sat.GetState().GetState()
        self.fm.GetDerivatives(pstate, dt=self.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot = self.fm.GetDerivativeArray()
        dx_dt = np.array(fdot[0:6])
        num_sf = len(self.solve_for_states)
        num_x = 6 + num_sf
        Jac = np.zeros((num_x, num_x))
        # Add Position and Velocity State Jacobians 
        Jac[0:6,0:6] = np.array(fdot[6:42]).reshape((6,6))
        # Add solve for state Jacobians
        if num_sf > 0:
            Jac = self.solve_for_state_jacobians(Jac, dx_dt[3:])
        return Jac

    def get_transition_matrix(self, taylor_order):
        num_sf = len(self.solve_for_states)
        num_x = 6 + num_sf
        Jac = self.get_jacobian_matrix()
        Phi = np.eye(num_x) + Jac * self.dt
        for i in range(2, taylor_order+1):
            Phi += np.linalg.matrix_power(Jac, i) * self.dt**i / math.factorial(i)
        return Phi

    def step(self):
        self.gator.Step(self.dt)
        num_sf = len(self.solve_for_states)
        xk = np.zeros(6 + num_sf)
        xk[0:6] = np.array(self.gator.GetState())
        if num_sf > 0:
            xk[6:] = self.propagate_solve_fors(False)
        return xk

    def get_state(self):
        num_sf = len(self.solve_for_states)
        xk = np.zeros(6 + num_sf)
        return np.array(self.gator.GetState() + self.solve_for_states)

    def get_state6_derivatives(self):
        pstate = self.gator.GetState() #sat.GetState().GetState()
        self.fm.GetDerivatives(pstate, dt=self.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot = self.fm.GetDerivativeArray()
        dx_dt = np.array(fdot[0:6])
        return dx_dt
    
    # Set Other Estimation Variables
    def set_solve_for(self, field = "Cd", dist="gauss", scale = -1, tau = -1, alpha = None):
        assert self.is_model_constructed
        assert field in self.solve_for_fields_acceptable 
        assert dist in self.solve_for_dists_acceptable 
        assert scale > 0 
        assert tau > 0
        if dist == "sas":
            assert alpha is not None 
            assert alpha >= 1

        self.solve_for_states.append(0.0)
        self.solve_for_taus.append(tau)
        self.solve_for_fields.append(field)
        self.solve_for_dists.append(dist)
        self.solve_for_scales.append(scale)
        self.solve_for_alphas.append(alpha)
        self.solve_for_nominals.append( float( self.sat.GetField(field) ) )

    def get_solve_for_noise_sample(self, j):
        if self.solve_for_dists[j] == "gauss":
            return np.random.randn() * self.solve_for_scales[j]
        elif self.solve_for_dists[j] == "sas":
            return ce.random_symmetric_alpha_stable(self.solve_for_alphas[j], self.solve_for_scales[j], 0)
        else:
            print( "Solve for distribution {} has not been implemented in get_solve_for_noise_sample() function! Please add it!".format(self.solve_for_dists[j]) )
            exit(1)
    
    def propagate_solve_fors(self, with_add_noise = False):
        new_sf_states = []
        noises = []
        for j in range(len(self.solve_for_states)):
            tau = self.solve_for_taus[j]
            self.solve_for_states[j] = np.exp(-self.dt / tau) * self.solve_for_states[j]
            if with_add_noise:
                noise = self.get_solve_for_noise_sample(j)
                self.solve_for_states[j] += noise
                self.solve_for_states[j] = np.clip(self.solve_for_states[j], -0.99, np.inf) # cannot be <= -1
                noises.append(noise)
            new_sf_states.append( self.solve_for_states[j] )
            new_nom_val = self.solve_for_nominals[j] * (1 + self.solve_for_states[j])
            self.sat.SetField(self.solve_for_fields[j], new_nom_val)
        if with_add_noise:
            return new_sf_states, noises
        else:
            return new_sf_states

    def simulate(self, num_orbits, gps_std_dev, W=None, with_density_jumps = False, num_steps = None):
        assert self.is_model_constructed
        if with_density_jumps:
            assert "Cd" in self.solve_for_fields
        self.gps_std_dev = gps_std_dev 
        r0 = np.linalg.norm(self.x0[0:3])
        v0 = np.linalg.norm(self.x0[3:6])
        omega0 = v0/r0 # rad/sec (angular rate of orbit)
        orbital_period = 2.0*np.pi / omega0 #Period of orbit in seconds
        time_steps_per_period = (int)(orbital_period / self.dt + 0.50) # number of dt's until 1 revolution is made
        num_mission_steps = num_orbits * time_steps_per_period if num_steps is None else num_steps
        self.num_sim_steps = num_mission_steps
        with_solve_fors = len(self.solve_for_states) > 0
        with_sim_state_reset = (W is not None) or with_solve_fors
        # Measurement before propagation
        len_x = 6 + len(self.solve_for_states)
        x0 = np.array( list(self.x0) + self.solve_for_states )
        v0 = np.random.randn(3) * self.gps_std_dev
        z0 = x0[0:3] + v0
        states = [x0]
        msmt_noises = [v0]
        msmts = [z0]
        proc_noises = []
        # Begin loop for propagation
        for i in range(num_mission_steps):
            wk = np.zeros(len_x)
            xk = np.zeros(len_x)
            # Now step the integrator and get new state
            self.gator.Step(self.dt)
            xk[0:6] = np.array(self.gator.GetState())
            
            wk = np.zeros(len_x)
            # Propagate solve fors
            if with_solve_fors:
                xk[6:], wk[6:] = self.propagate_solve_fors(True)
                # If with added density jumps is toggled
                if with_density_jumps:
                    j = self.solve_for_fields.index("Cd")
                    if i == 1200:
                        wk[6+j] = 4.5
                        self.solve_for_states[j] += wk[6+j]
                        new_nom_val = self.solve_for_nominals[j] * (1 + self.solve_for_states[j])
                        self.sat.SetField(self.solve_for_fields[j], new_nom_val)
                        xk[6+j] = self.solve_for_states[j]
                    if i == 3500:
                        wk[6+j] = 1.5
                        self.solve_for_states[j] += wk[6+j]
                        new_nom_val = self.solve_for_nominals[j] * (1 + self.solve_for_states[j])
                        self.sat.SetField(self.solve_for_fields[j], new_nom_val)
                        xk[6+j] = self.solve_for_states[j]

            # Add process noise to pos/vel, if given
            if W is not None:
                noise = np.random.multivariate_normal(np.zeros(6), W)
                xk[0:6] += noise
                wk[0:6] = noise

            # If solve fors are declared, or process noise is added, need to update state of simulator
            if with_sim_state_reset:
                self.reset_state(xk, i+1)
            
            proc_noises.append(wk)
            states.append(xk)
            #Form measurement
            vk = np.random.randn(3) * self.gps_std_dev
            zk = xk[0:3] + vk
            msmts.append(zk)
            msmt_noises.append(vk)
        
        # Reset Solve Fors
        for j in range(len(self.solve_for_states)):
            self.solve_for_states[j] = 0.0
            self.sat.SetField(self.solve_for_fields[j], self.solve_for_nominals[j])

        # Reset Simulation to x0, and return state info
        self.reset_state(x0, 0)
        return np.array(states), np.array(msmts), np.array(proc_noises), np.array(msmt_noises)

# 0.) Time conversion function using GMAT
def time_convert(time_in, type_in, type_out):
    if type(time_in) == datetime:
        millisec = str(np.round(time_in.microsecond / 1e6, 3)).split(".")[1]
        _time_in = time_in.strftime("%d %b %Y %H:%M:%S") + "." + millisec
        is_in_gregorian = True
    elif type(time_in) == str:
        _time_in = time_in
        is_in_gregorian = True
    elif type(time_in) == float:
        _time_in = time_in
        is_in_gregorian = False
    else:
        print("Time In Type: ", type(time_in), " Not Supported! Input was", time_in)
        exit(1)
    timecvt = gmat.TimeSystemConverter.Instance()
    if is_in_gregorian:
        time_in_greg = _time_in
        time_in_mjd = timecvt.ConvertGregorianToMjd(_time_in)
    else:
        time_in_mjd = _time_in
        time_in_greg = timecvt.ConvertMjdToGregorian(_time_in)
    time_types = {"A1": timecvt.A1, "TAI": timecvt.TAI, "UTC" : timecvt.UTC, "TDB": timecvt.TDB, "TT": timecvt.TT}
    assert type_in in time_types.keys()
    assert type_out in time_types.keys()
    time_code_in = time_types[type_in]
    time_code_out = time_types[type_out]
    time_out_mjt = timecvt.Convert(time_in_mjd, time_code_in, time_code_out)
    time_out_greg = timecvt.ConvertMjdToGregorian(time_out_mjt)
    time_dic = {"in_greg" : time_in_greg, 
                "in_mjd" : time_in_mjd, 
                "out_greg": time_out_greg, 
                "out_mjd": time_out_mjt}
    return time_dic

# 1.) Load in GPS Data and time stamps 
def load_gps_from_txt(fpath):
    # See if cached pickle file already exists 
    fprefix, fname = fpath.rsplit("/", 1)
    name_substrs = fname.split(".")
    name_prefix = name_substrs[0]
    name_suffix = name_substrs[2]
    assert(name_suffix in ["navsol", "gmd"])
    with_UTC_format = name_suffix == "navsol" # otherwise gmd
    if with_UTC_format:
        pickle_fpath = fprefix + "/" + name_prefix + "_navsol" + ".pickle"
    else:
        pickle_fpath = fprefix + "/" + name_prefix + "_gmd" + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached GPS Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            gps_msmts = pickle.load(handle)
        return gps_msmts
    # Read in gps and pickle (so it doesnt have to be done again)
    else:
        gps_msmts = []
        with open(fpath, 'r') as handle:
            lines = handle.readlines()
            for line in lines:
                cols = line.split()
                if with_UTC_format:
                    year = int(cols[1])
                    month = int(cols[2])
                    day = int(cols[3])
                    hour = int(cols[4])
                    minute = int(cols[5])
                    second = int(cols[6])//1000
                    microsecond = (int(cols[6]) % 1000) * 1000
                    date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                    pos_x = float(cols[9])
                    pos_y = float(cols[10])
                    pos_z = float(cols[11])
                    pos = np.array([pos_x, pos_y, pos_z])
                    gps_msmts.append((date_time, pos))
                else:
                    print("Enter Formatting for TAI times")
                    exit(1)

        print("Writing Cached GPS Data To Pickle File at: ", pickle_fpath)
        with open(pickle_fpath, "wb") as handle:
            pickle.dump(gps_msmts, handle)
        return gps_msmts 

def load_glast_file(fpath):
    fprefix, fname = fpath.rsplit("/", 1)
    name_prefix = fname.split(".")[0]
    pickle_fpath = fprefix + "/" + name_prefix + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            times, means, covars = pickle.load(handle)
        return times, means, covars
    else:
        with open(fpath, 'r') as handle:
            lines = handle.readlines()

            # Find state size
            header = lines[0]
            header_cols = header.split(",")
            len_line = len(header_cols)
            c = -2*(len_line-1)
            n = -1.5 + (9 - 4*c)**0.5 / 2
            n = int(n + 0.99)
            # Find covariance indices
            idxs = []
            for i in range(n+1, len_line):
                label = header_cols[i]
                str_cov = label.split("_")
                idxs.append( (int(str_cov[1])-1, int(str_cov[2])-1) ) 

            times = [] 
            means = []
            covars = []

            for line in lines[1:]:
                # Time Creation
                sub_strs = line.split(",")
                str_date = sub_strs[0]
                date_list = str_date.split()
                day = int(date_list[0])
                month = MonthDic[date_list[1]]
                year = int(date_list[2])
                time_list = date_list[3].split(":")
                hour = int(time_list[0])
                minute = int(time_list[1])
                str_second, str_millisec = time_list[2].split(".")
                second = int(str_second)
                microsecond = int(str_millisec) * 1000
                date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                times.append(date_time)
                # Append Means 
                xk = np.zeros(n)
                for i in range(1,n+1):
                    xk[i-1] = float(sub_strs[i])
                means.append(xk)
                cholPk = np.zeros((n,n))
                for k, idx in enumerate(idxs):
                    i = idx[0]
                    j = idx[1]
                    cPij = float(sub_strs[k+n+1])
                    cholPk[i,j] = cPij
                Pk = cholPk @ cholPk.T
                covars.append(Pk)
        print("Writing Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "wb") as handle:
            pickle.dump((times, means, covars), handle)
        return times, means, covars
        
# 2.) If provided, scan GLAST csv to find the a-priori covariance closest to the first GPS reading (returns state before first GPS reading)
def find_restart_point(fpath, gps_datetime):
    fprefix, fname = fpath.rsplit("/", 1)
    name_prefix = fname.split(".")[0]
    gps_timetag = "_gps_{}_{}_{}_{}_{}_{}_{}".format(gps_datetime.year,gps_datetime.month,gps_datetime.day,gps_datetime.hour,gps_datetime.minute,gps_datetime.second,gps_datetime.microsecond)
    pickle_fpath = fprefix + "/" + name_prefix + gps_timetag + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached Restart Point From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            date_time, x0, P0, state_labels, cov_labels = pickle.load(handle)
        return date_time, x0, P0, state_labels, cov_labels
    else:
        with open(fpath, 'r') as handle:
            lines = handle.readlines()
            count = 0
            for line in lines:
                if count == 0:
                    count +=1 
                    continue
                str_date = line.split(",")[0]
                dt_list = str_date.split()
                day = int(dt_list[0])
                month = MonthDic[dt_list[1]]
                year = int(dt_list[2])
                time_list = dt_list[3].split(":")
                hour = int(time_list[0])
                minute = int(time_list[1])
                str_second, str_millisec = time_list[2].split(".")
                second = int(str_second)
                microsecond = int(str_millisec) * 1000
                date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                time_delt = gps_datetime - date_time
                if time_delt.days < 0:
                    break
                count += 1
            # Take the warm-start right before count's current index
            assert(count > 2) # count would need to be larger than 2
            count -= 2
            count = int( np.clip(count, 1, np.inf) )
            final_choice = lines[count]
            str_date = final_choice.split(",")[0]
            dt_list = str_date.split()
            day = int(dt_list[0])
            month = MonthDic[dt_list[1]]
            year = int(dt_list[2])
            time_list = dt_list[3].split(":")
            hour = int(time_list[0])
            minute = int(time_list[1])
            str_second, str_millisec = time_list[2].split(".")
            second = int(str_second)
            microsecond = int(str_millisec) * 1000
            date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
            # Find state size
            len_line = len(final_choice.split(","))
            c = -2*(len_line-1)
            n = -1.5 + (9 - 4*c)**0.5 / 2
            n = int(n + 0.99)
            substrs = final_choice.split(",")
            # Read in the state and cholesky of the covariance matrix
            x0 = np.zeros(n)
            cholP0 = np.zeros((n,n))
            labels = lines[0].split(",")
            for i in range(1, n+1):
                x0[i-1] = float( substrs[i] )
            idxs = []
            for i in range(n+1, len_line):
                label = labels[i]
                str_cov = label.split("_")
                idxs.append( (int(str_cov[1])-1, int(str_cov[2])-1) ) 
            for k, idx in enumerate(idxs):
                i = idx[0]
                j = idx[1]
                cPij = float(substrs[k+n+1])
                cholP0[i,j] = cPij
            # Recreate Covariance matrix from its cholesky 
            P0 = cholP0 @ cholP0.T
    state_labels = labels[1:n+1]
    cov_labels = labels[n+1:] 
    # Log this start state to a pickled file
    # Enter code
    print("Writing Cached Restart Point From Pickled File at: ", pickle_fpath)
    with open(pickle_fpath, "wb") as handle:
            pickle.dump((date_time, x0, P0, state_labels, cov_labels), handle)
    return date_time, x0, P0, state_labels, cov_labels

# 3.) If GLAST csv not provided, we may need to run a small nonlinear least squares to find a passible initial state hypothesis            
def estimate_restart_stats(gps_msmts):
    exit(1)

# 4.) Run KF Then KF Smoother, log results of run as pickle
def run_fermi_kalman_filter_and_smoother(gps_msmts, t0, x0, P0):
    fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    fermi_dt = 60.0
    fermi_x0 = x0.copy() 
    fermi_P0 = P0.copy()
    fermi_Cd_sigma = 0.0013
    fermi_Cd_sigma_scale = 10000
    fermi_gps_std_dev = 7.5 / 1e3 # m -> km
    fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for(field="Cd", dist="sas", scale=fermi_Cd_sigma, tau = 21600, alpha=1.3)
    
    
    # Run Kalman Filter Forwards 
    I7 = np.eye(7)
    H = np.zeros((3,7))
    H[0:3,0:3] = np.eye(3)
    V = np.eye(3) * fermi_gps_std_dev
    #q = 8e-15
    #Qt = np.eye(7) * q
    #Qt[6,6] = fermi_Cd_sigma * 160
    #Gt = np.zeros((7,4))
    #Gt[3:,:] = np.eye(4)
    Wk = np.zeros((7,7))
    Wk[:6,:6] = leo6_process_noise_model(fermi_dt)
    Wk[6,6] = (fermi_Cd_sigma)**2 * fermi_Cd_sigma_scale
    x_kf = fermi_x0.copy() 
    P_kf = fermi_P0.copy() 
    xs_kf = [fermi_x0.copy()]
    Ps_kf = [fermi_P0.copy()]
    #Ms_kf = [fermi_P0.copy()]
    #Phis_kf = [] 
    tkm1 = t0
    for gps_msmt in gps_msmts:
        tk, _zk = gps_msmt
        zk = _zk.copy() / 1000 # m -> km 
        t_delta = tk - tkm1 
        dt_step = t_delta.seconds + (t_delta.microseconds / 1e6)
        # Conduct prop_steps of propagation to next time step (given by GPS)
        prop_steps = int( dt_step / fermi_dt )
        for i in range(prop_steps):
            Phi_k = fermiSat.get_transition_matrix(taylor_order=3)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wk
            x_kf = fermiSat.step()
        # Conduct sub-propagation steps to next time step (given by GPS)
        sub_dt = dt_step % fermi_dt
        if sub_dt != 0:
            Wk_sub = np.zeros((7,7))
            Wk_sub[0:6,0:6] = leo6_process_noise_model(sub_dt)
            Wk_sub[6,6] = (fermi_Cd_sigma * fermi_Cd_sigma_scale) * (sub_dt/fermi_dt)
            fermiSat.dt = sub_dt
            Phi_k = fermiSat.get_transition_matrix(taylor_order=3)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wk_sub
            x_kf = fermiSat.step()
            fermiSat.dt = fermi_dt
        # Conduct Measurement Update 
        zbar = H @ x_kf
        rk = zk - zbar
        rk_norm = np.linalg.norm(rk)
        print("Norm residual: ",  rk_norm)
        # Reject rk if we have the case rk is way to large (bad msmt time stamp)
        if(rk_norm < .03):
            K_kf = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T
            x_kf = x_kf + K_kf @ rk
            P_kf = (I7 - K_kf @ H) @ P_kf @ (I7 - K_kf @ H).T + K_kf @ V @ K_kf.T 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
            fermiSat.reset_state(x_kf, i) #/1000)
        else:
            print("At GPS Measurement Time: ", tk, "rk_norm too large! Rejecting Measurement!")
        # Append 
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
        # Update last time step to point to the current time instance
        tkm1 = tk

    # Now run the smoother backwards
    assert(False)

# 5.) Read outputted GMAT log of their Filter and Smoother xlsx file
def read_gmat_filter_output(fpath): 
    pass

# Debugging the GPS transformation required
def find_time_match(t_gps, ts_glast, start_idx = 0):
    count = 0
    for t_glast in ts_glast[start_idx:]:
        dt = t_gps - t_glast
        if (dt.days == 0) and (dt.seconds == 0):
            return count + start_idx
        count +=1
    print("No GLast Time Found to Be Same!")
    assert(False)

def test_gps_transformation(t0, x0, gps_msmts, inputted_glast_data):
    test_len = 10
    #mod = gmat.Moderator.Instance()
    #ss = mod.GetDefaultSolarSystem()
    #earth = ss.GetBody('Earth')
    #eop_file_path = gmat_data_dir + "eop_file.txt"
    #earth.SetField('EopFileName', eop_file_path)
    fixedState = gmat.Rvector6()
    ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
    eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
    csConverter = gmat.CoordinateConverter()
    #gmat.Initialize()

    fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    fermi_dt = 65.0
    fermi_x0 = x0.copy() 
    fermi_gps_std_dev = 7.5 / 1e3 # m -> km
    fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    x_next = fermiSat.step()

    t_idx = 0
    glast_times = inputted_glast_data[0]
    glast_means = inputted_glast_data[1]
    for i in range(test_len):
        # Find the closest glast vector. These time stamps should be identical
        gps_msmt = gps_msmts[i]
        date = gps_msmt[0]
        msmt = gps_msmt[1]
        t_idx = find_time_match(date, glast_times, start_idx = t_idx)
        cmp_vec = glast_means[t_idx]
        #cmp_vec = x_next.copy()

        # Create Transformed GPS Msmt in Earth MJ2000Eq Coordinates
        epoch = gmat.UtcDate(date.year, date.month, date.day, date.hour, date.minute, date.second + date.microsecond / 1e6)
        #_rvec = [*(msmt/1000), 0,0,0]
        _rvec = list(cmp_vec[0:6])
        rvec = gmat.Rvector6( *_rvec )
        
        # Foo Loop
        #for i in range(0,1000,5):
        #    time_a1mjd = epoch.ToA1Mjd() + 1.1574065865715965e-08*i
        #    csConverter.Convert(time_a1mjd, rvec, ecf, fixedState, eci)
        #    outvec = np.array([fixedState[0], fixedState[1], fixedState[2], fixedState[3], fixedState[4], fixedState[5]])
        #    # Print difference between transformation in meters
        #    diff_vec = (outvec[0:3] - cmp_vec[0:3]) * 1000
        #    print("Difference: i= ",i, ": ", diff_vec)
        #time_dic_tai = time_convert(date, "UTC", "TAI")
        time_dic_a1 = time_convert(date, "UTC", "A1")
        time_a1mjd = time_dic_a1["out_mjd"] #time_a1mjd = epoch.ToA1Mjd()
        #time_a1mjd += 1.1572410585358739e-08*210
        #csConverter.Convert(time_a1mjd, rvec, ecf, fixedState, eci)
        csConverter.Convert(time_a1mjd, rvec, eci, fixedState, ecf)



        outvec = np.array([fixedState[0], fixedState[1], fixedState[2], fixedState[3], fixedState[4], fixedState[5]])
        # Print difference between transformation in meters
        #diff_vec = (outvec[0:3] - cmp_vec[0:3]) * 1000
        diff_vec = (outvec[0:3]* 1000 - msmt[0:3])
        print("Difference: ", diff_vec)
    return outvec

# 3.) Run KF over subinterval
# 4.) Predict out KF subinterval estimates over given horizon 
# 5.) Score likelihood of projected estimate to ephem est, and score likelihood of position 
def test_all():
    gps_path =  gmat_data_dir + "G_navsol_from_gseqprt_2023043_2023137_thinned_stitched.txt.navsol"
    restart_path = gmat_data_dir + "Sat_GLAST_Restart_20230212_094850.csv"
    gps_msmts = load_gps_from_txt(gps_path)
    inputted_glast = load_glast_file(restart_path)
    t0, x0, P0, labels_x0, labels_P0 = find_restart_point(restart_path, gps_msmts[0][0])
    #run_fermi_kalman_filter_and_smoother(gps_msmts, t0, x0[:7], P0[:7,:7])
    test_gps_transformation(t0, x0, gps_msmts, inputted_glast)
    print("Thats all folks!")

def test_time_convert():
    time_in = datetime(2023, 2, 12, 0, 30, 29, 1000) #"11 Feb 2023 23:49:00.000" # 29987.521168993055
    type_in = "UTC"
    type_out = "A1"
    time_dic = time_convert(time_in, type_in, type_out)
    print("In {} Greg: {}".format(type_in, time_dic["in_greg"]) )
    print("In {} MJD: {}".format(type_in, time_dic["in_mjd"]) )
    print("Out {} Greg: {}".format(type_out, time_dic["out_greg"]) )
    print("In {} MJD: {}".format(type_out, time_dic["out_mjd"]) )

def test_single_gps_msmt():
    mod = gmat.Moderator.Instance()
    ss = mod.GetDefaultSolarSystem()
    earth = ss.GetBody('Earth')
    eop_file_path = gmat_data_dir + "eop_file.txt"
    earth.SetField('EopFileName', eop_file_path)
    in_pos_vec = [4656.8449747241457, 4230.0931453206676, 2811.4539689390904, 0, 0, 0]
    time_a1mjd = 29987.492789749787
    in_gps_vec = gmat.Rvector6(*in_pos_vec)
    ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
    eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
    csConverter = gmat.CoordinateConverter()
    gmat.Initialize()
    tmp_gps_vec = gmat.Rvector6()
    csConverter.Convert(time_a1mjd, in_gps_vec, eci, tmp_gps_vec, ecf)
    out_gps_vec = np.array([tmp_gps_vec[0],tmp_gps_vec[1],tmp_gps_vec[2],tmp_gps_vec[3],tmp_gps_vec[4],tmp_gps_vec[5]])
    print("Computed Value: ", out_gps_vec)
    known_gps_vec = np.array([-705.78640593524074, -6246.8126277142010, 2821.9433196988944, 0, 0, 0])
    print("True Value: ", known_gps_vec)
    print("Residual", 1000*(known_gps_vec[0:3] - out_gps_vec[0:3]) )

# Start of prediction application function 
# This seems to work really nicely, converging to the limit of gmats step
def iterative_time_closest_approach(dt, _t0, prim_tup, sec_tup, start_idx = 0, its = -1, with_plot=True):
    # For now this function assumes an integer time step
    assert(int(dt) == dt)
    # Initial iteration
    p_pks,p_vks,p_aks = copy.deepcopy(prim_tup)
    s_pks,s_vks,s_aks = copy.deepcopy(sec_tup)
    t0 = copy.deepcopy(_t0)
    tks = np.arange(p_pks.shape[0]) * dt
    i, troot, t_c, pp_c, pv_c, sp_c, sv_c = pc.closest_approach_info(tks[start_idx:], 
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
    print("t0: ", t0, "(sec)")
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
        fermiSat = FermiSatelliteModel(t0, x0_prim, dt)
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
        fermiSat = FermiSatelliteModel(t0, x0_sec, dt)
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
        i, troot, t_c, pp_c, pv_c, sp_c, sv_c = pc.closest_approach_info(tks, (p_pks,p_vks,p_aks), (s_pks,s_vks,s_aks))

        print("Iteration: ", it + 2, " (sec)")
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


    
    # i_star_lhs -> The nominal propagation index to stop at 
    # t_lhs -> The time at the nominal propagation index 
    # t_c -> The final time of closest approach 
    # pp_c -> The position of the primary at closest approach 
    # pv_c -> The velocity of the primary at closest approach 
    # sp_c -> The position of the secondary at closest approach
    # sv_c -> The velocity of the secondary at closest approach 
    return i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c

def find_cross_radial_error_mc_vs_density():
    # Given an initial condition, simulate out different realizations of change in atms density and look at the monte carlo avg of error vs nominal (no density change) over a week lookahead
    x0 = np.array([4996.245288270519, 3877.946463086103, 2736.0432364171807, -5.028093574446193, 5.575921341999267, 1.2698611722905329])
    P0 = np.array([ [ 1.05408888e-05, -8.97284021e-06, -1.89319050e-06, 1.10789874e-08,  6.77331750e-09,  5.31534425e-09],
                    [-8.97284021e-06,  9.49191574e-06,  1.31370671e-06, -9.52652261e-09, -7.09634679e-09, -5.12992550e-09],
                    [-1.89319050e-06,  1.31370671e-06,  1.91941294e-06, -1.62654495e-09, -1.54468399e-09, -9.88335840e-10],
                    [ 1.10789874e-08, -9.52652261e-09, -1.62654495e-09, 1.21566585e-11,  7.54301726e-12,  5.41815567e-12],
                    [ 6.77331750e-09, -7.09634679e-09, -1.54468399e-09, 7.54301726e-12,  6.14186208e-12,  3.36503199e-12],
                    [ 5.31534425e-09, -5.12992550e-09, -9.88335840e-10, 5.41815567e-12,  3.36503199e-12,  4.76717851e-12]])
    t0 = '11 Feb 2023 23:47:55.0'
    dt = 120.0
    mc_trials = 30
    days_lookahead = 7
    mode = 'gauss'
    orbit_period = 2*np.pi*np.linalg.norm(x0[0:3])/np.linalg.norm(x0[3:6]) # seconds in orbit
    prop_time = (days_lookahead * 24 * 60 * 60) # seconds in days_lookahead days 
    num_orbits = int( np.ceil( prop_time / orbit_period) )
    total_steps = int( prop_time / dt )
    step_lookouts = np.array( [ int( (i*24*60*60)/dt ) -1 for i in range(1,days_lookahead+1)] )


    # Take KF and get filtered result after a day 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_sastrials_2_1713828844.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    start_idx = 1000
    xs_kf,Ps_kf = mc_dic["ekf_runs"][0]
    # Now store cov of estimator at the end of filter period
    x0 = xs_kf[start_idx][0:6]
    P0 = Ps_kf[start_idx][0:6,0:6]

    # Now propagate the state estimate days_lookahead days into the future assuming no atms. density change 
    fermiSat = FermiSatelliteModel(t0,x0,dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xkf_preds = [] 
    Pkf_preds = [] 
    P_kf = P0.copy()
    for idx in range(total_steps):
        Phi = fermiSat.get_transition_matrix(mc_dic["STM_order"])
        x_kf = fermiSat.step()
        P_kf = Phi @ P_kf @ Phi.T
        xkf_preds.append(x_kf)
        Pkf_preds.append(P_kf)
    xkf_preds = np.array(xkf_preds)
    Pkf_preds = np.array(Pkf_preds)
    # Store xhats each day
    xbar_lookouts = xkf_preds[step_lookouts]
    Pbar_lookouts = Pkf_preds[step_lookouts]
    Ts = [pc.get_along_cross_radial_transformation(x) for x in xbar_lookouts]
    Pbar_Ts = np.array([T @ P[0:3,0:3] @ T.T for T,P in zip(Ts,Pbar_lookouts)])
    std_dev_Pbar_Ts = np.array([np.diag(P)**0.5 for P in Pbar_Ts])
    fermiSat.clear_model()

    # Now use a monte carlo and propagate the state estimate days_lookahead days into the future assuming atms. denisty change on the given distribution
    mc_data = np.zeros((mc_trials, len(step_lookouts), 6))
    fermiSat = FermiSatelliteModel(t0,x0,dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for(field="Cd", dist=mode, scale = 0.0013, tau = 21600, alpha = 2.0 if mode == "gauss" else 1.3)
    for i in range(mc_trials):
        print("Finished Trial", i)
        xks, _, _, ws = fermiSat.simulate(num_orbits, 0, None, False)
        mc_data[i, :,:] = xks[step_lookouts+1,0:6]
        x07 = np.concatenate((x0,np.array([0.0])))
        fermiSat.reset_initial_state(x07)
    # See via MC what the error bound is for cross and radial error after one day to seven days vs. Kalman filter 
    mc_data -= xbar_lookouts#.reshape((1,xbar_lookouts.size,1))
    mc_data = mc_data.transpose((1,2,0))
    Ppred_mcs = [np.cov(mc_data[i])[0:3,0:3] for i in range(days_lookahead)]
    Ts = [pc.get_along_cross_radial_transformation(x) for x in xbar_lookouts]
    Ppred_Ts = np.array([T @ P @ T.T for T,P in zip(Ts,Ppred_mcs)])
    std_dev_Ppred_Ts = np.array([np.diag(P)**0.5 for P in Ppred_Ts])

    # Now plot the covariances as a function of look ahead and vs KF
    plt.suptitle("Along Cross Radial (ACR)-Track KF Variance Projected 7 days (b) vs 7-day ACR-Track MC Variance (r)")
    plt.subplot(311)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,0], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,0], 'r')
    plt.subplot(312)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,1], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,1], 'r')
    plt.subplot(313)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,2], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,2], 'r')
    plt.show()
    foobar=2

def test_sat_pc():
    # Load out data 
    gps_path =  gmat_data_dir + "G_navsol_from_gseqprt_2023043_2023137_thinned_stitched.txt.navsol"
    restart_path = gmat_data_dir + "Sat_GLAST_Restart_20230212_094850.csv"
    gps_msmts = load_gps_from_txt(gps_path)
    inputted_glast = load_glast_file(restart_path)
    t0, x0, P0, labels_x0, labels_P0 = find_restart_point(restart_path, gps_msmts[0][0])
    x0 = x0[0:6]
    P0 = P0[0:6,0:6]
    dt = 60.0
    std_gps_noise = 7.5 / 1e3
    # Construct start conditions for primary
    x0_prim = x0.copy()
    R_prim = 0.003 #km
    P0_prim = P0.copy()
    # Construct start conditions for secondary
    x0_sec = x0.copy()
    x0_sec[3:] *= -1.0
    R_sec = 0.003 #km
    P0_sec = P0.copy()
    # Number of steps
    steps = int(55*(60/dt) + 0.50)

    # Create Satellite Model for Primary and Propagate 
    fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    fermiSat = FermiSatelliteModel(t0, x0_prim, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_pks = [] # Primary Pos 3-Vec
    p_vks = [] # Primary Vel 3-Vec
    p_aks = [] # Primary Acc 3-Vec
    p_Pks = []
    # Propagate Primary and Store
    xk = x0_prim.copy()
    Pk = P0_prim.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        p_pks.append(xk[0:3])
        p_vks.append(xk[3:6])
        p_aks.append(dxk_dt[3:6])
        p_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    p_pks = np.array(p_pks)
    p_vks = np.array(p_vks)
    p_aks = np.array(p_aks)
    p_Pks = np.array(p_Pks)
    fermiSat.clear_model()

    # Create Satellite Model for Secondary and Propagate
    fermiSat = FermiSatelliteModel(t0, x0_sec, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_pks = [] # Secondary Pos 3-Vec
    s_vks = [] # Secondary Vel 3-Vec
    s_aks = [] # Secondary Acc 3-Vec
    s_Pks = []
    # Propagate Secondary and Store
    xk = x0_sec.copy()
    Pk = P0_sec.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        s_pks.append(xk[0:3])
        s_vks.append(xk[3:6])
        s_aks.append(dxk_dt[3:6])
        s_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    s_pks = np.array(s_pks)
    s_vks = np.array(s_vks)
    s_aks = np.array(s_aks)
    s_Pks = np.array(s_Pks)
    fermiSat.clear_model()

    # Now test the closest approach method
    start_idx = 25
    
    '''
    tks = dt * np.arange(steps)
    i, troot, t_c, pp_c, sp_c = pc.closest_approach_info(tks[start_idx:], 
        (p_pks[start_idx:,:],p_vks[start_idx:,:],p_aks[start_idx:,:]), 
        (s_pks[start_idx:,:],s_vks[start_idx:,:],s_aks[start_idx:,:]))
    i += start_idx
    print("Step dt: ", dt)
    print("Tc: {}, Idx of Tc: {}".format(t_c, i) )
    print("Primary at Tc: ", pp_c)
    print("Secondary at Tc: ", sp_c)
    print("Pos Diff is: ", pp_c-sp_c)
    print("Pos Norm is: ", np.linalg.norm(pp_c-sp_c))
    # Plot orbits of primary and secondary
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    # Plot relative difference in position
    fig2 = plt.figure()
    plt.title("Pos Norm Diff over Time")
    #plt.plot(tks, p_pks[:,0] - s_pks[:,0], 'r')
    #plt.plot(tks, p_pks[:,1] - s_pks[:,1], 'g')
    #plt.plot(tks, p_pks[:,2] - s_pks[:,2], 'b')
    plt.plot(tks, np.linalg.norm(p_pks-s_pks,axis=1))
    plt.show()
    foo=3
    '''
    #'''
    # GMAT iterative closest time of approach
    i_star_lhs, t_lhs, t_c, pp_c, sp_c = iterative_time_closest_approach(
        dt, t0, 
        (p_pks,p_vks,p_aks), 
        (s_pks,s_vks,s_aks), 
        start_idx = start_idx,
        with_plot=False
        )
    #'''

    # Propagate the Primary Covariance to Point of Closest Approach
    x0_prim = np.concatenate((p_pks[i_star_lhs],p_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(t_lhs), x0_prim, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
    p_xtc = fermiSat.step()
    print("XPrim at TCA: ", p_xtc[0:3])
    print("Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
    fermiSat.clear_model()

    # Propagate the Secondary Covariance to Point of Closest Approach 
    x0_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(t_lhs), x0_sec, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    s_Ptc = s_Phi @ s_Pks[i_star_lhs] @ s_Phi.T
    s_xtc = fermiSat.step()
    print("XSec at TCA: ", s_xtc[0:3])
    print("Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
    fermiSat.clear_model()


    # Form Relative System and Project onto Encounter Plane
    rx_tc = s_xtc - p_xtc
    rP_tc = s_Ptc + p_Ptc
    rR = R_prim + R_sec
    
    # plane normal to relative velocity vector 
    _,_,T1 = np.linalg.svd(rx_tc[3:].reshape((1,3)))
    T1 = np.hstack((T1[1:], np.zeros((2,3))))
    rx_ep = T1 @ rx_tc # mean 
    rP_ep = T1 @ rP_tc @ T1.T #variance 
    rxx = rx_ep[0]
    rxy = rx_ep[1]

    
    # Possibly another way to do this 
    T2 = pc.get_along_cross_radial_transformation(s_xtc)
    T2 = np.hstack((T2[1:,:], np.zeros((2,3))))
    rx_ep2 = T2 @ rx_tc # mean
    rP_ep2 = T2 @ rP_tc @ T2.T # variance 

    # Take Integral over 2D projection 
    int_coeff = 1.0 / (2.0*np.pi * np.linalg.det(rP_ep)**0.5 )
    int_PI = np.linalg.inv(rP_ep)

    from scipy.integrate import dblquad
    area = dblquad(lambda x, y: int_coeff*np.exp(-1.0/2.0 * np.array([x-rxx,y-rxy]) @ int_PI @  np.array([x-rxx,y-rxy]) ), -rR, rR, lambda x: -(rR**2-x**2), lambda x: (rR**2-x**2) )
    print("Prob Collision Stat is: ", area)

    foobar = 3

def test_sat_crossing():
    # Load out data 
    t0 = '11 Feb 2023 23:47:55.0'
    x0 = np.array([550+6378, 0, 0, 0, 7.585175924227056, 0])
    dt = 120.0
    # Construct start conditions for primary
    x0_prim = x0.copy()
    R_prim = 0.003 #km
    #P0_prim = P0.copy()
    # Construct start conditions for secondary
    x0_sec = x0.copy()
    x0_sec[3:] *= -1.0
    R_sec = 0.003 #km
    # Number of steps
    steps = 5040

    # Create Satellite Model for Primary and Propagate 
    fermiSat = FermiSatelliteModel(t0, x0_prim, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_pks = [] # Primary Pos 3-Vec
    p_vks = [] # Primary Vel 3-Vec
    p_aks = [] # Primary Acc 3-Vec
    #p_Pks = []
    # Propagate Primary and Store
    xk = x0_prim.copy()
    #Pk = P0_prim.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        p_pks.append(xk[0:3])
        p_vks.append(xk[3:6])
        p_aks.append(dxk_dt[3:6])
        #p_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    p_pks = np.array(p_pks)
    p_vks = np.array(p_vks)
    p_aks = np.array(p_aks)
    #p_Pks = np.array(p_Pks)
    fermiSat.clear_model()

    # Create Satellite Model for Secondary and Propagate
    fermiSat = FermiSatelliteModel(t0, x0_sec, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_pks = [] # Secondary Pos 3-Vec
    s_vks = [] # Secondary Vel 3-Vec
    s_aks = [] # Secondary Acc 3-Vec
    s_Pks = []
    # Propagate Secondary and Store
    xk = x0_sec.copy()
    #Pk = P0_sec.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        s_pks.append(xk[0:3])
        s_vks.append(xk[3:6])
        s_aks.append(dxk_dt[3:6])
        #s_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    s_pks = np.array(s_pks)
    s_vks = np.array(s_vks)
    s_aks = np.array(s_aks)
    #s_Pks = np.array(s_Pks)
    fermiSat.clear_model()

    #'''
    # Plot Trajectories of both satellites 
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    ax.scatter(p_pks[0,0], p_pks[0,1], p_pks[0,2], color = 'k', s=80)
    fig2 = plt.figure()
    r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
    plt.plot(np.arange(steps), r_norms)
    plt.show()
    foobar = 5
    #'''

    # Now test the closest approach method
    #'''
    start_idx = 4920
    end_idx = 4950 #p_pks.shape[0]
    # GMAT iterative closest time of approach
    t0 =  datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f")
    i_star_lhs, t_lhs, t_c, pp_c, sp_c = iterative_time_closest_approach(
        dt, t0, 
        (p_pks[:end_idx],p_vks[:end_idx],p_aks[:end_idx]), 
        (s_pks[:end_idx],s_vks[:end_idx],s_aks[:end_idx]), 
        start_idx = start_idx,
        with_plot=False
        )
    
    # Propagate the Primary Covariance to Point of Closest Approach
    x0_prim = np.concatenate((p_pks[i_star_lhs],p_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(seconds = t_lhs), x0_prim, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    #p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    #p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
    p_xtc = fermiSat.step()
    print("XPrim at TCA: ", p_xtc[0:3])
    print("Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
    fermiSat.clear_model()
    
    # Propagate the Secondary Covariance to Point of Closest Approach 
    x0_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(seconds = t_lhs), x0_sec, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_xtc = fermiSat.step()
    print("XSec at TCA: ", s_xtc[0:3])
    print("Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
    fermiSat.clear_model()
    #'''

    # Now draw 

    '''
    # Form Relative System and Project onto Encounter Plane, Compute Prob
    rx_tc = s_xtc - p_xtc
    rP_tc = s_Ptc + p_Ptc
    rR = R_prim + R_sec
    '''

def test_sat_pc_mc():
    data_path = file_dir + "/pylog/gmat7/pred/pc/"
    # Some Initial Data and Declarations
    t0 = '11 Feb 2023 23:47:55.0'
    x0 = np.array([550+6378, 0, 0, 0, 7.585175924227056, 0])
    filt_dt = 60
    filt_orbits = 12
    pred_dt = 120.0
    pred_steps = 5040
    n = 7 # states
    R_prim = 0.003 #km
    R_sec = 0.003 #km
    mode = "sas" #"sas"
    std_gps_noise = .0075 # kilometers

    # Runtime Options
    with_filter_plots = False
    with_pred_plots = False
    cached_dir = "" #"gauss_realiz.pickle" # "" # if set to something, loads and adds to this dir, if set to nothing, creates a new directory 
    with_density_jumps = False
    
    # Set additional solve for states
    std_Cd = 0.0013
    tau_Cd = 21600
    sas_Cd = mode
    # STM Taylor Order Approx
    STM_order = 3
    # Process Noise Model
    W6 = leo6_process_noise_model2(filt_dt)
    Wn = np.zeros((n,n))
    if sas_Cd == "gauss":
        scale_pv = 1.0*1e3
        scale_d = 20.0
        sas_alpha = 2.0
    else:
        scale_pv = 10000
        scale_d = 10000
        sas_alpha = 1.3
        #scale_pv = 500
        #scale_d = 250
    Wn[0:6,0:6] = W6.copy()
    Wn[0:6,0:6] *= scale_pv
    # Process Noise for changes in Cd
    if sas_Cd != "gauss":
        Wn[6,6] = (1.3898 * std_Cd)**2 # tune to cauchy LSF
    else:
        Wn[6,6] = std_Cd**2
    Wn[6,6] *= scale_d #0 # Tunable w/ altitude
    V = np.eye(3) * std_gps_noise**2
    I7 = np.eye(7)
    H = np.hstack((np.eye(3), np.zeros((3,4))))

    if cached_dir is "":
        cache_dic = {
            'mode' : mode,
            'with_density_jumps' : with_density_jumps,
            'R_prim' : R_prim, 'R_sec' : R_sec,
            't0' : t0,
            'x0_P0_prim' : None,
            'x0_P0_sec' : None,
            'kf_prim_sim' : None,
            'kf_sec_sim' : None,
            'kf_prim' : None,
            'kf_sec' : None,
            'filt_dt' : filt_dt,
            'prim_pred_hist' : None,
            'sec_pred_hist' : None,
            'pred_dt' : pred_dt,
            'itca_window_idxs' : None,
            'itca_data' : None, 
            'nom_prim_tca' : None,
            'nom_sec_tca' : None,
            'mc_prim_tcas' : None, 
            'mc_sec_tcas' : None
        }
    else:
        with open(data_path + cached_dir, 'rb') as handle:
            cache_dic = pickle.load(handle)
    
    # Filtering for primary satellite
    if cached_dir is "":
        # Create Satellite Model of Primary Satellite
        p_x0 = x0.copy()
        fermiSat = FermiSatelliteModel(t0, p_x0, filt_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)

        # Simulate Primary, Setup its Kalman Filter, and Run
        p_xs, p_zs, p_ws, p_vs = fermiSat.simulate(filt_orbits, std_gps_noise, W=None, with_density_jumps=with_density_jumps)
        cache_dic['kf_prim_sim'] = (p_xs.copy(), p_zs.copy(), p_ws.copy(), p_vs.copy())
        P_kf = np.eye(n) * (0.001)**2
        P_kf[6,6] = .01
        x_kf = np.random.multivariate_normal(p_xs[0], P_kf)
        x_kf[6] = 0
        cache_dic['x0_P0_prim'] = (x_kf.copy(), P_kf.copy())
        fermiSat.reset_state(x_kf, 0)
        p_xs_kf = [x_kf.copy()]
        p_Ps_kf = [P_kf.copy()]
        N = p_zs.shape[0]
        for i in range(1, N):
            # Time Prop
            Phi_k = fermiSat.get_transition_matrix(STM_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
            x_kf = fermiSat.step() 
            # Measurement Update
            K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar = H @ x_kf
            zk = p_zs[i]
            r = zk - zbar 
            print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(p_xs[i] - x_kf))
            x_kf = x_kf + K @ r 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
            fermiSat.reset_state(x_kf, i) #/1000)
            P_kf = (I7 - K @ H) @ P_kf @ (I7 - K @ H).T + K @ V @ K.T 
            # Log
            p_xs_kf.append(x_kf.copy())
            p_Ps_kf.append(P_kf.copy())
        p_xs_kf = np.array(p_xs_kf)
        p_Ps_kf = np.array(p_Ps_kf)
        fermiSat.clear_model()
        cache_dic['kf_prim'] = (p_xs_kf.copy(), p_Ps_kf.copy())
    else:
        p_xs, p_zs, p_ws, p_vs = cache_dic['kf_prim_sim']
        p_x0_kf, p_P0_kf = cache_dic['x0_P0_prim']
        p_xs_kf, p_Ps_kf = cache_dic['kf_prim']
    if with_filter_plots:
        # Plot Primary 
        print("Primary Sateliite KF Run:")
        ce.plot_simulation_history(None, (p_xs, p_zs, p_ws, p_vs), (p_xs_kf, p_Ps_kf), scale=1)

    # Now Repeat for secondary Satellite
    if cached_dir is "":
        s_x0 = x0.copy()
        s_x0[3:] *= -1
        fermiSat = FermiSatelliteModel(t0, s_x0, filt_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        # Simulate Secondary, Setup its Kalman Filter, and Run
        s_xs, s_zs, s_ws, s_vs = fermiSat.simulate(filt_orbits, std_gps_noise, W=None, with_density_jumps=with_density_jumps)
        cache_dic['kf_sec_sim'] = (s_xs.copy(), s_zs.copy(), s_ws.copy(), s_vs.copy())
        P_kf = np.eye(n) * (0.001)**2
        P_kf[6,6] = .01
        x_kf = np.random.multivariate_normal(s_xs[0], P_kf)
        x_kf[6] = 0
        cache_dic['x0_P0_sec'] = (x_kf.copy(), P_kf.copy())
        fermiSat.reset_state(x_kf, 0)
        s_xs_kf = [x_kf.copy()]
        s_Ps_kf = [P_kf.copy()]
        N = s_zs.shape[0]
        for i in range(1, N):
            # Time Prop
            Phi_k = fermiSat.get_transition_matrix(STM_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
            x_kf = fermiSat.step() 
            # Measurement Update
            K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar = H @ x_kf
            zk = s_zs[i]
            r = zk - zbar 
            print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(s_xs[i] - x_kf))
            x_kf = x_kf + K @ r 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
            fermiSat.reset_state(x_kf, i) #/1000)
            P_kf = (I7 - K @ H) @ P_kf @ (I7 - K @ H).T + K @ V @ K.T 
            # Log
            s_xs_kf.append(x_kf.copy())
            s_Ps_kf.append(P_kf.copy())
        s_xs_kf = np.array(s_xs_kf)
        s_Ps_kf = np.array(s_Ps_kf)
        cache_dic['kf_sec'] = (s_xs_kf.copy(), s_Ps_kf.copy())
        fermiSat.clear_model()
    else:
        s_xs, s_zs, s_ws, s_vs = cache_dic['kf_sec_sim']
        s_x0_kf, s_P0_kf = cache_dic['x0_P0_sec']
        s_xs_kf, s_Ps_kf = cache_dic['kf_sec']
    if with_filter_plots:
        # Plot Secondary 
        print("Secondary Satelite KF Run:")
        ce.plot_simulation_history(None, (s_xs, s_zs, s_ws, s_vs), (s_xs_kf, s_Ps_kf), scale=1)
    
    # Time at the start of prediction
    filt_time = (p_xs_kf.shape[0]-1) * filt_dt # Number of filtering steps * filt_dt
    t0_pred = datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f") + timedelta(seconds = filt_time)

    # Prediction for primary and secondary satellites 7-days into future
    if cached_dir is "":
        # Propagate Primary Satellite 7 days into future + its covariance
        p_xpred = p_xs_kf[-1]
        p_Ppred = p_Ps_kf[-1]
        fermiSat = FermiSatelliteModel(t0_pred, p_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(p_xpred, 0)
        p_pks = [] # Primary Pos 3-Vec
        p_vks = [] # Primary Vel 3-Vec
        p_cdks = [] # Primary change in atms. drag
        p_aks = [] # Primary Acc 3-Vec
        p_Pks = [] # Primary Covariance 
        # Propagate Primary and Store
        xk = p_xpred.copy()
        Pk = p_Ppred.copy()
        for i in range(pred_steps):
            dxk_dt = fermiSat.get_state6_derivatives() 
            p_pks.append(xk[0:3])
            p_vks.append(xk[3:6])
            p_cdks.append(xk[6])
            p_aks.append(dxk_dt[3:6])
            p_Pks.append(Pk)
            xk = fermiSat.step()
            Phik = fermiSat.get_transition_matrix(taylor_order=3)
            Pk = Phik @ Pk @ Phik.T
        p_pks = np.array(p_pks)
        p_vks = np.array(p_vks)
        p_aks = np.array(p_aks)
        p_Pks = np.array(p_Pks)
        fermiSat.clear_model()
        cache_dic['prim_pred_hist'] = (p_pks, p_vks, p_cdks, p_aks, p_Pks)

        # Propagate Secondary Satellite 7 days into future + its covariance
        s_xpred = s_xs_kf[-1]
        s_Ppred = s_Ps_kf[-1]
        fermiSat = FermiSatelliteModel(t0_pred, s_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(s_xpred, 0)
        s_pks = [] # Secondary Pos 3-Vec
        s_vks = [] # Secondary Vel 3-Vec
        s_cdks = [] # Secondary change in atms. drag
        s_aks = [] # Secondary Acc 3-Vec
        s_Pks = [] # Secondary Covariance 
        xk = s_xpred.copy()
        Pk = s_Ppred.copy()
        for i in range(pred_steps):
            dxk_dt = fermiSat.get_state6_derivatives() 
            s_pks.append(xk[0:3])
            s_vks.append(xk[3:6])
            s_cdks.append(xk[6])
            s_aks.append(dxk_dt[3:6])
            s_Pks.append(Pk)
            xk = fermiSat.step()
            Phik = fermiSat.get_transition_matrix(taylor_order=3)
            Pk = Phik @ Pk @ Phik.T
        s_pks = np.array(s_pks)
        s_vks = np.array(s_vks)
        s_cdks = np.array(s_cdks)
        s_aks = np.array(s_aks)
        s_Pks = np.array(s_Pks)
        fermiSat.clear_model()
        cache_dic['sec_pred_hist'] = (s_pks, s_vks, s_cdks, s_aks, s_Pks)
    else:
        p_pks, p_vks, p_cdks, p_aks, p_Pks = cache_dic['prim_pred_hist']
        s_pks, s_vks, s_cdks, s_aks, s_Pks = cache_dic['sec_pred_hist']


    # Here, store the data before moving on to the second part 
    if cached_dir is "":
        input_ok = False
        while not input_ok:
            input_ok = True
            ui = input("Would you like to store your data? (Enter y or n):").lower()
            if( ui == 'y' ):
                timestamp = str( time.time() ) + ".pickle"
                fpath = data_path + timestamp
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    
    # Plot relative differences and choose a window of time where both satellite are very close to each other, 7-days out in future
    if with_pred_plots:
        # Plot Trajectories of both satellites 
        fig = plt.figure() 
        ax = fig.gca(projection='3d')
        plt.title("Predicted trajectories (primary=red, secondary=blue) over 7-day lookahead:")
        ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
        ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
        ax.scatter(p_pks[0,0], p_pks[0,1], p_pks[0,2], color = 'k', s=80)
        ax.set_xlabel("x-axis (km)")
        ax.set_ylabel("y-axis (km)")
        ax.set_zlabel("z-axis (km)")
        fig2 = plt.figure()
        plt.suptitle("Norm of predicted satellite seperation over 7-day lookahead")
        r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
        plt.plot( (np.arange(r_norms.size) * pred_dt) / (24*60*60), r_norms)
        plt.ylabel("Seperation (km)")
        plt.xlabel("# days lookahead")
        plt.show()

    # If user wishes to see prediction plot, ask if they would like to create MC over other window
    run_itca = False 
    if cache_dic['itca_data'] is None:
        cache_dic['itca_window_idxs'] = [4660, 4680] # Could manually reset this here
        run_itca = True
    if (cache_dic['itca_data'] is not None) and with_pred_plots:
        print("Old ITCA Left and Right Hand Side Window Indices: ", cache_dic['itca_window_idxs'][0], cache_dic['itca_window_idxs'][1] )
        while True:
            is_run = input("Would you like to rerun Iterative Time of Closest Approach (ITCA)? (Enter y or n): ")
            if is_run == 'y':
                run_itca = True
                print("Re-running ITCA!")
                valid_range = (0, r_norms.size-1)
                is_ok = False 
                while True:
                    cache_dic['itca_window_idxs'][0] = int( input("   Enter index for itca window start: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1]) ) )
                    if( (cache_dic['itca_window_idxs'][0] >= valid_range[0]) and (cache_dic['itca_window_idxs'][0] <= valid_range[1]) ):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(cache_dic['itca_window_idxs'][0]) )
                valid_range = (cache_dic['itca_window_idxs'][0]+1, r_norms.size-1)
                while True:
                    cache_dic['itca_window_idxs'][1] = int( input("   Enter index for itca window end: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1]) ) )
                    if( (cache_dic['itca_window_idxs'][1] >= valid_range[0]) and (cache_dic['itca_window_idxs'][1] <= valid_range[1]) ):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(cache_dic['itca_window_idxs'][1]) )
                print("Re-running ITCA with LHS/RHS indices of ", cache_dic['itca_window_idxs'])
                break
            elif is_run == 'n':
                run_itca = False
                print("Not rerunning ITCA!")
                break
            else: 
                print("Invalid entery. Try again!")
        
    # Now run the iterative time of closest approach algorithm if desired
    if run_itca:
        # Run iterative time of closest approach over this window, find exact point of closest approach
        start_idx = cache_dic['itca_window_idxs'][0]
        end_idx = cache_dic['itca_window_idxs'][1]
        # GMAT iterative closest time of approach
        i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c = iterative_time_closest_approach(
            pred_dt, t0_pred, 
            (p_pks[:end_idx],p_vks[:end_idx],p_aks[:end_idx]), 
            (s_pks[:end_idx],s_vks[:end_idx],s_aks[:end_idx]), 
            start_idx = start_idx,
            with_plot=with_pred_plots
            )
        cache_dic['itca_data'] = (i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c)

        # Step both the primary covariance and secondary covariance to the time of closest approach 
        xlhs_prim = np.concatenate(( p_pks[i_star_lhs], p_vks[i_star_lhs], np.array([p_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel( t0_pred + timedelta(seconds = t_lhs), xlhs_prim[0:6], t_c - t_lhs )
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_prim, 0)
        p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
        p_xtc = fermiSat.step()
        print("XPrim at TCA: ", p_xtc[0:3])
        print("Pos Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
        print("Vel Diff XPrim at TCA (meters/sec): ", 1000*(pv_c - p_xtc[3:6]) )
        fermiSat.clear_model()
        p_xtc[0:3] = pp_c.copy()
        p_xtc[3:6] = pv_c.copy()
        cache_dic['nom_prim_tca'] = (p_xtc, p_Ptc)

        # Propagate the Secondary Covariance to Point of Closest Approach 
        xlhs_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs], np.array([s_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel(t0_pred + timedelta(seconds = t_lhs), xlhs_sec[0:6], t_c - t_lhs)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_sec, 0)
        s_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        s_Ptc = s_Phi @ s_Pks[i_star_lhs] @ s_Phi.T
        s_xtc = fermiSat.step()
        print("XSec at TCA: ", s_xtc[0:3])
        print("Pos Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
        print("Vel Diff XSec at TCA (meters/sec): ", 1000*(sv_c - s_xtc[3:6]) )
        fermiSat.clear_model()
        s_xtc[0:3] = sp_c.copy()
        s_xtc[3:6] = sv_c.copy()
        cache_dic['nom_sec_tca'] = (s_xtc, s_Ptc)

        input_ok = False
        while not input_ok:
            input_ok = True
            ui = input("Would you like to store your data? (y/n)").lower()
            if( ui == 'y' ):
                if cached_dir is "":
                    fpath = data_path + timestamp
                else:
                    fpath = data_path + cached_dir
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    else:
        i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c = cache_dic['itca_data']
        p_xtc, p_Ptc = cache_dic['nom_prim_tca']
        s_xtc, s_Ptc = cache_dic['nom_sec_tca']

    # Repeat the following two steps for a select number of monte carlos ... caching the mc trial data as you go... this is expensive
    mc_trials = 0 #int( input("How many MC trials would you like to add: (i.e, 0 to 10000): ") )

    # Simulate a new atms. density realization under the current atms. distribution starting at primary at end of filtration. 
    print("Running MC for Primary:")
    new_mc_runs_prim = []
    for mc_it in range(mc_trials):
        print( "Primary trial {}/{}:".format(mc_it+1, mc_trials) )
        p_xpred = p_xs_kf[-1].copy()
        fermiSat = FermiSatelliteModel(t0_pred, p_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        xs, _, ws, _ = fermiSat.simulate(None, std_gps_noise, W=None, with_density_jumps = False, num_steps = i_star_lhs)
        fermiSat.clear_model()
        fermiSat = FermiSatelliteModel(t0_pred, xs[-1][0:6].copy(), t_c - t_lhs)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xs[-1].copy(), 0)
        xmc = fermiSat.step()
        new_mc_runs_prim.append(xmc.copy())
        fermiSat.clear_model()
    
    print("Running MC for Secondary:")
    new_mc_runs_sec = []
    for mc_it in range(mc_trials):
        print( "Secondary trial {}/{}:".format(mc_it+1, mc_trials) )
        s_xpred = s_xs_kf[-1].copy()
        fermiSat = FermiSatelliteModel(t0_pred, s_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        xs, _, ws, _ = fermiSat.simulate(None, std_gps_noise, W=None, with_density_jumps = False, num_steps = i_star_lhs)
        fermiSat.clear_model()
        fermiSat = FermiSatelliteModel(t0_pred, xs[-1][0:6].copy(), t_c - t_lhs)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xs[-1].copy(), 0)
        xmc = fermiSat.step()
        new_mc_runs_sec.append(xmc.copy())
        fermiSat.clear_model()
    
    if cache_dic['mc_prim_tcas'] is None:
        cache_dic['mc_prim_tcas'] = new_mc_runs_prim
    else:
        cache_dic['mc_prim_tcas'] += new_mc_runs_prim
    
    if cache_dic['mc_sec_tcas'] is None:
        cache_dic['mc_sec_tcas'] = new_mc_runs_sec
    else:
        cache_dic['mc_sec_tcas'] += new_mc_runs_sec
    
    plot_only_new_mcs = False 
    if plot_only_new_mcs:
        prim_mc = new_mc_runs_sec
        sec_mc = new_mc_runs_sec
    else:
        prim_mc = cache_dic['mc_prim_tcas']
        sec_mc = cache_dic['mc_sec_tcas']
    
    # Store data 
    if(mc_trials > 0):
        input_ok = False
        while not input_ok:
            input_ok = True
            ui = 'y' #input("Would you like to store your data? (y/n)").lower()
            if( ui == 'y' ):
                if cached_dir is "":
                    fpath = data_path + timestamp
                else:
                    fpath = data_path + cached_dir
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    
    # Plot this out in 3D
    pc.draw_3d_encounter_plane(s_xtc, p_xtc, s_Ptc[0:3,0:3], p_Ptc[0:3,0:3], mc_runs_prim = prim_mc, mc_runs_sec = sec_mc)

    # Plot this out in 2D
    pc.draw_2d_projected_encounter_plane_v2(s_xtc, p_xtc, s_Ptc, p_Ptc, cache_dic['mc_prim_tcas'], cache_dic['mc_sec_tcas'])

    # Analyze MC here
    foobar=5


if __name__ == "__main__":
    #test_time_convert()
    #test_all()

    #test_sat_pc()
    #find_cross_radial_error_mc_vs_density()
    #test_sat_crossing()
    test_sat_pc_mc()