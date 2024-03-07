import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import os, sys 
import cauchy_estimator as ce 
import math 

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
import pickle 

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

# Initial state given in distance units kilometers
class FermiSatelliteModel():
    def __init__(self, t0, x0, dt, gps_std_dev):
        self.t0 = t0
        self.x0 = x0.copy()
        self.dt = dt 
        self.gps_std_dev = gps_std_dev 
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
        self.reset_state(self, x, 0)

    def transform_Earth_MJ2000Eq_2_BodyFixed(self, time_a1mjd, state_mj2000):
        state_in = gmat.Rvector6(*list(state_mj2000[0:6])) # State in MJ2000Eq
        state_out = gmat.Rvector6() # State in Earth Body Fixed Coordinates
        self.csConverter.Convert(time_a1mjd, state_in, self.eci, state_out, self.ecf)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array
    
    def transform_Earth_BodyFixed_2_MJ2000Eq(self, time_a1mjd, state_earth_bf):
        state_in = gmat.Rvector6(*list(state_earth_bf[0:6])) # State in Earth Body Fixed Coordinates
        state_out = gmat.Rvector6() # State in MJ2000Eq
        self.csConverter.Convert(time_a1mjd, state_in, self.ecf, state_out, self.eci)
        so_array = np.array([ state_out[0], state_out[1], state_out[2], state_out[3], state_out[4], state_out[5] ])
        return so_array
    
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

    def simulate(self, num_orbits, W=None, with_density_jumps = False):
        assert self.is_model_constructed
        if with_density_jumps:
            assert "Cd" in self.solve_for_fields
        r0 = np.linalg.norm(self.x0[0:3])
        v0 = np.linalg.norm(self.x0[3:6])
        omega0 = v0/r0 # rad/sec (angular rate of orbit)
        orbital_period = 2.0*np.pi / omega0 #Period of orbit in seconds
        time_steps_per_period = (int)(orbital_period / self.dt + 0.50) # number of dt's until 1 revolution is made
        num_mission_steps = num_orbits * time_steps_per_period
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
    fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt, fermi_gps_std_dev)
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
    fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt, fermi_gps_std_dev)
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

if __name__ == "__main__":
    #test_time_convert()
    test_all()