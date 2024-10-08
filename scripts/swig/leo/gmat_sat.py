import os, sys, math 
import numpy as np 
file_dir = os.path.dirname(os.path.abspath(__file__))
gmat_root_dir = '/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/GMAT/application' # CHANGE THIS FOR YOUR SETUP!
gmat_data_dir = file_dir + "/gmat_data/gps_2_11_23/" # CHANGE THIS TO DIRECTORY WHERE EOP FILE and SPACE WEATHER FILE LIVE!
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
from cauchy_estimator import random_symmetric_alpha_stable
from cauchy_estimator import cd4_gvf

MonthDic = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
MonthDic2 = {v:k for k,v in MonthDic.items()}

def datetime_2_time_string(t):
    global MonthDic2
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

def time_string_2_datetime(t):
    return datetime.strptime(t, "%d %b %Y %H:%M:%S.%f")

# Time conversion function using GMAT
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
    time_out_mjd = timecvt.Convert(time_in_mjd, time_code_in, time_code_out)
    time_out_greg = timecvt.ConvertMjdToGregorian(time_out_mjd)
    time_dic = {"in_greg" : time_in_greg, 
                "in_mjd" : time_in_mjd, 
                "out_greg": time_out_greg, 
                "out_mjd": time_out_mjd}
    return time_dic


# Initial state given in distance units kilometers
class FermiSatelliteModel():
    def __init__(self, t0, x0, dt, gmat_print = True):
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
        self.gmat_print = gmat_print
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

    def create_model(self, with_jacchia=True, with_SRP=True, Cd0 = 2.1, Cr0 = 0.75):
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
        self.sat.SetField("Cd", Cd0)
        self.sat.SetField("Cr", Cr0)
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
        if self.gmat_print:
            self.fueltank.Help()
        self.sat.SetField("Tanks", "FuelTank") # ??? does this add the fueltank to satellite?
        if self.gmat_print:
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
        self.earthgrav.SetField("Degree", 70)
        self.earthgrav.SetField("Order", 70)
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
            self.srp.SetField("SRPModel", "Spherical")
            self.srp.SetField("Flux", 1370.052)
        # Drag Model
        if with_jacchia:
            self.jrdrag = gmat.Construct("DragForce")
            self.jrdrag.SetField("AtmosphereModel","JacchiaRoberts")
            self.jrdrag.SetField("HistoricWeatherSource", 'CSSISpaceWeatherFile')
            self.jrdrag.SetField("CSSISpaceWeatherFile", gmat_data_dir+"SpaceWeather-v1.2.txt")
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
        
        if self.gmat_print:
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
        self.pdprop.SetField("MinStep", 0)
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
    
    def reset_state_with_ellapsed_time(self, x, ellapsed_time):
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
        self.gator.SetTime(ellapsed_time)

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

    def _get_transition_matrix(self, taylor_order, use_units_km = True):
        num_sf = len(self.solve_for_states)
        num_x = 6 + num_sf
        Jac = self.get_jacobian_matrix()
        if not use_units_km:
            Jac[3:6,6] *= 1000 # convert Jac to meter-based Jacobian
        Phi = np.eye(num_x) + Jac * self.dt
        for i in range(2, taylor_order+1):
            Phi += np.linalg.matrix_power(Jac, i) * self.dt**i / math.factorial(i)
        return Phi
    
    def get_simple_transition_matrix(self, taylor_order, use_units_km = True):
        num_sf = len(self.solve_for_states)
        num_x = 6 + num_sf
        Jac = self.get_jacobian_matrix()
        if not use_units_km:
            Jac[3:6,6] *= 1000 # convert Jac to meter-based Jacobian
        Phi = np.eye(num_x) + Jac * self.dt
        for i in range(2, taylor_order+1):
            Phi += np.linalg.matrix_power(Jac, i) * self.dt**i / math.factorial(i)
        return Phi
    
    def get_transition_matrix(self, taylor_order, use_units_km = True):
        return self.get_precision_transition_matrix(taylor_order = taylor_order, use_units_km=use_units_km, dt_nom_step = 5.0)
    
    def get_ellapsed_time(self):
        return self.gator.GetTime()
    
    def get_precision_transition_matrix(self, taylor_order, use_units_km=True, dt_nom_step = 5.0):
        dt = self.dt
        x0 = self.get_state()
        ellapsed_time = self.gator.GetTime()
        sub_steps = int(dt + dt_nom_step - 1) // int(dt_nom_step)
        #sub_steps = 12
        dt_sub = dt / sub_steps
        self.dt = dt_sub
        n = 6 + len(self.solve_for_nominals)
        TAYLOR_ORDER = taylor_order
        STM_AVG_JAC = np.eye(n)
        for i in range(sub_steps):
            # Get Jacobians and STMs over time step DT_SUB
            Jac_i = self.get_jacobian_matrix()
            self.step()
            Jac_ip1 = self.get_jacobian_matrix()
            Jac_avg = (Jac_i+Jac_ip1)/2
            if not use_units_km:
                Jac_avg[3:6,6:] *= 1000
            STM_AVG_JAC = get_STM(Jac_avg, dt_sub, TAYLOR_ORDER) @ STM_AVG_JAC
        self.dt = dt
        self.reset_state_with_ellapsed_time(x0, ellapsed_time)
        return STM_AVG_JAC

    def step(self, noisy_prop_solve_for = False):
        self.gator.Step(self.dt)
        num_sf = len(self.solve_for_states)
        xk = np.zeros(6 + num_sf)
        xk[0:6] = np.array(self.gator.GetState())
        if (num_sf > 0):
            if noisy_prop_solve_for:
                xk[6:], wk = self.propagate_solve_fors(noisy_prop_solve_for)
                return xk, wk 
            else:
                xk[6:] = self.propagate_solve_fors(noisy_prop_solve_for)
                return xk
        else:
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
            return random_symmetric_alpha_stable(self.solve_for_alphas[j], self.solve_for_scales[j], 0)
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


def transform_coordinate_system(r3vec, date, mode = "ei2b", sat_handle = None):
    assert(sat_handle is not None)
    # Create Transformed GPS Msmt in Earth MJ2000Eq Coordinates
    _rvec = list([r3vec[0], r3vec[1], r3vec[2], 0, 0, 0])
    rvec = gmat.Rvector6( *_rvec )
    fixedState = gmat.Rvector6()
    time_dic_a1 = time_convert(date, "UTC", "A1")
    time_a1mjd = time_dic_a1["out_mjd"]
    if mode == "ei2b":
        sat_handle.csConverter.Convert(time_a1mjd, rvec, sat_handle.eci, fixedState, sat_handle.ecf)
        body_3vec = np.array([ fixedState[0], fixedState[1], fixedState[2] ])
        return body_3vec
    elif mode == "eb2i":
        sat_handle.csConverter.Convert(time_a1mjd, rvec, sat_handle.ecf, fixedState, sat_handle.eci)
        inertial_3vec = np.array([ fixedState[0], fixedState[1], fixedState[2] ])
        return inertial_3vec
    else:
        print("{} is not an option! Enter 'ei2b' -> earth inertial to body, or 'eb2i' -> earth body to inertial. Exiting!".format(mode))
        exit(1)

def transform_coordinate_system_jacobian_H(r3vec, date, mode = "ei2b", sat_handle = None): 
    assert(sat_handle is not None)
    H = cd4_gvf(r3vec, transform_coordinate_system, other_params=(date, mode, sat_handle))
    H = np.hstack((H,np.zeros((3,4))))
    return H

# Set convert_Jac_to_meters to True if the Jacobian comes in w.r.t km and you wanna convert it to meters before the Power Series
def _get_transition_matrix(Jac, dt, taylor_order, convert_Jac_to_meters = False):
    n = Jac.shape[0]
    if convert_Jac_to_meters:
        Jac[3:6,6] *= 1000 # convert Jac to meter-based Jacobian
    Phi = np.eye(n) + Jac * dt
    for i in range(2, taylor_order+1):
        Phi += np.linalg.matrix_power(Jac, i) * dt**i / math.factorial(i)
    return Phi

def get_STM(Jac, dt, taylor_order, convert_Jac_to_meters = False):
    return _get_transition_matrix(Jac, dt, taylor_order, convert_Jac_to_meters = convert_Jac_to_meters)

def get_along_cross_radial_rotation_matrix(x):
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

def get_along_cross_radial_state_cov(xhat, Phat):
    R = get_along_cross_radial_rotation_matrix(xhat)
    # Error w.r.t track frame
    x_track = R @ xhat
    # Error Covariance w.r.t track frame
    P_track = R @ Phat @ R.T
    return x_track, P_track, R

def get_along_cross_radial_errors_cov(xhat, Phat, xt):
    R = get_along_cross_radial_rotation_matrix(xhat)
    # Error w.r.t input coordinate frame 
    e = xt[0:3] - xhat[0:3]
    # Error w.r.t track frame
    e_track = R @ e
    # Error Covariance w.r.t track frame
    P_track = R @ Phat @ R.T
    return e_track, P_track, R
