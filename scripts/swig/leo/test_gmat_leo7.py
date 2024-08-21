import time
import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import sys, os
import cauchy_estimator as ce
import pickle 
from scipy.stats import chi2 


file_dir = os.path.dirname(os.path.abspath(__file__))
gmat_root_dir = '/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/GMAT/application'
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

def parse_gmat_data(path):
    print("Loading Data from path: {}".format(path))
    f = open(path)
    lines = f.readlines()
    print("Header is:\n {}".format(lines[0]) )
    Nlines = len(lines)
    cleaned_data = {"UTC_GREG":[], "X":[], "Y":[], "Z":[], "VX" :[], "VY":[], "VZ":[]}
    for i in range(1,Nlines):
        l = lines[i]
        sl = l.split()
        cleaned_data["UTC_GREG"].append((sl[0],sl[1],sl[2],sl[3]))
        cleaned_data["X"].append(float(sl[4]))
        cleaned_data["Y"].append(float(sl[5]))
        cleaned_data["Z"].append(float(sl[6]))
        cleaned_data["VX"].append(float(sl[7]))
        cleaned_data["VY"].append(float(sl[8]))
        cleaned_data["VZ"].append(float(sl[9]))
    cleaned_data["X"] = np.array(cleaned_data["X"])
    cleaned_data["Y"] = np.array(cleaned_data["Y"])
    cleaned_data["Z"] = np.array(cleaned_data["Z"])
    cleaned_data["VX"] = np.array(cleaned_data["VX"])
    cleaned_data["VY"] = np.array(cleaned_data["VY"])
    cleaned_data["VZ"] = np.array(cleaned_data["VZ"])
    return cleaned_data

def lookup_air_density(r_sat):
    if(r_sat == 550e3):
        return 2.384e-13
    elif(r_sat == 500e3):
        return 5.125e-13
    elif(r_sat == 450e3):
        return 1.184e-12
    elif(r_sat == 400e3):
        return 2.803e-12
    elif(r_sat == 350e3):
        return 7.014e-12
    elif(r_sat == 300e3):
        return 1.916e-11
    elif(r_sat == 250e3):
        return 6.073e-11
    elif(r_sat == 200e3):
        return 2.541e-10
    elif(r_sat == 150e3):
        return 2.076e-9
    elif(r_sat == 100e3):
        return 5.604e-7
    else:
        print("Lookup air density function does not have value for {}...please add! Exiting!\n", r_sat)
        exit(1)

def fermi_sat_prop(r_sat = 550e3):
    # Constant Parameters
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    x0 /= 1e3
    dt = 60

    # Havent Figured out how to enter this
    #SolarSystem.EphemerisSource = 'DE421'
    #Earth.EopFileName = 'blah/eop_file.txt'
    #Earth.NutationalUpdateInterval = 0

    # Dont think this is necessary
    #fogmcd = gmat.Construct("EstimatedParameter", "FogmCd") # Gives Error Message
    #fogmcd.SetField("Model", "FirstOrderGaussMarkov") # Gives Error Message
    #fogmcd.SetField("SolveFor","Cd") # Gives Error Message
    #fogmcd.SetField("SteadyStateValue", 2.1) # Gives Error Message
    #fogmcd.SetField("SteadyStateSigma", 0.21) # Gives Error Message
    #fogmcd.SetField("HalfLife", 600) # Gives Error Message
    
    sat = gmat.Construct("Spacecraft", "Fermi")
    sat.SetField("DateFormat", "UTCGregorian") #EpochFormat Form Box
    sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sat.SetField("DisplayStateType","Cartesian") #StateType Form Box
    
    sat.SetField("Epoch", "10 Jul 2023 19:31:54.000") # 19:31:54:000") 
    sat.SetField("DryMass", 3995.6)
    sat.SetField("Cd", 2.1)
    sat.SetField("CdSigma", 0.21)
    #sat.SetField("AtmosDensityScaleFactor", 1.0) # Gives Error Message
    #sat.SetField("AtmosDensityScaleFactorSigma", 0.8) # Gives Error Message
    sat.SetField("Cr", 1.8)
    sat.SetField("CrSigma", 0.1)
    sat.SetField("DragArea", 14.18)
    sat.SetField("SRPArea", 14.18)
    sat.SetField("Id", '2525')
    sat.SetField("X", x0[0])
    sat.SetField("Y", x0[1])
    sat.SetField("Z", x0[2])
    sat.SetField("VX", x0[3])
    sat.SetField("VY", x0[4])
    sat.SetField("VZ", x0[5])

    #sat.SetField("SolveFors", 'CartesianState, FogmCd, FogmAtmosDensityScaleFactor')
    fueltank = gmat.Construct("ChemicalTank", "FuelTank")
    fueltank.SetField("FuelMass", 359.9) #FuelMass = 359.9
    fueltank.Help()
    sat.SetField("Tanks", "FuelTank") # ??? does this add the fueltank to satellite?
    sat.Help()
    print(sat.GetGeneratingString(0))

    # Not sure if this is necessary
    #cordSysFermi = gmat.Construct("CoordinateSystem", "FermiVNB")
    #cordSysFermi.SetField("Origin", "Fermi")
    #cordSysFermi.SetField("Axes", "ObjectReferenced")
    #cordSysFermi.SetField("XAxis", "V") # Gives Error Message
    #cordSysFermi.SetField("YAxis", "N") # Gives Error Message
    #cordSysFermi.SetField("Primary", "Earth") # Gives Error Message
    #cordSysFermi.SetField("Secondary", "Fermi") # Gives Error Message

    # Create Force Model 
    fm = gmat.Construct("ForceModel", "TheForces")
    fm.SetField("ErrorControl", "None")
    # A 70x70 EGM96 Gravity Model
    earthgrav = gmat.Construct("GravityField")
    earthgrav.SetField("BodyName","Earth")
    earthgrav.SetField("Degree",70)
    earthgrav.SetField("Order",70)
    earthgrav.SetField("PotentialFile","EGM96.cof")
    earthgrav.SetField("TideModel", "SolidAndPole")
    # The Point Masses
    moongrav = gmat.Construct("PointMassForce")
    moongrav.SetField("BodyName","Luna")
    sungrav = gmat.Construct("PointMassForce")
    sungrav.SetField("BodyName","Sun")
    srp = gmat.Construct("SolarRadiationPressure")
    #srp.SetField("SRPModel", "Spherical")
    srp.SetField("Flux", 1370.052)
    # Drag Model
    jrdrag = gmat.Construct("DragForce")
    jrdrag.SetField("AtmosphereModel","JacchiaRoberts")
    #jrdrag.SetField("HistoricWeatherSource", 'CSSISpaceWeatherFile')
    #jrdrag.SetField("CSSISpaceWeatherFile", "SpaceWeather-v1.2.txt")
    # Build and set the atmosphere for the model
    atmos = gmat.Construct("JacchiaRoberts")
    jrdrag.SetReference(atmos)

    fm.AddForce(earthgrav)
    fm.AddForce(moongrav)
    fm.AddForce(sungrav)
    fm.AddForce(jrdrag)
    fm.AddForce(srp)
    fm.Help()
    print(fm.GetGeneratingString(0))

    # Build Integrator
    gator = gmat.Construct("RungeKutta89", "Gator")
    # Build the propagation container that connect the integrator, force model, and spacecraft together
    pdprop = gmat.Construct("Propagator","PDProp")  
    # Create and assign a numerical integrator for use in the propagation
    pdprop.SetReference(gator)
    # Set some of the fields for the integration
    pdprop.SetField("InitialStepSize", dt)
    pdprop.SetField("Accuracy", 1.0e-13)
    pdprop.SetField("MinStep", 0.0)
    pdprop.SetField("MaxStep", dt)
    pdprop.SetField("MaxStepAttempts", 50)

     # Assign the force model to the propagator
    pdprop.SetReference(fm)
    # It also needs to know the object that is propagated
    pdprop.AddPropObject(sat)
    # Setup the state vector used for the force, connecting the spacecraft
    psm = gmat.PropagationStateManager()
    psm.SetObject(sat)
    psm.SetProperty("AMatrix")
    #psm.SetProperty("STM") #increases state size, but gives undefined data for STM
    psm.BuildState()
    # Finish the object connection
    fm.SetPropStateManager(psm)
    fm.SetState(psm.GetState())
    # Perform top level initialization
    gmat.Initialize()


    # Finish force model setup:
    ##  Map the spacecraft state into the model
    fm.BuildModelFromMap()
    ##  Load the physical parameters needed for the forces
    fm.UpdateInitialData()

    # Perform the integation subsysem initialization
    pdprop.PrepareInternals()
    # Refresh the integrator reference
    gator = pdprop.GetPropagator()

    omega0 = v0/r0 # rad/sec (angular rate of orbit)
    orbital_period = 2.0*np.pi / omega0 #Period of orbit in seconds
    time_steps_per_period = (int)(orbital_period / dt + 0.50) # number of dt's until 1 revolution is made
    num_orbits = 20
    num_mission_steps = num_orbits * time_steps_per_period + 1
    times = []
    positions = []
    velocities = []
    
    for i in range(num_mission_steps):
        gatorstate = np.array(gator.GetState())
        r = gatorstate[0:3]
        v = gatorstate[3:6]
        positions.append(r)
        velocities.append(v)
        times.append(dt * i)

        # State / Derivative w.r.t time data 
        # Now access the state and get the derivative data
        #pstate = gator.GetState() #sat.GetState().GetState()
        #fm.GetDerivatives(pstate, dt=dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        #fdot = fm.GetDerivativeArray()
        #dx_dt = fdot[0:6]
        #A = np.array(fdot[6:42]).reshape(6,6)
        #Phi = np.array(fdot[42:78]).reshape(6,6)
        #vec = fm.GetDerivativesForSpacecraft(sat) # same result as above (at first step)
        #print("State Vector: ", pstate)
        #print("Derivative dx_dt:   ", dx_dt)
        #print("Jacobian A:   ", jac_A)
        #print()

        # To Dynamically Change the State of the Spacecraft (i.e, for estimation), can use following:
        #pert_state = np.random.randn(6)
        #new_state = gatorstate + pert_state
        #sat.SetState(*new_state) 
        #fm.BuildModelFromMap()
        #fm.UpdateInitialData()
        #pdprop.PrepareInternals()
        #gator = pdprop.GetPropagator()

        # Now step the integrator
        gator.Step(dt)

    positions = np.array(positions) #append(r)
    velocities = np.array(velocities)

    fig = plt.figure() #figsize=(15,11))
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(positions[:,0], positions[:,1], positions[:,2])
    plt.show()
    foobar = 2

# Process Noise Model 
def leo6_process_noise_model(dt):
    q = 1e-18 #8e-21; # Process noise ncertainty in the process position and velocity
    W = np.zeros((6,6))
    W[0:3,0:3] = q * np.eye(3) * dt**3 / 3 
    W[0:3,3:6] = q * np.eye(3) * dt**2 / 2 
    W[3:6,0:3] = q * np.eye(3) * dt**2 / 2 
    W[3:6,3:6] = q * np.eye(3) * dt 
    return W

# Initial state given in distance units kilometers
class FermiSatelliteModel():
    def __init__(self, x0, dt, gps_std_dev):
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
        # Create Fermi Model
        self.sat = gmat.Construct("Spacecraft", "Fermi")
        self.sat.SetField("DateFormat", "UTCGregorian") #EpochFormat Form Box
        self.sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
        self.sat.SetField("DisplayStateType","Cartesian") #StateType Form Box
    
        self.sat.SetField("Epoch", "10 Jul 2023 19:34:54.000") # 19:31:54:000") 
        self.sat.SetField("DryMass", 3995.6)
        self.sat.SetField("Cd", 2.1)
        #self.sat.SetField("CdSigma", 0.21)
        #sat.SetField("AtmosDensityScaleFactor", 1.0) # Gives Error Message
        #sat.SetField("AtmosDensityScaleFactorSigma", 0.8) # Gives Error Message
        self.sat.SetField("Cr", 1.8)
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
            #jrdrag.SetField("HistoricWeatherSource", 'CSSISpaceWeatherFile')
            #jrdrag.SetField("CSSISpaceWeatherFile", "SpaceWeather-v1.2.txt")
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
        self.reset_state(x, 0)

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

    # Must be less than self.dt
    def step_arbitrary_dt(self, dt):
        assert(dt < self.dt)
        self.gator.Step(dt)
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
                    if i == 800:
                        wk[6+j] = 6.5
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

def test_jacchia_roberts_influence():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    
    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 30 
    num_orbits = 3
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)

    # Test With and Without Jacchia + SRP
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xs1, zs, ws, vs = fermiSat.simulate(num_orbits, W=None)
    fermiSat.clear_model()
    fermiSat.create_model(with_jacchia=False, with_SRP=False)
    xs2, zs, _, vs = fermiSat.simulate(num_orbits, W=None)
    diffx = xs2 - xs1
    T = np.arange(xs1.shape[0]) 
    plt.plot(T, diffx[:,0])
    plt.plot(T, diffx[:,1])
    plt.plot(T, diffx[:,2])
    plt.show()
    
    fermiSat.clear_model()
    fermiSat.create_model(with_jacchia=False, with_SRP=False)
    xs1, _, _, _ = fermiSat.simulate(num_orbits, W=None)
    xs2, _, _, _ = fermiSat.simulate(num_orbits, W=W)
    diffx = xs2 - xs1
    T = np.arange(xs1.shape[0]) 
    plt.plot(T, diffx[:,0])
    plt.plot(T, diffx[:,1])
    plt.plot(T, diffx[:,2])
    plt.show()

    fermiSat.clear_model()
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xs1, _, _, _ = fermiSat.simulate(num_orbits, W=None)
    xs2, _, _, _ = fermiSat.simulate(num_orbits, W=W)
    diffx = xs2 - xs1
    T = np.arange(xs1.shape[0]) 
    plt.plot(T, diffx[:,0])
    plt.plot(T, diffx[:,1])
    plt.plot(T, diffx[:,2])
    plt.show()


    foobar=2 

def test_reset_influence():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    
    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 30 
    num_orbits = 3
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model()
    xs, zs, ws, vs = fermiSat.simulate(1, W=None)

    fermiSat.reset_state(x0, 0)
    xs2 = [x0]
    for i in range(1, xs.shape[0]):
        xk = fermiSat.step()
        fermiSat.reset_state(xk, i)
        xs2.append(xk)
    xs2 = np.array(xs2)
    diff_x = xs2 - xs 
    print(diff_x)

def test_solve_for_state_derivatives_influence():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    #x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])

    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 60 
    num_orbits = 10
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    
    for i in range(20):
        statek = np.array(fermiSat.gator.GetState())
        _ = fermiSat.fm.GetDerivatives(list(statek), dt=fermiSat.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot_k = np.array(fermiSat.fm.GetDerivativeArray()[0:6])
        fermiSat.sat.SetField("Cd", 2.105)
        #fermiSat.reset_state(statek, i)
        fermiSat.fm.GetDerivatives(list(statek), dt=fermiSat.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot_k_cd = np.array(fermiSat.fm.GetDerivativeArray()[0:6])
        partial_cd = (fdot_k_cd[3:] - fdot_k[3:]) / 0.005
        fermiSat.sat.SetField("Cd", 2.10)
        #fermiSat.reset_state(statek, i)

        fuel_weight = 0 # 359.6
        fdot_k_est = - 0.5*2.1*(14.18)/(3995.6 + fuel_weight)*fermiSat.jrdrag.GetDensity(list(statek), fermiSat.sat.GetEpoch(),1)*np.linalg.norm(1000*statek[3:])*1000*statek[3:]
        fdot_k_cd_est = - 0.5*2.105*(14.18)/(3995.6 + fuel_weight)*fermiSat.jrdrag.GetDensity(list(statek), fermiSat.sat.GetEpoch(),1)*np.linalg.norm(1000*statek[3:])*1000*statek[3:]
        partial_cd_est = (fdot_k_cd_est - fdot_k_est) / 0.005 / 1000 #(m -> km)

        fermiSat.sat.SetField("Cr", 1.805)
        #fermiSat.reset_state(statek, i)
        fermiSat.fm.GetDerivatives(list(statek), dt=fermiSat.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot_k_cr = np.array(fermiSat.fm.GetDerivativeArray()[0:6])
        partial_cr = (fdot_k_cr[3:] - fdot_k[3:]) / 0.005
        fermiSat.sat.SetField("Cr", 1.80)

        print("----- Step", i, "----------")
        print("Partial Cd GMAT:", partial_cd)
        print("Partial Cd NAT:", partial_cd_est)
        print("% Cd Differences:", 100*(partial_cd_est - partial_cd)/partial_cd)
        print("-----")
        print("Partial Cr GMAT:", partial_cr)
        print("Partial Cr NAT:", "NA")
        print("% Cr Differences:", "NA")
        fermiSat.step()

def test_gmat_ekf6():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    #x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])

    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 60 
    num_orbits = 10
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=W)
    ws = np.zeros((vs.shape[0]-1, 6))
    W *= 20
    V = np.eye(3) * std_gps_noise**2
    I6 = np.eye(6)

    P_kf = np.eye(6) * (0.001)**2
    x_kf = np.random.multivariate_normal(xs[0], P_kf)
    H = np.hstack((np.eye(3), np.zeros((3,3))))
    fermiSat.reset_state(x_kf, 0) #/1000)

    xs_kf = [x_kf.copy()]
    Ps_kf = [P_kf.copy()]
    STM_order = 3
    N = zs.shape[0]
    for i in range(1, N):
        # Time Prop
        Phi_k = fermiSat.get_transition_matrix(STM_order)
        P_kf = Phi_k @ P_kf @ Phi_k.T + W
        x_kf = fermiSat.step() #* 1000
        # Measurement Update
        K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
        zbar = H @ x_kf
        zk = zs[i]
        r = zk - zbar 
        print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
        x_kf = x_kf + K @ r 
        fermiSat.reset_state(x_kf, i) #/1000)
        P_kf = (I6 - K @ H) @ P_kf @ (I6 - K @ H).T + K @ V @ K.T 
        # Log
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
    xs_kf = np.array(xs_kf)
    Ps_kf = np.array(Ps_kf)
    # Plot KF Results
    xs *= 1000
    zs *= 1000
    vs *= 1000
    ws *= 1000
    xs_kf *= 1000
    Ps_kf *= 1000**2
    scale = 1
    ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=scale)

    fig = plt.figure() #figsize=(15,11))
    ax = fig.gca(projection='3d')
    ax.set_title("Leo Trajectory over Time")
    ax.plot(xs[:,0], xs[:,1], xs[:,2], color = 'r')
    ax.plot(xs_kf[:,0], xs_kf[:,1], xs_kf[:,2], color = 'b')
    plt.show()
    foobar = 2

def test_gmat_ekf7():
    np.random.seed(10)
    junk = np.random.randn(3000)
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    #x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])

    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5e-3 #7.5 / 1e10 # kilometers
    dt = 60 
    num_orbits = 10
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    # Set additional solve for states
    std_Cd = 0.0013
    tau_Cd = 21600
    fermiSat.set_solve_for("Cd", "sas", std_Cd, tau_Cd, alpha=1.3)
    #fermiSat.set_solve_for("Cd", "gauss", std_Cd, tau_Cd)
    #std_Cr = 0.0013
    #tau_Cr = 21600
    #fermiSat.set_solve_for("Cr", "gauss", std_Cr, tau_Cr)#, alpha=1.3)
    xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=0*W)

    n = 6 + len(fermiSat.solve_for_states)
    Wn = np.zeros((n,n))
    # Process noise for Position and Velocity
    Wn[0:6,0:6] = W.copy()
    #Wn[0:6,0:6] *= 1000 # Tunable w/ altitude
    # Process Noise for changes in Cd
    if n > 6:
        Wn[6,6] = std_Cd #(1.3898 * std_Cd)**2 #50*(1.3898 * std_Cd)**2
        #Wn[6,6] *= 1000#0 # Tunable w/ altitude
    # Process Noise for changes in Cr
    #if n > 7:
    #    Wn[7,7] = (1.3898 * std_Cr)**2
    #    #Wn[7,7] *= 100#0 # Tunable w/ altitude

    V = np.eye(3) * std_gps_noise**2
    I = np.eye(n)
    P_kf = np.eye(n) * (0.001)**2
    x_kf = np.random.multivariate_normal(xs[0], P_kf)
    H = np.hstack((np.eye(3), np.zeros((3,n-3))))
    fermiSat.reset_state(x_kf, 0)

    xs_kf = [x_kf.copy()]
    Ps_kf = [P_kf.copy()]
    STM_order = 3
    N = zs.shape[0]
    for i in range(1, N):
        # Time Prop
        Phi_k = fermiSat.get_transition_matrix(STM_order)
        P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
        x_kf = fermiSat.step() #* 1000
        # Measurement Update
        K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
        zbar = H @ x_kf
        zk = zs[i]
        r = zk - zbar 
        print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
        x_kf = x_kf + K @ r 
        # Make sure changes in Cd/Cr are within bounds
        x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)

        fermiSat.reset_state(x_kf, i) #/1000)
        P_kf = (I - K @ H) @ P_kf @ (I - K @ H).T + K @ V @ K.T 
        # Log
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
    xs_kf = np.array(xs_kf)
    Ps_kf = np.array(Ps_kf)
    # Plot KF Results
    scale_km_m = 1000
    xs[:,0:6] *= scale_km_m
    zs *= scale_km_m
    vs *= scale_km_m
    ws[:,0:6] *= scale_km_m
    xs_kf[:,0:6] *= scale_km_m
    Ps_kf[:, 0:6,0:6] *= scale_km_m**2
    #xs[:,6] = 2.1 * (scale_km_m + xs[:,6])
    #xs_kf[:,6] = 2.1 * (scale_km_m + xs_kf[:,6])
    scale = 1
    ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=scale)

    fig = plt.figure() #figsize=(15,11))
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(xs[:,0], xs[:,1], xs[:,2], color = 'r')
    ax.plot(xs_kf[:,0], xs_kf[:,1], xs_kf[:,2], color = 'b')
    plt.show()
    foobar = 2

def test_gmat_ekfs7():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    #x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])

    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 60 
    num_orbits = 50
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    # Set additional solve for states
    std_Cd = 0.0013
    tau_Cd = 21600
    fermiSat.set_solve_for("Cd", "sas", std_Cd, tau_Cd, alpha=1.3)
    std_Cr = 0.0013
    tau_Cr = 21600
    #fermiSat.set_solve_for("Cr", "sas", std_Cr, tau_Cr, alpha=1.1)
    xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=50*W, with_density_jumps=False)
    #xs = xs[:,:7]
    #ws = ws[:,:7]
    n = 6 + len(fermiSat.solve_for_states)
    V = np.eye(3) * std_gps_noise**2
    I = np.eye(n)

    scale_pv = 10000
    scale_d = 10000
    scale_pv2 = 500
    scale_d2 = 250

    Wn = np.zeros((n,n))
    # Process noise for Position and Velocity
    Wn[0:6,0:6] = W.copy()
    Wn[0:6,0:6] *= scale_pv # Tunable w/ altitude
    # Process Noise for changes in Cd
    if n > 6:
        Wn[6,6] = (1.3898 * std_Cd)**2
        Wn[6,6] *= scale_d#0 # Tunable w/ altitude
    # Process Noise for changes in Cr
    if n > 7:
        Wn[7,7] = (1.3898 * std_Cr)**2
        Wn[7,7] *= 250#0 # Tunable w/ altitude

    V = np.eye(3) * std_gps_noise**2
    I = np.eye(n)
    P_kf = np.eye(n) * (0.001)**2
    P_kf[6,6] = 2
    x_kf = np.random.multivariate_normal(xs[0], P_kf)
    x_kf[6] = 0
    H = np.hstack((np.eye(3), np.zeros((3,n-3))))
    fermiSat.reset_state(x_kf, 0)

    xs_kf = [x_kf.copy()]
    Ps_kf = [P_kf.copy()]
    Phis_kf = [] 
    Ms_kf = [P_kf.copy()]
    STM_order = 3
    N = zs.shape[0]
    for i in range(1, N):
        # Time Prop
        Phi_k = fermiSat.get_transition_matrix(STM_order)
        P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
        # For Smoother
        Phis_kf.append(Phi_k.copy())
        Ms_kf.append(P_kf.copy())

        x_kf = fermiSat.step() #* 1000
        # Measurement Update
        K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
        zbar = H @ x_kf
        zk = zs[i]
        r = zk - zbar 
        print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
        x_kf = x_kf + K @ r 
        # Make sure changes in Cd/Cr are within bounds
        x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)

        fermiSat.reset_state(x_kf, i) #/1000)
        P_kf = (I - K @ H) @ P_kf @ (I - K @ H).T + K @ V @ K.T 
        # Log
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
    xs_kf = np.array(xs_kf)
    Ps_kf = np.array(Ps_kf)
    
    # Get smoother estimates
    x_smoothed = xs_kf[-1].copy()
    P_smoothed = Ps_kf[-1].copy()
    xs_smoothed = [x_smoothed.copy()]
    Ps_smoothed = [P_smoothed.copy()]
    for i in reversed(range(N-1)):
        Phi = Phis_kf[i]
        x_kf = xs_kf[i]
        P_kf = Ps_kf[i]
        M_kf = Ms_kf[i+1]

        #C = P_kf @ Phi.T @ np.linalg.inv(M_kf)
        C = np.linalg.solve(M_kf, Phi @ P_kf).T
        fermiSat.reset_state(x_kf, i)
        x_smoothed = x_kf + C @ (x_smoothed - fermiSat.step() )
        P_smoothed = P_kf + C @ ( P_smoothed - M_kf ) @ C.T
        print("Smoothed Diff is: ", x_smoothed - x_kf)
        xs_smoothed.append(x_smoothed.copy())
        Ps_smoothed.append(P_smoothed.copy())
    xs_smoothed.reverse()
    Ps_smoothed.reverse()
    xs_smoothed = np.array(xs_smoothed)
    Ps_smoothed = np.array(Ps_smoothed)
    foo = np.zeros(xs_kf.shape[0])
    moment_info = {"x": xs_smoothed, "P": Ps_smoothed, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=1)

    
    Wn = np.zeros((n,n))
    # Process noise for Position and Velocity
    Wn[0:6,0:6] = W.copy()
    Wn[0:6,0:6] *= scale_pv2 # Tunable w/ altitude
    # Process Noise for changes in Cd
    if n > 6:
        Wn[6,6] = (1.3898 * std_Cd)**2
        Wn[6,6] *= scale_d2#0 # Tunable w/ altitude
    # Process Noise for changes in Cr
    if n > 7:
        Wn[7,7] = (1.3898 * std_Cr)**2
        Wn[7,7] *= 250#0 # Tunable w/ altitude
    
    P_kf = np.eye(n) * (0.001)**2
    P_kf[6,6] = 2
    x_kf = np.random.multivariate_normal(xs[0], P_kf)
    x_kf[6] = 0
    fermiSat.reset_state(x_kf, 0)
    xs_kf2 = [x_kf.copy()]
    Ps_kf2 = [P_kf.copy()]
    STM_order = 3
    N = zs.shape[0]
    for i in range(1, N):
        # Time Prop
        Phi_k = fermiSat.get_transition_matrix(STM_order)
        P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
        x_kf = fermiSat.step() #* 1000
        # Measurement Update
        K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
        zbar = H @ x_kf
        zk = zs[i]
        r = zk - zbar 
        print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
        x_kf = x_kf + K @ r 
        # Make sure changes in Cd/Cr are within bounds
        x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)

        fermiSat.reset_state(x_kf, i) #/1000)
        P_kf = (I - K @ H) @ P_kf @ (I - K @ H).T + K @ V @ K.T 
        # Log
        xs_kf2.append(x_kf.copy())
        Ps_kf2.append(P_kf.copy())
    xs_kf2 = np.array(xs_kf2)
    Ps_kf2 = np.array(Ps_kf2)

    # Plot KF Results
    scale_km_m = 1000
    zs *= scale_km_m
    vs *= scale_km_m
    ws[:,0:6] *= scale_km_m
    xs[:,0:6] *= scale_km_m
    xs_kf[:,0:6] *= scale_km_m
    Ps_kf[:, 0:6,0:6] *= scale_km_m**2
    xs_kf2[:,0:6] *= scale_km_m
    Ps_kf2[:, 0:6,0:6] *= scale_km_m**2

    scale = 1
    plt.figure()
    plt.title("Robust vs Normal EKF Position Differences (Top)\nNorm Diff (Bottom):")
    Ts = np.arange(xs_kf.shape[0])
    plt.subplot(211)
    plt.plot(Ts, xs_kf[:,0] - xs_kf2[:,0])
    plt.plot(Ts, xs_kf[:,1] - xs_kf2[:,1])
    plt.plot(Ts, xs_kf[:,2] - xs_kf2[:,2])
    plt.subplot(212)
    plt.plot(Ts, np.linalg.norm(xs_kf[:,0:3] - xs_kf2[:,0:3],axis=1) )

    foo = np.zeros(xs_kf2.shape[0])
    moment_info = {"x": xs_kf2, "P": Ps_kf2, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=scale)

    foobar = 2

### Testing Cauchy ###
global_leo = None
INITIAL_H = False

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Set Phi and Gamma
    x = pyduc.cget_x()
    Jac = global_leo.get_jacobian_matrix()
    Jac[3:6,6] *= 1000 # km to m
    taylor_order = 3
    Phi_k = np.eye(7) + Jac * global_leo.dt
    for i in range(2,taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * global_leo.dt**i / math.factorial(i)
    Gamma_k = np.zeros((7,1))
    Gamma_c = np.zeros((7,1)) # continous time Gamma 
    Gamma_c[6,0] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * global_leo.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c

    pyduc.cset_Phi(Phi_k)
    pyduc.cset_Gamma(Gamma_k)
    # Propagate and set x
    xbar = global_leo.step() 
    xbar[0:6] *= 1000
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if(INITIAL_H):
        zbar = np.array([0, 0, xbar[0] + xbar[1] + xbar[2]])
    else:
        zbar = xbar[0:3]
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    #xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if INITIAL_H:
        H = np.zeros((3,7))
        H[2,0] = 1
        H[2,1] = 1
        H[2,2] = 1
        global global_leo 
        gam = global_leo.gps_std_dev * 1000 * (1.0/1.3898)
        gamma = np.array([gam, gam, 3*gam])
        pyduc.cset_gamma(gamma)
    else:
        H = np.hstack(( np.eye(3), np.zeros((3,4)) ))
    pyduc.cset_H(H)

# choose window with last estimate's covariance defined
def best_window_est(cauchyEsts, window_counts):
    W = len(cauchyEsts)
    okays = np.zeros(W, dtype=np.bool8)
    idxs = []
    for i in range(W):
        if(window_counts[i] > 0):
            err = cauchyEsts[i]._err_code
            if (err[2] & (1<<1)) or (err[2] & (1<<3)):
                pass
            else:
                idxs.append((i, window_counts[i]))
                okays[i] = True
    if(len(idxs) == 0):
        print("No window is available without an error code!")
        exit(1)
    sorted_idxs = list(reversed(sorted(idxs, key = lambda x : x[1])))
    return sorted_idxs[0][0], okays

def weighted_average_win_est(win_moms, win_counts, usable_wins):
        num_windows = len(win_moms)
        win_avg_mean = np.zeros(7)
        win_avg_cov = np.zeros((7,7))
        win_norm_fac = 0.0
        for i in range(num_windows):
            win_count = win_counts[i]
            if win_counts[i] > 1:
                win_okay = usable_wins[i]
                if win_okay:
                    norm_fac = win_count / num_windows
                    win_norm_fac += norm_fac
                    win_avg_mean += win_moms[i][-1][0] * norm_fac
                    win_avg_cov += win_moms[i][-1][1] * norm_fac
        win_avg_mean /= win_norm_fac
        win_avg_cov /= win_norm_fac
        return win_avg_mean, win_avg_cov

def edit_means(cauchyEsts, window_counts, state_idx, low, high):
    W = len(cauchyEsts)
    for i in range(W):
        if window_counts[i] > 1:
            xhat, _ = cauchyEsts[i].get_last_mean_cov()
            if (xhat[state_idx] < low) or (xhat[state_idx] > high):
                xhat[state_idx] = np.clip(xhat[state_idx], low, high)
                pyduc = cauchyEsts[i].get_pyduc()
                pyduc.cset_x(xhat)
                print("Window", i+1, "underwent mean editing!")

def plot_all_windows(win_moms, xs_true, e_hats_kf, one_sigs_kf, best_idx, idx_min):
    W = len(win_moms)
    Ts_kf = np.arange(e_hats_kf.shape[0])
    for win_idx in range(W):
        if len(win_moms[win_idx]) > 1: #k > win_idx:
            x_hats = np.array([ win_moms[win_idx][i][0] for i in range(len(win_moms[win_idx])) ])
            P_hats = np.array([ win_moms[win_idx][i][1] for i in range(len(win_moms[win_idx])) ])
            T_cur = win_idx + x_hats.shape[0] + 1
            one_sigs = np.array([np.sqrt(np.diag(P_hat)) for P_hat in P_hats])
            e_hats = np.array([xt - xh for xt,xh in zip(xs_true[win_idx+1:T_cur], x_hats)])
            
            plt.figure()
            plt.subplot(711)
            plt.title("Win Err" + str(win_idx) + " PosX/PosY/VelX/VelY")
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,0], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,0], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,0], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 0], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 0], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 0], 'm')
            # Plot a black star indicating this window reinitialized the other
            if win_idx == best_idx:
                plt.scatter(Ts_kf[T_cur-1], one_sigs[-1,0], color='k', marker='*')
                plt.scatter(Ts_kf[T_cur-1], -one_sigs[-1,0], color='k', marker='*')
            if win_idx == idx_min:
                plt.scatter(Ts_kf[T_cur-1], one_sigs[-1,0], color='k', marker='o')
                plt.scatter(Ts_kf[T_cur-1], -one_sigs[-1,0], color='k', marker='o')
            plt.subplot(712)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,1], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,1], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,1], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 1], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 1], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 1], 'm')
            plt.subplot(713)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,2], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,2], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,2], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 2], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 2], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 2], 'm')
            plt.subplot(714)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,3], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,3], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,3], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 3], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 3], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 3], 'm')
            plt.subplot(715)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,4], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,4], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,4], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 4], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 4], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 4], 'm')
            plt.subplot(716)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,5], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,5], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,5], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 5], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 5], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 5], 'm')
            plt.subplot(717)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,6], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,6], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,6], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 6], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 6], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 6], 'm')
    plt.show()
    plt.close('all')

def _legacy_get_cross_along_radial_trans(xhat):
    # position and velocity 3-vector components
    # I think this one is slighlty worng
    rh = xhat[0:3]
    rhn = np.linalg.norm(rh)
    vh = xhat[3:6]
    vhn = np.linalg.norm(xhat[3:6])
    # Unit directions 
    uv = vh / vhn # x-axis - along track
    ur = rh / rhn # z-axis - radial
    uc = np.cross(ur, uv) # y-axis - cross track direction is radial direction cross_prod along track direction
    uc /= np.linalg.norm(uc)
    R = np.vstack( (uv,uc,ur) )

def get_cross_along_radial_errors_cov(xhat, Phat, xt):
    # position and velocity 3-vector components
    rh = xhat[0:3]
    vh = xhat[3:6]
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

    # Error w.r.t input coordinate frame 
    e = xt[0:3] - xhat[0:3]
    # Error w.r.t track frame
    e_track = R @ e
    # Error Covariance w.r.t track frame
    P_track = R @ Phat @ R.T 
    return e_track, P_track, R

def test_gmat_ece7():
    seed = int(np.random.rand() * (2**32 -1))
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Log or Load Setting
    LOAD_RESULTS_AND_EXIT = False
    WITH_LOG = False
    assert(not (LOAD_RESULTS_AND_EXIT and WITH_LOG))

    # Cauchy and Kalman Tunables
    WITH_PLOT_ALL_WINDOWS = True
    WITH_SAS_DENSITY = True
    WITH_ADDED_DENSITY_JUMPS = True
    WITH_PLOT_MARG_DENSITY = False
    reinit_methods = ["speyer", "init_cond", "H2", "H2Boost", "H2Boost2", "H2_KF", "diag_boosted"]
    reinit_method = reinit_methods[6]
    r_sat = 550e3 #km
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 60 
    num_orbits = 1
    num_windows = 3 # Number of Cauchy Windows
    ekf_scale = 10000 # Scaling factor for EKF atmospheric density
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)
    time_tag = False

    alt_and_std = str(int(r_sat/1000)) + "km" + "_A" + str(int(10*14)) + "_std" + str(int(std_gps_noise * 1e4)) + "_dt" + str(int(dt))
    ekf_scaled = "_ekfs" + str(ekf_scale)
    beta_scaled = "_bs" + str(beta_scale)
    gamma_scaled = "_gs" + str(gamma_scale)
    density_type = "_sas" if WITH_SAS_DENSITY else "_gauss"
    added_jumps = "_wj" if WITH_ADDED_DENSITY_JUMPS else "_nj"
    #time_id = str(time.time()) if time_tag else "" ### ADD SEEDING LOAD/LOG LOGIC!!

    # Log Files
    if WITH_LOG:
        log_dir = file_dir + "/pylog/gmat7/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += alt_and_std + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += reinit_method + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        with open(log_dir + "seed.txt", "w") as handle:
            handle.write( "Seeded with: " + str(seed) )
    # Load Files
    if LOAD_RESULTS_AND_EXIT:
        log_dir = file_dir + "/pylog/gmat7/"
        log_dir += alt_and_std + "/"
        log_dir += reinit_method + "/"
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
    
    # Possibly only plot logged simulation results and exit
    if LOAD_RESULTS_AND_EXIT:
        scale = 1
        ce_moments = ce.load_cauchy_log_folder(log_dir, False)
        xs_kf, Ps_kf = ce.load_kalman_log_folder(log_dir)
        xs_kf[:,0:6] *= 1000
        Ps_kf[:,0:6,0:6] *= 1000**2
        xs, zs, ws, vs = ce.load_sim_truth_log_folder(log_dir)
        xs[:,0:6] *= 1000
        zs *= 1000
        vs *= 1000
        ws[:, 0:6] *= 1000
        weighted_ce_hist_path = log_dir + "weighted_ce.pickle"
        found_pickle = False
        if os.path.isfile(weighted_ce_hist_path):
            with open(weighted_ce_hist_path, "rb") as handle:
                found_pickle = True
                avg_ce_xhats, avg_ce_Phats, win_moms = pickle.load(handle)
                foo = np.zeros(avg_ce_xhats.shape[0])
                avgd_moment_info = {"x": avg_ce_xhats, "P": avg_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
                print("All Window Cauchy Estimator History:")
                one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)) for P_kf in Ps_kf ])
                e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ])
                plot_all_windows(win_moms, xs, e_hats_kf, one_sigs_kf, 0, 1)
        print("Full Window History:")
        ce.plot_simulation_history(ce_moments, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        if found_pickle:
            print("Weighted Cauchy Estimator History:")
            ce.plot_simulation_history(avgd_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        foobar=2
        exit(1)

    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    #x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])

    # Convert to kilometers
    x0 /= 1e3 # kilometers
    # Process Noise Model
    W = leo6_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    # Set additional solve for states
    std_Cd = 0.0013
    tau_Cd = 21600
    dist_type_Cd = "sas" if WITH_SAS_DENSITY else "gauss" 
    fermiSat.set_solve_for("Cd", dist_type_Cd, std_Cd, tau_Cd, alpha=1.3)
    std_Cr = 0.0013
    tau_Cr = 21600
    #fermiSat.set_solve_for("Cr", "sas", std_Cr, tau_Cr, alpha=1.3)
    xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=W, with_density_jumps=WITH_ADDED_DENSITY_JUMPS)
    if WITH_LOG:
        ce.log_sim_truth(log_dir, xs, zs, ws, vs)

    n = 6 + len(fermiSat.solve_for_states)
    Wn = np.zeros((n,n))
    # Process noise for Position and Velocity
    Wn[0:6,0:6] = W.copy()
    Wn[0:6,0:6] *= 1000 # Tunable w/ altitude
    # Process Noise for changes in Cd
    if n > 6:
        Wn[6,6] = (1.3898 * std_Cd)**2
        Wn[6,6] *= ekf_scale # Tunable w/ altitude
    # Process Noise for changes in Cr
    if n > 7:
        Wn[7,7] = (1.3898 * std_Cr)**2
        #Wn[7,7] *= 100#0 # Tunable w/ altitude

    V = np.eye(3) * std_gps_noise**2
    I = np.eye(n)
    P_kf = np.eye(n) * (0.001)**2
    x_kf = np.random.multivariate_normal(xs[0], P_kf)
    H = np.hstack((np.eye(3), np.zeros((3,n-3))))
    fermiSat.reset_state(x_kf, 0)

    # Run EKF
    xs_kf = [x_kf.copy()]
    Ps_kf = [P_kf.copy()]
    STM_order = 3
    N = zs.shape[0]
    for i in range(1, N):
        # Time Prop
        Phi_k = fermiSat.get_transition_matrix(STM_order)
        P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
        x_kf = fermiSat.step() #* 1000
        # Measurement Update
        K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
        zbar = H @ x_kf
        zk = zs[i]
        r = zk - zbar 
        print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
        x_kf = x_kf + K @ r 
        # Make sure changes in Cd/Cr are within bounds
        x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
        # Reset Satellite about new estimate
        fermiSat.reset_state(x_kf, i) #/1000)
        P_kf = (I - K @ H) @ P_kf @ (I - K @ H).T + K @ V @ K.T 
        # Log
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
    xs_kf = np.array(xs_kf)
    Ps_kf = np.array(Ps_kf)
    if WITH_LOG:
        ce.log_kalman(log_dir, xs_kf, Ps_kf)
    # Compute 1-sigma bounds for KF for Window Plot Compares  
    one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)* 1000**2) for P_kf in Ps_kf ])
    one_sigs_kf[:,6] /= 1000
    e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ]) * 1000
    WITH_PLOT_KF_SEPERATELY = False
    if WITH_PLOT_KF_SEPERATELY:
        xs[:,0:6] *= 1000
        xs_kf[:,0:6] *= 1000
        Ps_kf[:, 0:6,0:6] *= 1000**2
        ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf) )
        exit(1)

    # Run ECE
    fermiSat.reset_state(xs[0], 0)
    global global_leo 
    global_leo = fermiSat

    beta = np.array([0.0013])
    gamma = np.array([std_gps_noise, std_gps_noise, std_gps_noise]) / 1.3898 * 1000
    beta /= beta_scale
    gamma *= gamma_scale

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Jac = fermiSat.get_jacobian_matrix()
    Jac[3:6,6] *= 1000 # km to m
    taylor_order = 3
    Phi = np.eye(7) + Jac * global_leo.dt
    for i in range(2,taylor_order+1):
        Phi += np.linalg.matrix_power(Jac, i) * global_leo.dt**i / math.factorial(i)

    xbar = xs[1].copy()
    xbar[0:6] *= 1000
    A0 = Phi.T.copy()
    p0 = np.repeat(.01, 7) #/ 1e3
    b0 = np.zeros(7)
    num_controls = 0
    zs_without_z0 = zs[1:]

    ce.set_tr_search_idxs_ordering([5,4,6,3,2,1,0])
    debug_print = True
    fermiSat.step()
    win_idxs = np.arange(num_windows)
    win_counts = np.zeros(num_windows, dtype=np.int64)
    cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, debug_print) for _ in range(num_windows)]#ce.PySlidingWindowManager("nonlin", num_windows, total_steps, log_dir=log_dir, log_seq=True, log_full=True)
    for i in range(num_windows):
        cauchyEsts[i].initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
        cauchyEsts[i].set_window_number(i)
    win_moms = { i : [] for i in range(num_windows) }
    win_moms[0].append( cauchyEsts[0].step(1000*zs_without_z0[0], None, False) )
    win_counts[0] = 1

    ce_xhats = [win_moms[0][-1][0].copy()]
    ce_Phats = [win_moms[0][-1][1].copy()]

    avg_ce_xhats = [win_moms[0][-1][0].copy()]
    avg_ce_Phats = [win_moms[0][-1][1].copy()]
    
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    #win_marginals = { i : [] for i in range(num_windows) }
    N = zs_without_z0.shape[0]
    for k in range(1, N):
        print("---- Step {}/{} -----".format(k+2, N+1))
        zk = zs_without_z0[k] 
        # find max and min indices
        idx_max = np.argmax(win_counts)
        idx_min = np.argmin(win_counts)
        # Step all windows that are not uninitialized
        for win_idx, win_count in zip(win_idxs, win_counts):
            if(win_count > 0):
                print("  Window {} is on step {}/{}".format(win_idx+1, win_count+1, num_windows) )
                x_reset = win_moms[win_idx][-1][0].copy()
                x_reset[0:6] /= 1000
                x_reset[6] = np.clip(x_reset[6], -.85, 10)
                fermiSat.reset_state(x_reset, k)
                win_moms[win_idx].append( cauchyEsts[win_idx].step(1000*zk, None, False) )
                print("    x_k|k:   ", win_moms[win_idx][-1][0] )
                x_true = xs[k+1].copy()
                x_true[0:6] *= 1000
                print("    e_k|k:   ", x_true - win_moms[win_idx][-1][0] )
                win_counts[win_idx] += 1
        
        best_idx, usable_wins = best_window_est(cauchyEsts, win_counts)
        xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
        ce_xhats.append(xhat.copy())
        ce_Phats.append(Phat.copy())
        print("Best Window Index For Reinit is: Window ", best_idx+1)
        
        # Mean edit for safety of values
        #if leo5_alt > 300e3:
        #edit_means(cauchyEsts, win_counts, 4, -.05, 0.05)
        #else:
        edit_means(cauchyEsts, win_counts, 6, -.85, 10)
        xhat[6] = np.clip(xhat[6], -.85, 10)
        

        # Compute Weighted Average Window Estimate
        avg_xhat, avg_Phat = weighted_average_win_est(win_moms, win_counts, usable_wins)
        avg_ce_xhats.append(avg_xhat)
        avg_ce_Phats.append(avg_Phat)

        # Reinitialize empty estimator
        if(reinit_method == "speyer"):
            # using speyer's start method
            speyer_restart_idx = 1
            xreset, Preset = cauchyEsts[idx_min].reset_about_estimator(cauchyEsts[best_idx], msmt_idx = speyer_restart_idx)
            print("  Window {} is on step {}/{} and has mean:\n  {}".format(idx_min+1, win_counts[idx_min]+1, num_windows, np.around(xreset,4)) )
        elif(reinit_method == "init_cond"):
            _A0 = cauchyEsts[best_idx]._Phi.copy().reshape((7,7)).T # np.eye(5)
            _p0 = np.repeat(0.01, 7)
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(1000*zk[2], _A0, _p0, b0, xhat) )
        elif(reinit_method == "diag_boosted"):
            _xbar = cauchyEsts[best_idx]._xbar[14:]
            _dz = zk[2] - _xbar[2]
            _dx = xhat - _xbar
            _P = Phat + np.diag(np.ones(7)*1.0)
            _H = np.zeros((3,7))
            _H[0:3,0:3] = np.eye(3)
            _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H[2,:], gamma[2], _dz)
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(zk[2], _A0, _p0, b0, _xbar) )
        elif("H2" in reinit_method):
            # Both H channels concatenated
            _H = np.array([1.0, 1.0, 1.0, 0, 0, 0, 0])
            _gamma = 3 * gamma[0]
            _xbar = cauchyEsts[best_idx]._xbar[14:]
            _dz = 1000*(zk[0] + zk[1] + zk[2]) - _xbar[0] - _xbar[1] - _xbar[2]
            _dx = xhat - _xbar
            
            # Covariance Selection
            if("KF" in reinit_method):
                _P = Ps_kf[k+1].copy() # KF COVAR DOUBLES LOOKS GOOD
                assert(0)
            else:
                _P = Phat.copy() # CAUCHY COVAR LOOKS GOOD
            if("Boost" in reinit_method):
                # Boost
                _pos_scale = np.ones(6)
                _P_kf = Ps_kf[k+1].copy() * 1000**2
                _P_cauchy = _P
                for i in range(6):
                    if( (_P_kf[i,i] / _P_cauchy[i,i]) > 1):
                        _pos_scale[i] = (_P_kf[i,i] / _P_cauchy[i,i]) * 1.3898
                        _P[i,i] *= _pos_scale[i]
                if "Boost2" in reinit_method:
                    _P *= 2
                    _P += np.eye(7) * 0.000001
                    assert( np.all( np.linalg.eig(_P)[0] > 0 ) )
            # Reset
            _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H, _gamma, _dz)
            global INITIAL_H
            INITIAL_H = True
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(1000*(zk[0] + zk[1] + zk[2]), _A0, _p0, _b0, _xbar) )
            pyduc = cauchyEsts[idx_min].get_pyduc()
            pyduc.cset_gamma(gamma)
            INITIAL_H = False
            foobar=2
        else:
            print("Reinitialization Scheme ", reinit_method, "Not Implemented! Please Add! Exiting!")
            exit(1)
        # Increment Initialized Estimator Count
        win_counts[idx_min] += 1

        # Now plot all windows 
        if WITH_PLOT_ALL_WINDOWS and (k==(N-1)): #and (k > 1): # 
            _xs = xs.copy()
            _xs[:,0:6] *= 1000
            plot_all_windows(win_moms, _xs, e_hats_kf, one_sigs_kf, best_idx, idx_min)

        # Density Marginals
        if WITH_PLOT_MARG_DENSITY:
            if k > 18:
                plt.figure(figsize = (8,12))
                print("  Window Counts at Step {} are:\n  {}\n  Marginal 1D CPDFs of Atmospheric Density are:".format(k+2, win_counts) )
                print("---------------------------------------------------")
                #y_avg = np.zeros(10001)
                #weight_avg = 0.0
                x_true = xs[k+1]
                top = min(k+1, num_windows)
                for win_idx in range(top):
                    win_xhat, _ = cauchyEsts[win_idx].get_last_mean_cov()
                    wgl = -1 - win_xhat[4]
                    wgh = 9 - win_xhat[4]
                    wx, wy = cauchyEsts[win_idx].get_marginal_1D_pointwise_cpdf(4, wgl, wgh, 0.001)
                    plt.subplot(top, 1, win_idx+1)
                    plt.plot(win_xhat[4] + wx, wy, 'b')
                    plt.scatter(x_true[4], 0, color='r', marker = 'x')
                    plt.scatter(win_xhat[4], 0, color='b', marker = 'x')
                    plt.ylabel("Win"+str(win_idx+1))
                    if win_idx == 0:
                        plt.title("Densities at Step {}/{}".format(k+2,N+1))
                    #weight_avg += (win_counts[win_idx] / num_windows)
                    #y_avg += wy[:10001] * (win_counts[win_idx] / num_windows)
                plt.xlabel("Change in Atms. Density State")
                print("---------------------------------------------------")
                #y_avg /= weight_avg
                #weights = y_avg / np.sum(y_avg)
                #plt.figure()
                #plt.plot(win_xhat[4] + wx, y_avg)
                #plt.scatter(x_true[4], 0, color='r', marker = 'x')
                #plt.scatter(np.sum(weights * (win_xhat[4] + wx) ), 0, color='b', marker = 'x')
                plt.show()
                foobar = 2

        # reset full estimator
        if(win_counts[idx_max] == num_windows):
            cauchyEsts[idx_max].reset()
            win_counts[idx_max] = 0
            
    ce_xhats = np.array(ce_xhats)
    ce_Phats = np.array(ce_Phats)
    avg_ce_xhats = np.array(avg_ce_xhats)
    avg_ce_Phats = np.array(avg_ce_Phats)
    foo = np.zeros(ce_xhats.shape[0])
    moment_info = {"x": ce_xhats, "P": ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    avg_moment_info = {"x": avg_ce_xhats, "P": avg_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    if WITH_LOG:
        ce.log_cauchy(log_dir, moment_info)
        with open(log_dir + "weighted_ce.pickle", "wb") as handle:
            pickle.dump((avg_ce_xhats, avg_ce_Phats, win_moms), handle)

    # Plot KF Results
    scale_km_m = 1000
    xs[:,0:6] *= scale_km_m
    zs *= scale_km_m
    vs *= scale_km_m
    ws[:,0:6] *= scale_km_m
    # KF
    xs_kf[:,0:6] *= scale_km_m
    Ps_kf[:, 0:6,0:6] *= scale_km_m**2

    print("Full Window History:")
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    print("Weighted Cauchy Estimator History:")
    ce.plot_simulation_history(avg_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2


    # Plot KF Results
    '''
    scale_km_m = 1000
    xs[:,0:6] *= scale_km_m
    zs *= scale_km_m
    vs *= scale_km_m
    ws[:,0:6] *= scale_km_m
    xs_kf[:,0:6] *= scale_km_m
    Ps_kf[:, 0:6,0:6] *= scale_km_m**2
    #xs[:,6] = 2.1 * (scale_km_m + xs[:,6])
    #xs_kf[:,6] = 2.1 * (scale_km_m + xs_kf[:,6])
    scale = 1
    ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=scale)

    fig = plt.figure() #figsize=(15,11))
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(xs[:,0], xs[:,1], xs[:,2], color = 'r')
    ax.plot(xs_kf[:,0], xs_kf[:,1], xs_kf[:,2], color = 'b')
    plt.show()
    foobar = 2
    '''

def test_point_in_ellipse():
    from scipy.stats import chi2
    conf_int = 0.70
    N_samples = 200
    # find whether a point lies within the 1,2,3,4,5,6-sigma covariance ellipse
    P = np.array([2,1,1,2]).reshape((2,2))
    # The chi squared ppf which corresponds to some quantile (conf_int)
    s = chi2.ppf(conf_int, P.shape[0])
    # Ellipse is the matrix square root of covariance
    D, U = np.linalg.eig(P)
    E = U @ np.diag(D * s)**0.5 
    # Generate points on ellipse x = E @ unit
    unit = np.array([[np.cos(t), np.sin(t)] for t in np.arange(0,2*np.pi, 0.05)])
    points = unit @ E.T # batch these
    # Sample Gaussian Points
    sampled = np.random.multivariate_normal(np.zeros(2), P, size=(N_samples,))
    # Inverse Cov
    Pinv = np.linalg.inv(P)
    # Points inside ellipse have form x^T Pinv @ x <= s
    inside_ell = np.zeros(N_samples, dtype=np.int64)
    for i in range(N_samples):
        inside_ell[i] = sampled[i] @ Pinv @ sampled[i] < s 
    print("Percent inside ellipse: ", np.sum(inside_ell) / N_samples)
    plt.scatter(sampled[:,0], sampled[:,1], color='b')
    plt.plot(points[:,0], points[:,1], 'r')
    plt.show()
    foobar = 2

def test_point_in_ellipse2():
    from scipy.stats import chi2
    conf_int = 0.95
    N_samples = 2000
    # find whether a point lies within the 1,2,3,4,5,6-sigma covariance ellipse
    #P = np.array([[2, -1.0, 0.5],
    #              [-1, 3.0, -0.6],
    #              [.5,-0.6, 2.0]])#.reshape((3,3))
    P = np.array([[ 0.00385068, -0.00154697,  0.00260381],
                  [-0.00154697,  0.00062195, -0.0010462 ],
                  [ 0.00260381, -0.0010462 ,  0.0017614 ]])
    # The chi squared ppf which corresponds to some quantile (conf_int)
    s = chi2.ppf(conf_int, P.shape[0])
    # Ellipse is the matrix square root of covariance
    D, U = np.linalg.eig(P)
    E = U @ np.diag(D * s)**0.5 
    # Generate points on ellipse x = E @ unit
    unit = np.array([[np.cos(t), np.sin(t), 0] for t in np.arange(0,2*np.pi, 0.05)])
    points = unit @ E.T # batch these
    # Sample Gaussian Points
    sampled = np.random.multivariate_normal(np.zeros(3), P, size=(N_samples,))
    # Inverse Cov
    Pinv = np.linalg.inv(P)
    # Points inside ellipse have form x^T Pinv @ x <= s
    inside_ell = np.zeros(N_samples, dtype=np.int64)
    for i in range(N_samples):
        inside_ell[i] = sampled[i] @ Pinv @ sampled[i] < s 
    print("Percent inside ellipse: ", np.sum(inside_ell) / N_samples)
    #plt.scatter(sampled[:,0], sampled[:,1], color='b')
    #plt.plot(points[:,0], points[:,1], 'r')
    #plt.show()
    foobar = 2

def test_mc_trial_logger():
    r_sat = 550e3 #km
    r_earth = 6378.1e3
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    #rho = lookup_air_density(r_sat)
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    #x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), 0.0, v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
    #x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])
    x0 = np.array([4.99624529e+03,  3.87794646e+03,  2.73604324e+03, -5.02809357e+00, 5.57592134e+00,  1.26986117e+00])*1000
    # Convert to kilometers
    x0 /= 1e3 # kilometers
    std_gps_noise = 7.5 / 1e3 # kilometers
    dt = 60 
    num_orbits = 20 #105
    # Set additional solve for states
    std_Cd = 0.0013
    tau_Cd = 21600
    sas_Cd = "sas" #"gauss" # sas
    # With smoother 
    with_smoother = False
    # With Plotting 
    with_plots = True
    # Get Chi-Squared Stats for three states 
    n_pred_states = 3
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    qs = np.array([chi2.ppf(q, n_pred_states) for q in quantiles]) # quantile scores

    # STM Taylor Order Approx
    STM_order = 3
    # Process Noise Model
    W6 = leo6_process_noise_model(dt)
    Wn = np.zeros((7,7))
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

    # Monte Carlo Settings
    mc_trials = 2
    mc_dic = { "mode" : sas_Cd, 
               "trials": mc_trials, 
               "x0" : x0.copy(),
               "dt" : dt,
               "std_gps_noise" : std_gps_noise,
               "sas_Cd" : sas_Cd,
               "std_Cd" : std_Cd,
               "tau_Cd" : tau_Cd,
               "sas_alpha" : sas_alpha,
               "SimProcNoise6" : W6.copy(),
               "STM_order" : STM_order,
               "EKFProcNoiseMat" : Wn.copy(),
               "scale_pv" : scale_pv,
               "scale_d" : scale_d,
               "sim_truth" : [], 
               "ekf_runs" : [], 
               "ekf_smoothed" : [],
               }
    
    # Create Satellite Model 
    x07 = np.zeros(7)
    x07[0:6] = x0.copy()
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)

    # Run trials
    n = 6 + len(fermiSat.solve_for_states)
    V = np.eye(3) * std_gps_noise**2
    I = np.eye(n)
    H = np.hstack((np.eye(3), np.zeros((3,n-3))))
    for trial in range(mc_trials):
        print("Trial {}/{}".format(trial+1, mc_trials))
        xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=None, with_density_jumps=True)
        mc_dic["sim_truth"].append((xs,zs,ws,vs))

        # Setup Kalman Filter
        P_kf = np.eye(n) * (0.001)**2
        P_kf[6,6] = .01
        x_kf = np.random.multivariate_normal(xs[0], P_kf)
        x_kf[6] = 0
        fermiSat.reset_state(x_kf, 0)
        xs_kf = [x_kf.copy()]
        Ps_kf = [P_kf.copy()]
        Phis_kf = [] 
        Ms_kf = [P_kf.copy()]
        N = zs.shape[0]
        for i in range(1, N):
            # Time Prop
            Phi_k = fermiSat.get_transition_matrix(STM_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
            # For Smoother
            Phis_kf.append(Phi_k.copy())
            Ms_kf.append(P_kf.copy())

            x_kf = fermiSat.step() #* 1000
            # Measurement Update
            K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar = H @ x_kf
            zk = zs[i]
            r = zk - zbar 
            print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(xs[i] - x_kf))
            x_kf = x_kf + K @ r 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)

            fermiSat.reset_state(x_kf, i) #/1000)
            P_kf = (I - K @ H) @ P_kf @ (I - K @ H).T + K @ V @ K.T 
            # Log
            xs_kf.append(x_kf.copy())
            Ps_kf.append(P_kf.copy())
        xs_kf = np.array(xs_kf)
        Ps_kf = np.array(Ps_kf)
        mc_dic["ekf_runs"].append((xs_kf, Ps_kf))
        
        if with_plots:
            trial_filter_hits = np.zeros(qs.size)
            for i in range(N):
                x_kf = xs_kf[i]
                P_kf = Ps_kf[i]
                e_kf = (xs[i] - x_kf)[0:n_pred_states]
                Pinv = np.linalg.inv(P_kf[0:n_pred_states, 0:n_pred_states])
                score = e_kf.T @ Pinv @ e_kf
                hits = score < qs
                #print("Score: {}, qs:{}".format(score, qs))
                trial_filter_hits += hits
            trial_filter_hits /= N
            print("Checking % Qauntiles of:\n  ", quantiles)
            print("Filter Trial", trial+1, "% Quantiles:\n  ", trial_filter_hits)
            xlabels = ["Q=" + str(q*100)+"%\n(Realized {})".format(np.round(trial_filter_hits[i]*100,2)) for i,q in enumerate(quantiles)]
            bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
            plt.figure()
            plt.title(r"Trial Realization Error Quantiles of the Filtered Position Covariance $P_k$" + "\n" + r"Computed as $R_Q=\frac{\sum_{k=1}^N\mathbb{1}(s_k \leq s_{Q})}{N}, \quad s_k = e^T_k P^{-1}_k e_k,\quad e_k = x^k_{1:3} - \hat{x}^k_{1:3}$")
            plt.ylabel("Trial Realization Quantiles Percentages " + r"$100*R_Q$")
            plt.bar(xlabels, trial_filter_hits*100, color=bar_colors)
            plt.xlabel("Total Filtering Steps in Realization={}, dt={} sec between steps".format(N, mc_dic["dt"]))
            plt.show()
            Ts = np.arange(xs.shape[0])
            es_track = [] 
            Ps_track = [] 
            for xhat,Phat,xt in zip(xs_kf,Ps_kf,xs):
                et, Pt, _ = get_cross_along_radial_errors_cov(xhat, Phat[0:3,0:3], xt)
                es_track.append(et)
                Ps_track.append(Pt)
            es_track = np.array(es_track)
            Ps_track = np.array(Ps_track)
            sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_kf])
            plt.figure()
            plt.suptitle("Filter Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
            plt.subplot(3,1,1)
            plt.plot(Ts, es_track[:, 0], 'g--')
            plt.plot(Ts, sig_bound[:, 0], 'm--')
            plt.plot(Ts, -sig_bound[:, 0], 'm--')
            plt.subplot(3,1,2)
            plt.plot(Ts, es_track[:, 1], 'g--')
            plt.plot(Ts, sig_bound[:, 1], 'm--')
            plt.plot(Ts, -sig_bound[:, 1], 'm--')
            plt.subplot(3,1,3)
            plt.plot(Ts, es_track[:, 2], 'g--')
            plt.plot(Ts, sig_bound[:, 2], 'm--')
            plt.plot(Ts, -sig_bound[:, 2], 'm--')
            plt.show()
            foobar = 2
            foobar = 2

        # Get smoother estimates
        if with_smoother:
            x_smoothed = xs_kf[-1].copy()
            P_smoothed = Ps_kf[-1].copy()
            xs_smoothed = [x_smoothed.copy()]
            Ps_smoothed = [P_smoothed.copy()]
            for i in reversed(range(N-1)):
                Phi = Phis_kf[i]
                x_kf = xs_kf[i]
                P_kf = Ps_kf[i]
                M_kf = Ms_kf[i+1]

                #C = P_kf @ Phi.T @ np.linalg.inv(M_kf)
                C = np.linalg.solve(M_kf, Phi @ P_kf).T
                fermiSat.reset_state(x_kf, i)
                x_smoothed = x_kf + C @ (x_smoothed - fermiSat.step() )
                P_smoothed = P_kf + C @ ( P_smoothed - M_kf ) @ C.T
                print("Smoothed Diff is: ", x_smoothed - x_kf)
                xs_smoothed.append(x_smoothed.copy())
                Ps_smoothed.append(P_smoothed.copy())
            xs_smoothed.reverse()
            Ps_smoothed.reverse()
            xs_smoothed = np.array(xs_smoothed)
            Ps_smoothed = np.array(Ps_smoothed)
            mc_dic["ekf_smoothed"].append((xs_kf, xs_smoothed))
            # Plot
            if with_plots:
                foo = np.zeros(xs_kf.shape[0])
                moment_info = {"x": xs_smoothed, "P": Ps_smoothed, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
                print("Smoother + EKF Results")
            ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=1)
        # Plot
        elif with_plots:
            print("EKF Results (No Smoother)")
            ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf), scale=1)

        # Clear Sim
        fermiSat.reset_initial_state(x07)

    # Log Data 
    dir_path = file_dir + "/pylog/gmat7/pred/mcdata_" + sas_Cd + "trials_" + str(mc_trials) + "_" + str(int(time.time())) + ".pickle"
    print("Writing MC Data To: ", dir_path)
    with open(dir_path, "wb") as handle:
        pickle.dump(mc_dic, handle)

def test_mc_trial_prediction():
    # Load Data 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_sastrials_2_1713828844.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    
    # Whether to use smoother or true state as error reference
    with_sim_truth = True 
    # Whether to include process noise in covariance propagation
    with_proc_noise_prop = False
    # Whether to check the trial's filter plot
    with_trial_filter_plot = False
    # Whether to check the trial's filter along, cross, and radial error/covariance plot
    with_trial_filter_along_cross_radial_track_plot = False
    # Whether to check each trial's filtered elliptic error quantiles
    with_trial_filter_quantile_plot = False
    # Whether to debug and plot each trials predicted state error vs the predicted variance bound
    with_trial_pred_state_err_plot = True
    # Whether to debug and plot each trials predicted along, cross, and radial state error vs the predicted along, cross, and radial variance bound
    with_trial_pred_along_cross_radial_track_plot = True
    # Whether to compute/plot the monte carlo trial averages of predictive elliptic error quantiles
    with_trial_pred_quantile_plot = True
    # Whether to compute/plot the monte carlo trial averages of predictive elliptic error quantiles
    with_mc_pred_quantile_plot = True
    # Whether to compute a monte carlo over all trials and compare it to the KFs Predictor Covariance
    with_mc_pred_covariance_avg_plot = True
    start_percent = 0.250 # halfway into the simulation

    # Load up the fermi sat model
    fermiSat = FermiSatelliteModel(mc_dic["x0"], mc_dic["dt"], mc_dic["std_gps_noise"])
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for("Cd", mc_dic["sas_Cd"], mc_dic["std_Cd"], mc_dic["tau_Cd"], alpha=mc_dic["sas_alpha"])
    Wn = mc_dic["EKFProcNoiseMat"] * with_proc_noise_prop

    # Get Chi-Squared Stats for three states 
    n_pred_states = 3
    quantiles = np.array([0.7, 0.9, 0.95, 0.99, 0.9999]) # quantiles
    qs = np.array([chi2.ppf(q, n_pred_states) for q in quantiles]) # quantile scores

    # Get sim length and set start index 
    N = mc_dic["sim_truth"][0][0].shape[0]
    start_idx = int(N * start_percent)
    mc_trials = mc_dic["trials"]

    # Get array of prediction steps vs chi squared bucket counts
    chi2_labels = [str(q) for q in quantiles]
    sim_steps = N-start_idx
    ellipsoid_trial_hits = np.zeros( (sim_steps, qs.size) )
    ellipsoid_hits = np.zeros(qs.size)
    ellipsoid_trial_scores = np.zeros(sim_steps)
    ellipsoid_scores = np.zeros(sim_steps)
    # Monte Carlo Stats
    eIs = np.zeros((mc_trials, sim_steps, n_pred_states))
    Ts = np.arange(N) #* mc_dic["dt"]
    print("Loaded Data From: ", dir_path)
    # For each trial, 
    # 1.) Propagate the state and covariance from start idx to end
    # 2.) At each propagation, compute whether the error lies within the covariances selected confidence ellipsoid percents
    for trial in range(mc_trials):
        print("Processing Trial {}/{}".format(trial+1, mc_trials))
        if with_sim_truth:
            xs,zs,ws,vs = mc_dic["sim_truth"][trial]
        else:
            xs,_ = mc_dic["ekf_smoothed"][trial]
        xs_kf,Ps_kf = mc_dic["ekf_runs"][trial]

        if with_trial_filter_quantile_plot:
            trial_filter_hits = np.zeros(qs.size)
            for i in range(N):
                x_kf = xs_kf[i]
                P_kf = Ps_kf[i]
                e_kf = (xs[i] - x_kf)[0:n_pred_states]
                Pinv = np.linalg.inv(P_kf[0:n_pred_states, 0:n_pred_states])
                score = e_kf.T @ Pinv @ e_kf
                hits = score < qs
                #print("Score: {}, qs:{}".format(score, qs))
                trial_filter_hits += hits
            trial_filter_hits /= N
            print("Checking % Quantiles of:\n  ", quantiles)
            print("Filter Trial", trial+1, "% Quantiles:\n  ", trial_filter_hits)
            xlabels = ["Q=" + str(q*100)+"%\n(Realized {})".format(np.round(trial_filter_hits[i]*100,2)) for i,q in enumerate(quantiles)]
            bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
            plt.figure()
            plt.title(r"Trial Realization Error Quantiles of the Filtered Position Covariance $P_k$" + "\n" + r"Computed as $R_Q=\frac{\sum_{k=1}^N\mathbb{1}(s_k \leq s_{Q})}{N}, \quad s_k = e^T_k P^{-1}_k e_k,\quad e_k = x^k_{1:3} - \hat{x}^k_{1:3}$")
            plt.ylabel("Trial Realization Quantiles Percentages " + r"$100*R_Q$")
            plt.bar(xlabels, trial_filter_hits*100, color=bar_colors)
            plt.xlabel("Total Filtering Steps in Realization={}, dt={} sec between steps".format(N, mc_dic["dt"]))
            plt.show()
            foobar = 2
        
        if with_trial_filter_plot:
            ce.plot_simulation_history(None, (xs,zs,ws,vs), (xs_kf,Ps_kf) )
            foobar = 2
        if with_trial_filter_along_cross_radial_track_plot:
            es_track = [] 
            Ps_track = [] 
            for xhat,Phat,xt in zip(xs_kf,Ps_kf,xs):
                et, Pt, _ = get_cross_along_radial_errors_cov(xhat, Phat[0:3,0:3], xt)
                es_track.append(et)
                Ps_track.append(Pt)
            es_track = np.array(es_track)
            Ps_track = np.array(Ps_track)
            sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_kf])
            plt.figure()
            plt.suptitle("Filter Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
            plt.subplot(3,1,1)
            plt.plot(Ts, es_track[:, 0], 'g--')
            plt.plot(Ts, sig_bound[:, 0], 'm--')
            plt.plot(Ts, -sig_bound[:, 0], 'm--')
            plt.subplot(3,1,2)
            plt.plot(Ts, es_track[:, 1], 'g--')
            plt.plot(Ts, sig_bound[:, 1], 'm--')
            plt.plot(Ts, -sig_bound[:, 1], 'm--')
            plt.subplot(3,1,3)
            plt.plot(Ts, es_track[:, 2], 'g--')
            plt.plot(Ts, sig_bound[:, 2], 'm--')
            plt.plot(Ts, -sig_bound[:, 2], 'm--')
            plt.show()
            foobar = 2


        # Set satellite at start index 
        x_kf = xs_kf[start_idx]
        P_kf = Ps_kf[start_idx]
        fermiSat.reset_state(x_kf, start_idx)
        # Propagate state and variance in prediction mode
        xkf_preds = [] 
        Pkf_preds = [] 
        ellipsoid_trial_scores *= 0
        #Wn *= 0
        #Wn[0,0] = 1e-5
        for idx in range(start_idx, N):
            # Propagate State and Covariance
            if idx != start_idx:
                Phi = fermiSat.get_transition_matrix(mc_dic["STM_order"])
                x_kf = fermiSat.step()
                P_kf = Phi @ P_kf @ Phi.T + Wn
                
                # Kinda works to bound the error
                #_,_, R = get_cross_along_radial_errors_cov(x_kf, P_kf[0:3,0:3], xs[idx])
                #foo = Wn.copy()
                #foo[0:3,0:3] = R.T @ foo[0:3,0:3] @ R
                #P_kf = Phi @ P_kf @ Phi.T + foo
                
            e_kf = (xs[idx] - x_kf)[0:n_pred_states]
            i = idx - start_idx
            if with_mc_pred_quantile_plot:
                Pinv = np.linalg.inv(P_kf[0:n_pred_states, 0:n_pred_states])
                score = e_kf.T @ Pinv @ e_kf
                ellipsoid_trial_scores[i] = score
                hits = score < qs
                ellipsoid_trial_hits[i, :] = hits
            # Append state info for predictive state error plots
            if with_trial_pred_state_err_plot:
                xkf_preds.append(x_kf[0:6])
                Pkf_preds.append(P_kf[0:n_pred_states, 0:n_pred_states])
            if with_mc_pred_covariance_avg_plot:
                eIs[trial, i, :] = e_kf
                # Store the Kalman Predictors covariance for a single trial, 
                # it should not vary very much from one simulation to another,
                # given that the simulation starts from the same location
                # This is to compare the MC covariance with the Kalman Predictors
                if not with_trial_pred_state_err_plot:
                    if trial == (mc_trials-1):
                        Pkf_preds.append(P_kf[0:n_pred_states, 0:n_pred_states])

        # Compute trial scores and hits
        sum_trial_hits = np.sum(ellipsoid_trial_hits, axis=0) / (N - start_idx)
        ellipsoid_hits += sum_trial_hits
        ellipsoid_scores += ellipsoid_trial_scores
        if with_trial_pred_state_err_plot:
            xkf_preds = np.array(xkf_preds)
            Pkf_preds = np.array(Pkf_preds)
            plt_title="Estimation Error of Predicted Position, Bounded by Predicted 1-Sigma Bound"
            plt_ylabels=("PosX Error (km)", "Posy (km)", "PosZ (km)")
            plt_xlabel = "Realization has Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(start_idx, N-start_idx,N, mc_dic["dt"])
            ce.plot_simulation_history(None, (xs[start_idx:, 0:n_pred_states],zs[start_idx:],ws[start_idx:,0:n_pred_states],vs[start_idx:]), (xkf_preds[:,0:3],Pkf_preds), labels=(plt_title,plt_ylabels,plt_xlabel), xshift=start_idx)
            foobar = 2
        if with_trial_pred_along_cross_radial_track_plot:
            es_track = [] 
            Ps_track = [] 
            for xhat,Phat,xt in zip(xkf_preds,Pkf_preds,xs[start_idx:,:]):
                et, Pt, _ = get_cross_along_radial_errors_cov(xhat, Phat[0:3,0:3], xt)
                es_track.append(et)
                Ps_track.append(Pt)
            es_track = np.array(es_track)
            Ps_track = np.array(Ps_track)
            sig_bound = np.array([np.diag(_P)**0.5 for _P in Ps_track])
            plt.figure()
            plt.suptitle("Predictor Along Track, Cross Track, Radial Track Errors vs 1-Sigma Bound")
            plt.subplot(3,1,1)
            plt.plot(Ts[start_idx:], es_track[:, 0], 'g--')
            plt.plot(Ts[start_idx:], sig_bound[:, 0], 'm--')
            plt.plot(Ts[start_idx:], -sig_bound[:, 0], 'm--')
            plt.subplot(3,1,2)
            plt.plot(Ts[start_idx:], es_track[:, 1], 'g--')
            plt.plot(Ts[start_idx:], sig_bound[:, 1], 'm--')
            plt.plot(Ts[start_idx:], -sig_bound[:, 1], 'm--')
            plt.subplot(3,1,3)
            plt.plot(Ts[start_idx:], es_track[:, 2], 'g--')
            plt.plot(Ts[start_idx:], sig_bound[:, 2], 'm--')
            plt.plot(Ts[start_idx:], -sig_bound[:, 2], 'm--')
            plt.show()
            foobar=2
        # Plot Ellipsoid Trial percentages
        if with_trial_pred_quantile_plot:
            plt.figure()
            plt.title(r"Trial Realization of Elliptic Error Scores $s_{k+i|k} = e^T_{k+i|k} P^{-1}_{k+i|k} e_{k+i|k},\quad e_{k+i|k} = x^{k+i|k}_{1:3} - \hat{x}^{k+i|k}_{1:3}$ of the Predicted Position Covariance $P_{k+i|k}$" + "\n(Ie: What ellipsoid of the covariance matrix does the propagated state (error) reside within?)")
            plt.plot(Ts[start_idx:], ellipsoid_trial_scores, label = "Ellipsoid scores")
            for i in range(quantiles.size):
                plt.plot(Ts[start_idx:], np.ones(Ts[start_idx:].size) * qs[i], label = "quantile-" + str(quantiles[i]*100) + "%" )
            plt.legend(loc=1, prop={'size' : 12})
            plt.xlabel("Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(start_idx, N-start_idx,N, mc_dic["dt"]))
            plt.show() 
            xlabels = ["Q=" + str(q*100)+"%\n(Realized {})".format(np.round(sum_trial_hits[i]*100,2)) for i,q in enumerate(quantiles)]
            bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
            plt.figure()
            plt.title(r"Trial Realization of Elliptic Error Quantiles of the Predicted Position Covariance $P_{k+i|i}$" + "\n" + r"Computed as $R_Q=\frac{\sum_{i=1}^{N-k}\mathbb{1}(s_{k+i|k} \leq s_{Q})}{N}, \quad s_{k+i|k} = e^T_{k+i|k} P^{-1}_{k+i|k} e_{k+i|k},\quad e_{k+i|k} = x^{k+i|k}_{1:3} - \hat{x}^{k+i|k}_{1:3}$")
            plt.ylabel("Trial Quantile Percentage " + r"$100*R_Q$")
            plt.bar(xlabels, sum_trial_hits*100, color=bar_colors)
            plt.xlabel("Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(start_idx, N-start_idx,N, mc_dic["dt"]))
            plt.show()
            foobar = 2
    ellipsoid_hits /= mc_trials
    ellipsoid_scores /= mc_trials

    if with_mc_pred_covariance_avg_plot:
        e_bar = np.mean(eIs, axis = 0) # average error across all trials
        P_mc = np.zeros( (sim_steps, n_pred_states, n_pred_states) )
        for i in range(mc_trials):
            e_ks = eIs[i] - e_bar
            e_ks = e_ks.reshape((*e_ks.shape,1))
            P_mc += e_ks @ e_ks.transpose((0,2,1))
        if mc_trials > 1:
            P_mc /= mc_trials
        ylabels = ["PosX", "PosY", "PosZ", "VelX", "VelY", "VelZ", "Drag"] 
        if with_trial_pred_state_err_plot:
            Pkf_preds = np.array(Pkf_preds)
        one_sigs_mc = np.array([np.diag(pmc)**0.5 for pmc in P_mc])
        one_sigs_kf = np.array([np.diag(pkf)**0.5 for pkf in Pkf_preds])
        plt.figure()
        plt.suptitle("Monte Carlo Average of Predicted (one-sig) standard deviation (r) vs Kalman Predictor's (one-sig) standard deviation (b)")
        for i in range(n_pred_states):
            plt.subplot(n_pred_states,1,i+1)
            plt.plot(Ts[start_idx:], one_sigs_mc[:, i], 'r')
            plt.plot(Ts[start_idx:], one_sigs_kf[:, i], 'b')
            plt.ylabel(ylabels[i])
            if i == (n_pred_states-1):
                plt.xlabel("Total MC Trials Averaged={}, Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(mc_trials, start_idx, N-start_idx,N, mc_dic["dt"]))
        plt.show()
        foobar = 2

    # Plot Ellipsoid MC percentages
    if with_mc_pred_quantile_plot:
        plt.figure()
        plt.title(r"Monte Carlo Averages of Elliptic Error Scores $s_{k+i|k} = e^T_{k+i|k} P^{-1}_{k+i|k} e_{k+i|k},\quad e_{k+i|k} = x^{k+i|k}_{1:3} - \hat{x}^{k+i|k}_{1:3}$ of the Predicted Position Covariance $P_{k+i|k}$" + "\n(Ie: What ellipsoid of the covariance matrix does the propagated state (error) reside within?)")
        plt.plot(Ts[start_idx:], ellipsoid_scores, label = "Ellipsoid scores")
        for i in range(quantiles.size):
            plt.plot(Ts[start_idx:], np.ones(Ts[start_idx:].size) * qs[i], label = "quantile-" + str(quantiles[i]*100) + "%" )
        plt.legend(loc=1, prop={'size' : 12})
        plt.xlabel("Total MC Trials Averaged={}, Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(mc_trials, start_idx, N-start_idx,N, mc_dic["dt"]))
        plt.show() 
        xlabels = ["Q=" + str(q*100)+"%\n(Realized {})".format(np.round(ellipsoid_hits[i]*100,2)) for i,q in enumerate(quantiles)]
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
        plt.figure()
        plt.title(r"Monte Carlo Averages of Elliptic Error Quantiles of the Predicted Position Covariance $P_{k+i|i}$" + "\n" + r"Computed as $R_Q=\frac{\sum_{i=1}^{N-k}\mathbb{1}(s_{k+i|k} \leq s_{Q})}{N}, \quad s_{k+i|k} = e^T_{k+i|k} P^{-1}_{k+i|k} e_{k+i|k},\quad e_{k+i|k} = x^{k+i|k}_{1:3} - \hat{x}^{k+i|k}_{1:3}$")
        plt.ylabel("MC Average Quantile Percentage " + r"$100*R_Q$")
        plt.bar(xlabels, ellipsoid_hits*100, color=bar_colors)
        plt.xlabel("Total MC Trials Averaged={}, Filter Steps={}, Prediction Steps={}, Total Steps={}. dt={} sec between steps".format(mc_trials, start_idx, N-start_idx,N, mc_dic["dt"]))
        plt.show()
    foobar=2


if __name__ == "__main__":
    #tut1_simulating_an_orbit()
    #fermi_sat_prop()
    #test_jacchia_roberts_influence()
    #test_reset_influence()
    #test_solve_for_state_derivatives_influence()
    #test_gmat_ekf6()
    test_gmat_ekf7()
    #test_gmat_ekfs7()
    #test_gmat_ece7()
    #test_prediction_results()
    #test_point_in_ellipse()
    #test_mc_trial_logger()
    #test_mc_trial_prediction()