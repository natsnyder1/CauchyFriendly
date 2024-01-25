import time
import numpy as np
import math
import matplotlib.pyplot as plt 
import sys, os
import cauchy_estimator as ce

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
def leo7_process_noise_model(dt):
    q = 8e-18; # Process noise ncertainty in the process position and velocity
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

    def create_model(self, with_jacchia=True, with_SRP=True):
        # Create Fermi Model
        self.sat = gmat.Construct("Spacecraft", "Fermi")
        self.sat.SetField("DateFormat", "UTCGregorian") #EpochFormat Form Box
        self.sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
        self.sat.SetField("DisplayStateType","Cartesian") #StateType Form Box
    
        self.sat.SetField("Epoch", "10 Jul 2023 19:31:54.000") # 19:31:54:000") 
        self.sat.SetField("DryMass", 3995.6)
        self.sat.SetField("Cd", 2.1)
        self.sat.SetField("CdSigma", 0.21)
        #sat.SetField("AtmosDensityScaleFactor", 1.0) # Gives Error Message
        #sat.SetField("AtmosDensityScaleFactorSigma", 0.8) # Gives Error Message
        self.sat.SetField("Cr", 1.8)
        self.sat.SetField("CrSigma", 0.1)
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

    def clear_model(self):
        gmat.Clear()
    
    def get_state(self):
        return self.gator.GetState() 

    def reset_state(self, x, iter):
        self.sat.SetState(*x)
        self.fm.BuildModelFromMap()
        self.fm.UpdateInitialData()
        self.pdprop.PrepareInternals()
        self.gator = self.pdprop.GetPropagator() # refresh integrator
        self.gator.SetTime(iter * self.dt)

    def reset_initial_state(self, x):
        self.x0 = x.copy()

    def get_transition_matrix(self, order):
        pstate = self.gator.GetState() #sat.GetState().GetState()
        self.fm.GetDerivatives(pstate, dt=self.dt, order=1) #, dt=dt) #, dt=dt, order=1) #, t, 2, -1)
        fdot = self.fm.GetDerivativeArray()
        #dx_dt = fdot[0:6]
        Jac = np.array(fdot[6:42]).reshape(6,6)
        Phi = np.eye(6) + Jac * self.dt
        for i in range(2, order+1):
            Phi += np.linalg.matrix_power(Jac, i) * self.dt**i / math.factorial(i)
        return Phi

    def step(self):
        self.gator.Step(self.dt)
        xk = np.array(self.gator.GetState())
        return xk

    # Must be less than self.dt
    def step_arbitrary_dt(self, dt):
        assert(dt < self.dt)
        self.gator.Step(dt)
        xk = np.array(self.gator.GetState())
        return xk

    def simulate(self, num_orbits, W=None):
        r0 = np.linalg.norm(self.x0[0:3])
        v0 = np.linalg.norm(self.x0[3:])
        omega0 = v0/r0 # rad/sec (angular rate of orbit)
        orbital_period = 2.0*np.pi / omega0 #Period of orbit in seconds
        time_steps_per_period = (int)(orbital_period / self.dt + 0.50) # number of dt's until 1 revolution is made
        num_mission_steps = num_orbits * time_steps_per_period
        # Measurement before propagation
        x0 = self.x0.copy() 
        v0 = np.random.randn(3) * self.gps_std_dev
        z0 = x0[0:3] + v0
        states = [x0]
        msmt_noises = [v0]
        msmts = [z0]
        proc_noises = [] if W is not None else None
        # Begin loop for propagation
        for i in range(num_mission_steps):
            # Now step the integrator
            self.gator.Step(self.dt)
            # Get new state
            xk = np.array(self.gator.GetState())
            # Add process noise, if given
            if W is not None:
                wk = np.random.multivariate_normal(np.zeros(6), W)
                xk += wk
                self.reset_state(xk, i+1)
                proc_noises.append(wk)
            states.append(xk)
            #Form measurement
            vk = np.random.randn(3) * self.gps_std_dev
            zk = xk[0:3] + vk
            msmts.append(zk)
            msmt_noises.append(vk)
        # Reset Simulation to x0, and return state info
        self.reset_state(self.x0, 0)
        return np.array(states), np.array(msmts), np.array(proc_noises), np.array(msmt_noises)

def test_gmat_ekf():
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
    W = leo7_process_noise_model(dt)
    # Create Satellite Model 
    fermiSat = FermiSatelliteModel(x0, dt, std_gps_noise)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xs, zs, ws, vs = fermiSat.simulate(num_orbits, W=None)
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
    plt.title("Leo Trajectory over Time")
    ax.plot(xs[:,0], xs[:,1], xs[:,2], color = 'r')
    ax.plot(xs_kf[:,0], xs_kf[:,1], xs_kf[:,2], color = 'b')
    plt.show()
    foobar = 2

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
    W = leo7_process_noise_model(dt)
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
    W = leo7_process_noise_model(dt)
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


if __name__ == "__main__":
    #tut1_simulating_an_orbit()
    #fermi_sat_prop()
    test_gmat_ekf()
    #test_jacchia_roberts_influence()
    #test_reset_influence()