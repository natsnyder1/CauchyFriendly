import time
import numpy as np
import matplotlib.pyplot as plt 
import sys, os

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

def fermi_sat(r_sat = 550e3):
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
    #psm.SetProperty("AMatrix")
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
    num_orbits = 25
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


def tut1_simulating_an_orbit():
    sat = gmat.Construct("Spacecraft", "Sat")
    # Set properties of spacecraft
    sat.SetField("DateFormat", "UTCGregorian") #EpochFormat Form Box
    sat.SetField("Epoch", "22 Jul 2014 11:29:10.811")
    sat.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sat.SetField("DisplayStateType","Keplerian") #StateType Form Box
    sat.SetField("SMA", 83474.318) #km
    sat.SetField("ECC", 0.89652) 
    sat.SetField("INC", 12.4606) #deg
    sat.SetField("RAAN", 292.8362) #deg
    sat.SetField("AOP", 218.9805) #deg
    sat.SetField("TA", 180)#deg
    sat.SetField("DryMass", 850)
    sat.SetField("Cd", 2.2)
    sat.SetField("Cr", 1.8)
    sat.SetField("DragArea", 15)
    sat.SetField("SRPArea", 1)

    # Create Force Model
    # Force model of forces acting upon spacecraft
    fm = gmat.Construct("ForceModel", "TheForces")
    fm.SetField("ErrorControl", "RSSStep")
    # An 8x8 JGM-3 Gravity Model
    earthgrav = gmat.Construct("GravityField")
    earthgrav.SetField("BodyName","Earth")
    earthgrav.SetField("Degree",10)
    earthgrav.SetField("Order",10)
    earthgrav.SetField("PotentialFile","JGM2.cof")
    # atmosphere model
    jrdrag = gmat.Construct("DragForce")
    jrdrag.SetField("AtmosphereModel","JacchiaRoberts")
    # Build and set the atmosphere for the model
    atmos = gmat.Construct("JacchiaRoberts")
    jrdrag.SetReference(atmos)
    # The Point Masses
    moongrav = gmat.Construct("PointMassForce")
    moongrav.SetField("BodyName","Luna")
    sungrav = gmat.Construct("PointMassForce")
    sungrav.SetField("BodyName","Sun")
    # Solar Radiation Pressure
    srp = gmat.Construct("SolarRadiationPressure")
    srp.SetField("SRPModel", "Spherical")

    # Add all of the forces into the ODEModel container
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
    pdprop.SetField("InitialStepSize", 60.0)
    pdprop.SetField("Accuracy", 1.0e-12)
    pdprop.SetField("MinStep", 0.001)
    pdprop.SetField("MaxStep", 60)
    pdprop.SetField("MaxStepAttempts", 50)


    # Assign the force model to the propagator
    pdprop.SetReference(fm)
    # It also needs to know the object that is propagated
    pdprop.AddPropObject(sat)

    # Setup the state vector used for the force, connecting the spacecraft
    psm = gmat.PropagationStateManager()
    psm.SetObject(sat)
    #psm.SetProperty("AMatrix")
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

    # Run simulation

    num_steps_mission = 4002 
    dt = 60
    t = 0
    times = []
    positions = []
    velocities = []

    for _ in range(num_steps_mission):
        gatorstate = np.array(gator.GetState())
        r = gatorstate[0:3]
        v = gatorstate[3:6]
        positions.append(r)
        velocities.append(v)
        times.append(t)

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

        # Now step the integrator
        gator.Step(dt)
        t += dt
    # Final State
    gatorstate = np.array(gator.GetState())
    r = gatorstate[0:3]
    v = gatorstate[3:]
    positions.append(r)
    velocities.append(v)
    positions = np.array(positions)
    velocities = np.array(velocities)
    times.append(t)

    #plt.plot(positions[:,0], positions[:,1])
    #plt.show()
    #barfoo = 11
    data_folder = gmat_api_dir + '/nat_tests'
    gmat_gui_data = parse_gmat_data(data_folder + "/hello_world_data.txt")

    Ncmp = 4000
    print("Max Diff X: ", np.max(np.abs(positions[:Ncmp,0] - gmat_gui_data["X"][:Ncmp])))
    print("Max Diff Y: ", np.max(np.abs(positions[:Ncmp,1] - gmat_gui_data["Y"][:Ncmp])))
    print("Max Diff Z: ", np.max(np.abs(positions[:Ncmp,2] - gmat_gui_data["Z"][:Ncmp])))
    print("Max Diff VX: ", np.max(np.abs(velocities[:Ncmp,0] - gmat_gui_data["VX"][:Ncmp])))
    print("Max Diff VY: ", np.max(np.abs(velocities[:Ncmp,1] - gmat_gui_data["VY"][:Ncmp])))
    print("Max Diff VZ: ", np.max(np.abs(velocities[:Ncmp,2] - gmat_gui_data["VZ"][:Ncmp])))
    

if __name__ == "__main__":
    #tut1_simulating_an_orbit()
    fermi_sat()