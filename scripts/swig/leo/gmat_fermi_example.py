import numpy as np 
import gmat_sat as gsat

# Basic Example -- No Solve Fors
gmat_print = True

t0 = '11 Feb 2023 23:47:55.0' # t0 can also be a datetime object
x0 = np.array([
    4996.245288270519, 3877.946463086103, 2736.0432364171807,  # km
    -5.028093574446193, 5.575921341999267, 1.2698611722905329])  # km/s
print("x0: ", x0)
dt = 60 # seconds
fermiSat = gsat.FermiSatelliteModel(t0, x0, dt, gmat_print = gmat_print)
fermiSat.create_model(with_jacchia=True, with_SRP=True)
x1 = fermiSat.step()
print("x1: ", x1)
# Can get STM or Jacobian (6x6) for current state vector x1 easily as 
#Jac = fermiSat.get_jacobian_matrix()
#Phi = fermiSat.get_transition_matrix(taylor_order=4)

# Can Dynamically Change Step Size (dt)... will now propagate satellite for 30 seconds
fermiSat.dt = 30
x2 = fermiSat.step()
print("x2: ", x2)
# Can get STM at new state x2 and assuming your newly set dt
#Phi = fermiSat.get_transition_matrix(taylor_order=4)

# Can reset state arbitrarily
fermiSat.reset_state_with_ellapsed_time(x1, ellapsed_time=60) # ellapsed time since t0
x2 = fermiSat.step()
print("x2 (again): ", x2)

fermiSat.clear_model()
print("-------------")

# Examples with solve fors added (Useful for simulation and estimation purposes)
# With Solve Fors -> Currently changes in Cd and Cr Supported
x0 = np.array([
    4996.245288270519, 3877.946463086103, 2736.0432364171807,  # km
    -5.028093574446193, 5.575921341999267, 1.2698611722905329, # km/s
    0.15 # Change in Cd -> Cd = Cd0 * (1 + delta_Cd0) -> Here delta_Cd0 is our state and delta_Cd0 = 0.15
    ])  
print("x0': ", x0)

sas_dist = "sas" # or "gauss" (sas is the heavy tailed distribution class when <= 2.0)
std_Cd = 0.0013 # scaling parameter of sas distribution -> Only used if you want to use "fermiSat.simulate(...)" -> otherwise doesnt matter, can set to 0
tau_Cd = 21600 # FOGM param -> \dot{x} = -1/tau * x + noise
sas_Cd = 1.3 if sas_dist == "sas" else 2.0 # alpha parameter of SAS distribution... Only used if you want to use "fermiSat.simulate(...)" (sas_Cd = 2.0 -> GAUSSIAN) (Sas_Cd = 1.0 -> CAUCHY) (1.0 <= sas_Cd <= 2.0)
fermiSat = gsat.FermiSatelliteModel(t0, x0[0:6], dt, gmat_print = gmat_print)
fermiSat.create_model(with_jacchia=True, with_SRP=True)
fermiSat.set_solve_for("Cd", sas_dist, std_Cd, tau_Cd, alpha=sas_Cd)
fermiSat.reset_state_with_ellapsed_time(x0, 0)
#fermiSat.get_state() will return to you current state vector
x1 = fermiSat.step()
print("x1': ", x1)
# Can get STM or Jacobian (7x7) for State Vector easily now as 
#fermiSat.get_jacobian_matrix()
#fermiSat.get_transition_matrix(taylor_order=4)

# Can Conduct Simulation as ...
# This will simulate changes to Cd using the distribution / scaling you've set above
num_orbits = 3
gps_std_dev = 0.0075 # km 
W = None # 6x6 Proc Noise Covariance ... If you'd like to add noise to position/velocity 
xs, zs, ws, vs = fermiSat.simulate(num_orbits, gps_std_dev, W = W)

#Can also do so with "steps" and not "orbits"
steps = 250
gps_std_dev = 0.0075 # km 
W = None # 6x6 Proc Noise Covariance ... If you'd like to add noise to position/velocity 
xs, zs, ws, vs = fermiSat.simulate(None, gps_std_dev, W = W, num_steps=steps)
foobar = 2