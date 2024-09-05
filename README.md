This repository contains the Multivariate Cauchy Estimator (MCE) API for C++/Python/Matlab and is compilable on Mac/Linux/Windows OS. The Python and Matlab wrappers (built via Swig, Mex, respectively) bind the Python/Matlab MCE's modules to the fast C++ backend library. See beneath the attached video for installation instructions. 

## What is the Multivariate Cauchy Estimator?
Much like the Kalman filter (KF) / Extended Kalman filter (EKF), the MCE / Extended MCE (EMCE) can be used for state estimation of linear time-invariant (LTI), linear time-varying (LTV) and non-linear dynamic systems. The MCE algorithm scales well to moderate state space dimensions of up to sizes seven or eight, much like the limitations on a particle filter.

Aside from the Kalman filter, the MCE is the only other analytic, recursive, and closed-form Bayesian state estimation algorithm (that is capable of dynamically propagating a multivariate conditional probability density function (cpdf) of the system state, given its measurement history). Moreover, it is the only estimator capable of generating a multi-modal cpdf solely as a function of its measurement history and in a closed form. This makes the MCE very robust to impulsive process noise and outliers in the measurement; a byproduct of its modeled Cauchy process and measurement noise assumption (whereas the Kalman filter assumes Gaussian process and measurement noise). Empirically too, it is seen to perform robustly when system dynamic parameters become misspecified.

## Video Demonstrations
The following video shows the robust structure the Cauchy Estimator's CPDF can take on: shown here for a simple 1D LTI state estimation problem and compared to the performance of a Kalman filter tuned equivalently.

https://github.com/user-attachments/assets/c798460e-651b-450b-9f19-3c9902800d5f

## Auto-Installation Instructions for Linux/Max/Windows:

##Manual Installation Instructions For Linux:
> #C++ API and Examples:
    
> #Python API and Examples:
    
> #Matlab API and Examples:
    
##Manual Installation Instructions For Mac:
> #C++ API and Examples:
    
> #Python API and Examples:
    
> #Matlab API and Examples:
    
##Manual Installation Instructions for Windows:
> #C++ API and Examples:
    
> #Python API and Examples:
    
> #Matlab API and Examples:
    
