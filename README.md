This repository contains the Multivariate Cauchy Estimator (MCE) API for C++/Python/Matlab and is compilable on Mac/Linux/Windows OS. The Python and Matlab wrappers (built via Swig, Mex, respectively) bind the Python/Matlab MCE's modules to the fast C++ backend library. See beneath the attached video for installation instructions. 

Much like the Kalman filter (KF) / Extended Kalman filter (EKF), the MCE / Extended MCE (EMCE) can be used for state estimation of linear time-invariant (LTI), linear time-varying (LTV) and non-linear dynamic systems. The MCE scales well to moderate state space dimensions of up to sizes seven or eight, much like the limitations on a particle filter.

Aside from the Kalman filter, the MCE is the only other analytic, recursive, and closed-form Bayesian state estimation algorithm (capable of dynamically propagating a multivariate conditional probability density function (cpdf) of the system state, given its measurement history). Moreover, it is the only estimator capable of generating a multi-modal cpdf solely as a function of its measurement history and in a closed form. 

This repository provides access to this new estimation tool. The following video shows the robust structure the Cauchy Estimator's CPDF can take on: shown here for a simple 1D LTI state estimation problem and compared to the performance of a Kalman filter tuned equivalently.

https://github.com/user-attachments/assets/c798460e-651b-450b-9f19-3c9902800d5f

Installation Instructions For Linux:
  Auto C++, Python, Matlab Configuration:
    
  C++ API and Examples:
    
  Python API and Examples:
    
  Matlab API and Examples:
    
Installation Instructions For Mac:

Installation Instructions for Windows:
