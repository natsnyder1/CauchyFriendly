## Overview
This repository contains C++, Python, and Matlab interfaces for the Multivariate Cauchy Estimator (MCE) algorithm. The repository is compilable on Linux/Mac/Windows operating systems. The Python and Matlab wrappers (built via Swig, Mex, respectively) bind the Python/Matlab interfaces to the C++ backend. See beneath the attached video for installation instructions. 

## What is the Multivariate Cauchy Estimator?
Much like the Kalman filter (KF) + Extended Kalman filter (EKF), the MCE + Extended MCE (EMCE) can be used for state estimation of linear time-invariant (LTI), linear time-varying (LTV) and non-linear dynamic systems. The MCE algorithm scales well to moderate state space dimensions of up to seven or eight: similar to the limitations of a particle filter.

Aside from the Kalman filter, the MCE is the only other analytic, recursive, and closed-form Bayesian state estimation algorithm. Moreover, it is the only estimator capable of evolving its cpdf into rich, sometimes non-symmetric, and sometimes multi-modal state hypotheses, solely as a function of its measurement history. The MCE is very robust (because of the aforementioned attribute) to impulsive process noises and outliers in the measurement. The key difference between the KF and the MCE is that the MCE algorithm assumes additive Cauchy process and measurement noise, whereas the Kalman filter assumes Gaussian process and measurement noise. Empirically too, it is seen to perform robustly when system dynamic parameters become misspecified. The following papers are good resources to learn more about the estimator:
> LINK1

> LINK2

> LINK3

## Video Demonstrations
The following videos show the rich CDPF the Cauchy Estimator constructs for: 

> 1.) a simple 1D LTI state estimation problem, and compared to the performance of a Kalman filter, tuned similarly.

> 2.) a simple 2D LTI state estimation problem, and compared to a Kalman filter, tuned similarly.

https://github.com/user-attachments/assets/c798460e-651b-450b-9f19-3c9902800d5f

this rich cpdf structure is the key to the MCE's robust state estimation performance.

## Dependencies:
> Linux:
>> C++: g++, make
>> Python: see scripts/requirements.txt, swig, g++
>> Matlab: matlab installed, g++
> Mac:
>> C++: clang++, make
>> Python: see scripts/requirements.txt, swig, clang++
>> Matlab: matlab installed, Xcode installed (and license accepted), clang++
> Windows:
>> C++: a Microsoft Visual Studio installation (for cl.exe and link.exe), with Windows Kit SDK for C++ compilation (ucrt, um, shared)
>> Python: see scripts/requirements.txt, swig, cl.exe and link.exe
>> Matlab: matlab installed, clang++

## Auto-Installation Instructions for Linux/Mac/Windows:
The Python script auto_config.py can be run using a python version>=3.0 of your choosing:
> i.e, python3.7 auto_config.py
which will ask you whether you'd like to build the C++ MCE examples, the Python MCE module, and the Matlab MCE module. If you intend to build the Python MCE module, please run "auto_config.py" with your INTENDED Python version. If you run into a bug with this script, please email natsnyder1@gmail.com with a screenshot and a brief explanation. Doing so will make this configuration script more robust for others. Successful compilation yields:

> For C++ Build:
>>  C++ examples in bin/ . As a side note, the include/cauchy_estimator.hpp and include/cauchy_windows.hpp files contain the MCE source codes.

> For Python Build:
>> The Python MCE module: scripts/swig/cauchy/cauchy_estimator.py
>> Jupyter Notebook tutorial on how to use the MCE for LTI systems: scripts/tutorials/lti_systems.ipynb
>> Jupyter Notebook tutorial on how to use the MCE for nonlinear systems: scripts/tutorials/nonlin_systems.ipynb
>> Python script examples: scripts/swig/cauchy/test_pycauchy.py and scripts/swig/cauchy/test_pycauchy_nonlin.py (and more)

> For Matlab Build
>> The Matlab MCE modules: matlab/matlab_pure/MCauchyEstimator.m and matlab/matlab_pure/MSlidingWindowManager.m
>> Tutorial on how to use the MCE for LTI systems: matlab/lti_systems.mlx
>> Tutorial on how to use the MCE for nonlinear systems: matlab/nonlin_systems.mlx
>> Matlab script examples: matlab/test_lti.m and matlab/test_nonlin.m (and more)

## Manual Installation Instructions For Linux:
The Python script auto_config.py is configuring:
> C++ Build:
>> Linux/Mac: The Makefile, which builds the C++ examples in bin/
>> Windows: The batch file win_cpp_make.bat, which builds the C++ examples in bin/ . The win_cpp_make.bat also sets up the C++ MSVC project solution located at scripts/windows/CauchyWindows/CauchyWindows.sln, which you can use

> Python Build:
>> Linux/Mac: The shell script scripts/swig/cauchy/swigit_unix.sh, which uses swig to build the C++ backend for the Python MCE module scripts/swig/cauchy/cauchy_estimator.py
>> Windows: The batch file scripts/swig/cauchy/swigit_windows.bat, which uses swig to build the C++ backend for the Python MCE module scripts/swig/cauchy/cauchy_estimator.py

> Matlab Build:
>> Linux/Mac/Windows: The matlab file matlab/mex_files/build.m, which uses mex to build the C++ backend for the MCE modules matlab/matlab_pure/MCauchyEstimator.m and matlab/matlab_pure/MSlidingWindowManager.m

> # C++ Build Manual Configuration
>> Linux: In Makefile, change variable CC=g++ (if not already set). Run make clean && make cauchy window D=0
>> Mac: In Makefile, change variable CC=clang++ (if not already set). Run make clean && make cauchy window D=0
>> Windows: In the batch file win_cpp_make.bat, change variables INC_MSVC through LIB_UCRT to the appropriate paths. Run .\win_cpp_make.bat or click on this batch file in finder.
> # Python Build Manual Configuration
>> Linux: In scripts/swig/cauchy/swigit_unix.sh, set variables INC_PYTHON, LIB_PYTHON, INC_NUMPY. Change swig executable path if needed (line 30). Change compiler to g++ (lines 35, 40), if not already set. Run the script as ./swigit_unix.sh
>> Mac: In scripts/swig/cauchy/swigit_unix.sh, set variables INC_PYTHON, LIB_PYTHON, INC_NUMPY. Change swig executable path if needed (line 30). Change compiler to clang++ (lines 35, 40), if not already set. Run the script as ./swigit_unix.sh
>> Windows: In the batch file scripts/swig/cauchy/swigit_windows.bat, set variables MY_EXE through LIB_MSVC (lines 18-35). Run .\win_cpp_make.bat or click on this batch file in finder.
> # Matlab Build Manual Configuration
>> Linux: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath. Open the matlab GUI and run matlab/mex_files/build.m
>> Mac: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath. Note you must have Xcode installed. Open the matlab GUI and run matlab/mex_files/build.m
>> Windows: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath. Note you must have a Microsoft Visual Studio version installed. Open the matlab GUI and run matlab/mex_files/build.m

## Software Restrictions and License Note:
The software is free and open to use by anyone for non-commercial purposes. For commercial licensure, contact natsnyder1@gmail.com or speyer@g.ucla.edu.
