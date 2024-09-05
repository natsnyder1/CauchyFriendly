## Overview
This repository contains C++, Python, and Matlab interfaces for the Multivariate Cauchy Estimator (MCE) algorithm. The repository is compilable for Linux/Mac/Windows operating systems. The Python and Matlab wrappers (built via Swig, Mex, respectively) bind the Python/Matlab interfaces to its C++ backend. See beneath the attached video for installation instructions. 

## What is the Multivariate Cauchy Estimator?
Much like the Kalman filter (KF), the MCE can be used for state estimation of linear time-invariant (LTI) or linear time-varying (LTV) dynamic systems. For nonlinear systems, the extended MCE (EMCE) can be applied just as an extended Kalman filter (EKF) could. The (E)MCE algorithm scales well to moderate state space dimensions of up to seven or eight: similar to the limitations of a particle filter.

Aside from the KF, the MCE is the only other analytic, recursive, and closed-form Bayesian state estimation algorithm. The key difference between the KF and the MCE is that the MCE algorithm assumes additive Cauchy process and measurement noise, whereas the KF assumes Gaussian process and measurement noise. Moreover, it is the only estimator capable of evolving its cpdf (of the system state, given the measurement histroy) into a rich, sometimes non-symmetric, and sometimes multi-modal distrubution; solely as a function of its measurement history. The aforementioned fact allows the MCE to estimate robustly when subjected to impulsive process noises, outliers in the measurement, or both. Empirically, the MCE is seen to perform well too when system dynamic parameters become misspecified. The tutorials in this repository will allow you to become familiar with the MCE by example. The following papers are good resources to learn more about the mathematics of the Cauchy Estimator:
> LINK1

> LINK2

> LINK3

## Video Demonstrations
The following videos show the evolution of the MCE's cpdf (given its measurement history), which was simulated using impulsive process and measurement noise, and for: 

> 1.) a simple 1D LTI state estimation problem, with performance compared to a similarly tuned KF.

https://github.com/user-attachments/assets/c798460e-651b-450b-9f19-3c9902800d5f

> 2.) a simple 2D LTI state estimation problem, with performance compared to a similarly tuned KF.

https://github.com/user-attachments/assets/aea57e84-3765-43ce-bcad-e302bb40d380

the rich cpdf structure observed is the key to the MCE's robust state estimation performance.

## Dependencies:
> Linux:
>> C++: g++, make

>> Python: see scripts/requirements.txt, swig, g++
>>> Note you can 'pip3.7 install -r scripts/requirements.txt' using your pip version

>> Matlab: matlab installed, g++

> Mac:
>> C++: clang++, make

>> Python: see scripts/requirements.txt, swig, clang++
>>> Note you can 'pip3.7 install -r scripts/requirements.txt' using your pip version

>> Matlab: matlab installed, Xcode installed (and license accepted), clang++

> Windows:
>> C++: a Microsoft Visual Studio installation (for cl.exe and link.exe), with Windows Kit SDK for C++ compilation (ucrt, um, shared)

>> Python: see scripts/requirements.txt, swig, cl.exe and link.exe
>>> Note you can 'pip3.7 install -r scripts/requirements.txt' using your pip version

>> Matlab: matlab installed, (MSVC cl.exe and link.exe + Windows Kit SDK) -> the C++ reqs

## Auto-Installation Instructions for Linux/Mac/Windows:
The Python script auto_config.py can be run using a python version>=3.0 of your choosing:
> i.e, python3.7 auto_config.py

which will ask you whether you'd like to build the C++ MCE examples, the Python MCE module, and the Matlab MCE module. If you intend to build the Python MCE module, please run "auto_config.py" with your INTENDED Python version. If you run into a bug with this script, please email natsnyder1@gmail.com with a screenshot and a brief explanation. Doing so will make this configuration script better for others. If the script exits successfully, it will have built:

> For C++ Build:
>>  C++ examples in bin/ . For windows, the script also sets up the C++ MSVC project solution located at scripts/windows/CauchyWindows/CauchyWindows.sln which can be used to compile, debug, and run src/cauchy_estimator.cpp

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

## Manual Installation Instructions:
The Python script auto_config.py is configuring:
> C++ Build:
>> Linux/Mac: The Makefile, which builds the C++ examples in bin/

>> Windows: The batch file win_cpp_make.bat, which builds the C++ examples in bin/ . The win_cpp_make.bat also sets up the C++ MSVC project solution located at scripts/windows/CauchyWindows/CauchyWindows.sln which can be used to compile, debug, and run src/cauchy_estimator.cpp

> Python Build:
>> Linux/Mac: The shell script scripts/swig/cauchy/swigit_unix.sh, which uses swig to build the C++ backend for the Python MCE module scripts/swig/cauchy/cauchy_estimator.py

>> Windows: The batch file scripts/swig/cauchy/swigit_windows.bat, which uses swig to build the C++ backend for the Python MCE module scripts/swig/cauchy/cauchy_estimator.py

> Matlab Build:
>> Linux/Mac/Windows: The matlab file matlab/mex_files/build.m, which uses mex to build the C++ backend for the MCE modules matlab/matlab_pure/MCauchyEstimator.m and matlab/matlab_pure/MSlidingWindowManager.m

### C++ Build Manual Configuration
> Linux: In Makefile, change variable CC=g++ (if not already set). Run make clean && make cauchy window D=0

> Mac: In Makefile, change variable CC=clang++ (if not already set). Run make clean && make cauchy window D=0

> Windows: In the batch file win_cpp_make.bat, ammend the paths of the variables INC_MSVC through LIB_UCRT to your system paths. Run .\win_cpp_make.bat or click on this batch file in finder.

### Python Build Manual Configuration
> Linux: In scripts/swig/cauchy/swigit_unix.sh, ammend the paths of the variables INC_PYTHON, LIB_PYTHON, INC_NUMPY to your system paths. Change swig executable path if needed (line 30). Change compiler to g++ (lines 35, 40), if not already set. Run the script as ./swigit_unix.sh

> Mac: In scripts/swig/cauchy/swigit_unix.sh, ammend the paths of the variables INC_PYTHON, LIB_PYTHON, INC_NUMPY to your system paths. Change swig executable path if needed (line 30). Change compiler to clang++ (lines 35, 40), if not already set. Run the script as ./swigit_unix.sh

> Windows: In the batch file scripts/swig/cauchy/swigit_windows.bat, ammend the paths of the variables MY_EXE through LIB_MSVC (lines 18-35) to your system paths. Run .\win_cpp_make.bat or click on this batch file in finder.

### Matlab Build Manual Configuration
> Linux: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath to your system paths. Open the matlab GUI and run matlab/mex_files/build.m

> Mac: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath to your system paths. Note you must have Xcode installed. Open the matlab GUI and run matlab/mex_files/build.m

> Windows: In matlab/mex_files/build.m, amend the paths for the variables includePath and libraryPath to your system paths. Note you must have a Microsoft Visual Studio version installed. Open the matlab GUI and run matlab/mex_files/build.m

## Software Restrictions and License Note:
See License. The software is free and open to use by anyone for non-commercial purposes. For commercial licensure, contact natsnyder1@gmail.com or speyer@g.ucla.edu.
