import os, sys, platform
import subprocess, requests, shutil, sysconfig
import distutils.sysconfig as dusc
import tarfile
np = None # numpy handle

RED_START = "\033[31m"
RED_END = "\033[0m"
GREEN_START = "\033[32m"
GREEN_END = "\033[0m"
YELLOW_START="\033[93m"
YELLOW_END="\033[0m"


# Naming conventions for eveything below may need to change for windows...certainly paths
def get_python_version(major = False):
    version_info = sys.version
    py_version = version_info.split(' ')[0]
    if major:
        py_version_pieces = py_version.split(".")
        return py_version_pieces[0] + "." + py_version_pieces[1]
    else:
        return py_version

def get_operating_sys():
    os_name = platform.system()
    if os_name == "Darwin":
        return "mac"
    elif os_name == "Windows":
        return "windows"
    elif os_name == "Linux":
        return "linux"
    else:
        print(RED_START+"[ERROR get_operating_sys:] the operatingg system {} does not match Darwin (macOS), Windows, or Linux. Please Debug Here!".format(os_name)+RED_END)
        exit(1)

def is_numpy_installed():
    try:
        global np
        import numpy as _np
        np = _np
        print(GREEN_START+"numpy version: {} is installed. Continuing...".format(np.__version__)+GREEN_END)
        return True
    except ImportError:
        return False

def get_numpy_include_path():
    return np.get_include()

def get_numpy_version(maj_only):
    if maj_only:
        return np.__version__.rsplit('.', 1)[0]
    else:
        return np.__version__

def get_python_include_path():
    return dusc.get_python_inc()

def get_python_lib_path(os_type):
    if os_type == 'linux':
        libpython_path = sysconfig.get_config_var('LIBDIR')
        if not libpython_path:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_path could not be found. Exiting!"+RED_END)
            exit(1)
        libpython_so_name = sysconfig.get_config_var('INSTSONAME')
        if not libpython_so_name:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name could not be found. Exiting!"+RED_END)
            exit(1)
        if ".so" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.so')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        elif ".a" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.a')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        else:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name extension {} not (.a,.so,.dylib). Debug here. Exiting!"+RED_END)
            exit(1)
        return libpython_path, libpython_so_name, libpython_so_tag
    else:
        libpython_path = sysconfig.get_config_var('LIBDIR')
        if not libpython_path:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_path could not be found. Exiting!"+RED_END)
            exit(1)
        # walk through directory and store shared lib versions
        shr_libs = [] 
        for dirpath, dirnames, filenames in os.walk(libpython_path):
            for filename in filenames:
                if (".dylib" in filename) and ("libpython{}".format(get_python_version(major=True)) in filename):
                    shr_libs.append(filename)
                if (".so" in filename) and ("libpython{}".format(get_python_version(major=True)) in filename):
                    shr_libs.append(filename)
            break 
        # Form full path for selected shr_lib 
        libpython_so_name = shr_libs[0]
        #libpython_path += "/{}".format(libpython_so_name)
        if not libpython_so_name:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name could not be found. Exiting!"+RED_END)
            exit(1)
        if ".so" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.so')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        elif ".dylib" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.dylib')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        elif ".a" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.a')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        else:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name extension {} not (.a,.so,.dylib). Debug here. Exiting!"+RED_END)
            exit(1)
        return libpython_path, libpython_so_name, libpython_so_tag

def download_numpy_i_file(url, local_filepath):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a file
        with open(local_filepath, 'wb') as file:
            file.write(response.content)
        print("File numpy.i downloaded successfully!")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        exit(1)

def get_auto_config_path():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return file_dir

def download_file(url, local_filename):
    # Send a GET request to the URL
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for HTTP errors
        # Write the response content to a local file
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return True

# Path to your .m file
def run_matlab_script(matlab_script):
    os_name = get_operating_sys()
    # Command to run MATLAB with the .m file
    command = ['matlab', '-batch', f"run('{matlab_script}')"]
    # Run the MATLAB script
    try:
        result = subprocess.run(command, check=True)
        # Check the return code to see if the script executed successfully
        if result.returncode == 0:
            print("MATLAB script executed successfully!")
            return 0
        else:
            print(f"MATLAB script failed with return code {result.returncode}")
            return 1
    except Exception:
        print(YELLOW_START+"The matlab command is not found!"+YELLOW_END)
        print(YELLOW_START+"Attempting to auto-locate it..."+YELLOW_END)
        # Lets try and find it automatically
        app_dir = "/Applications" if os_name == "mac" else "/usr/opt..."
        matlab_versions = []
        for dirpath, dirnames, filenames in os.walk(app_dir):
            for dirname in dirnames:
                fn = dirname.lower()
                if ("matlab" in fn) and (".app" in fn):
                    matlab_versions.append(dirname)
                    print(GREEN_START+"  Found matlab version {}".format(dirname) + GREEN_END)
            break
        if len(matlab_versions) == 0:
            print(RED_START+"Cannot auto-locate matlab! Not seen in Applications directory..."+RED_END)
            # Ask user to specify the matlab executable folder
            while True:
                response = input("Specify the path to the matlab executable below (for example, /YOUR/PATH/TO/MATLAB_2022a/bin/matlab), otherwise enter q to quit and see the simple manual Matlab Wrapper installation steps.\nEnter path or quit:")
                if response == 'q':
                    return 1
                elif "/" in response:
                     command = [matlab_exec_path, '-batch', f"run('{matlab_script}')"]
                     break
                else:
                    print("Invalid input:")
        elif len(matlab_versions) == 1:
            matlab_exec_path = "{}/{}/{}/matlab".format(app_dir, matlab_versions[0], "bin")
            print("Using Matlab executable: {}".format(matlab_exec_path))
            command = [matlab_exec_path, '-batch', f"run('{matlab_script}')"]
            
        elif len(matlab_versions) > 1:
            print("Located the following matlab versions: ")
            for i,mv in enumerate(matlab_versions):
                print("{}.) {}".format(i+1, mv))
            while True:
                response = int(input("Enter {} to {} to select the corresponding Matlab version above:".format(1,len(matlab_versions))) )
                if (response >= 1) and (response <= len(matlab_versions)):
                    matlab_version = matlab_versions[response-1]
                    break 
                else:
                    print("Invalid input: Enter the number {} to {} corresponding to your Matlab version above".format(1,len(matlab_version)))
            matlab_exec_path = "{}/{}/{}/matlab".format(app_dir, matlab_version, "bin")
            command = [matlab_exec_path, '-batch', f"run('{matlab_script}')"]
        # Try subprocess call again
        try:
            print(YELLOW_START+"Running Matlab Wrapper Build Script..."+YELLOW_END)
            result = subprocess.run(command, check=True)
            # Check the return code to see if the script executed successfully
            if result.returncode == 0:
                print(GREEN_START+"MATLAB script executed successfully!"+GREEN_END)
                return 0
            else:
                print(RED_START+"MATLAB script failed with return code {}".format(result.returncode)+RED_END)
                return 1
        except Exception:
            print(RED_START+"Matlab script did not compile correctly!"+RED_END)
            return 0

def unix_setup_c_examples():
    print(GREEN_START+"--- Building Cauchy Estimator C++ Examples ---"+GREEN_END)
    os_name = get_operating_sys()
    cwd = os.getcwd()
    auto_config_path = get_auto_config_path()
    os.chdir(auto_config_path)
    c_examples_src_path = auto_config_path + "/src"
    c_examples_bin_path =  auto_config_path + "/bin"
    # Change g++ to clang++ for mac and clang++ to g++ for linux
    count = 0
    make_path = auto_config_path + "/Makefile" 
    with open(make_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
    for i in range(num_lines):
        if os_name == "mac":
            lines[i] = lines[i].replace("CC=g++","CC=clang++") #overkill but quick
        else:
            lines[i] = lines[i].replace("CC=clang++","CC=g++") #overkill but quick
    with open(make_path, 'w') as handle:
        handle.writelines(lines)
    print(YELLOW_START+"Running: make clean"+YELLOW_END)
    result = subprocess.run(["make", "clean"], check=True)
    print(YELLOW_START+"Running: make all D=0"+YELLOW_END)
    result = subprocess.run(["make", "all", "D=0"], check=True)
    os.chdir(cwd)
    if result.returncode == 0:
        print(GREEN_START+"C++ examples from\n {} have build successful in\n {}".format(c_examples_src_path, c_examples_bin_path)+GREEN_END)
    else:
        print(RED_END+"C++ examples did not compile successfully! Please look at print out above and assess problem." + RED_END)
        exit(1)       
    print(GREEN_START+"-----------------------------------------"+GREEN_END) 

def unix_setup_python_wrapper():
    cwd = os.getcwd()
    os_name = get_operating_sys()
    assert(os_name in "mac linux")
    # Check Python Version >= 3
    py_ver = get_python_version()
    py_ver_maj = py_ver[:3]
    if int(py_ver[0]) < 3:
        print(RED_START+"[ERROR unix_setup_python_wrapper:] Python version {} is less than version 3...This is untested, and a very deprectated python version (Maybe it will work? Comment this block if you want to try). To prevent possible unforseen errors, use python >= 3.0.0 ... Exiting!" + RED_END)
        exit(1)
    # Check Numpy Installed
    if not is_numpy_installed():
        print(RED_START+"[ERROR unix_setup_python_wrapper:] Numpy is not installed. Please install numpy through your package manager...i.e., pip3 install numpy" + RED_END)
        exit(1)
    # Get path to this file 
    auto_config_path = get_auto_config_path()
    # Find Python include and lib paths
    py_include_path = get_python_include_path()
    py_lib_path, py_lib_so_name, py_lib_so_tag = get_python_lib_path(os_name)
    # Get Numpy include path 
    np_include_path = get_numpy_include_path() 
    print(GREEN_START+"--- Auto Configuration Script Found The Following Paths: ---"+GREEN_END)
    print("  Auto Config Path: ", auto_config_path)
    print("  Python Include Path: ", py_include_path)
    print("  Python Lib Path: ", py_lib_path)
    print("  Python Shared Object Name: ", py_lib_so_name)
    print("  Python Shared Object Tag: ", py_lib_so_tag)
    print("  Numpy Include Path: ", np_include_path)
    print(GREEN_START+"---------------------------------------------------"+GREEN_END)

    # Configure swigit_unix.sh
    swigit_file = "swigit_unix.sh"
    swigit_path = auto_config_path + "/scripts/swig/cauchy/" + swigit_file
    print(YELLOW_START+"Configuring Swig Install File:\n Located: {}".format(swigit_path)+YELLOW_END)
    with open(swigit_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
        # Change INC_PYTHON
        count = 0
        while "INC_PYTHON" != lines[count][0:10]:
            count += 1
            if count == num_lines:
                print(RED_START+"[ERROR unix_setup_python_wrapper:] INC_PYTHON variable could not be found in swigit_{}.sh...indicating file corruption...please redownload this file and rerun auto_config.py...Exiting!".format(os_name) + RED_END)
                exit(1)
        py_include_swigit = "INC_PYTHON=-I" + "\"" + py_include_path + "\"" + "\n"
        lines[count] = py_include_swigit
        # Change LIB_PYTHON
        count = 0
        while "LIB_PYTHON" != lines[count][0:10]:
            count += 1
            if count == num_lines:
                print(RED_START+"[ERROR unix_setup_python_wrapper:] LIB_PYTHON variable could not be found in swigit_{}.sh...indicating file corruption...please redownload this file and rerun auto_config.py...Exiting!".format(os_name) + RED_END)
                exit(1)
        py_lib_swigit = "LIB_PYTHON=-L" + "\"" + py_lib_path + " " + py_lib_so_tag + "\"" + "\n"
        lines[count] = py_lib_swigit
        # Change INC_NUMPY
        count = 0
        while "INC_NUMPY" != lines[count][0:9]:
            count += 1
            if count == num_lines:
                print(RED_START+"[ERROR unix_setup_python_wrapper:] INC_NUMPY variable could not be found in swigit_{}.sh...indicating file corruption...please redownload this file and rerun auto_config.py...Exiting!".format(os_name) + RED_END)
                exit(1)
        np_include_swigit = "INC_NUMPY=-I" + "\"" + np_include_path + "\"" + "\n"
        lines[count] = np_include_swigit
        # Change g++ line 
        for i in range(num_lines):
            if os_name == "mac":
                if lines[i][0:3] == "g++":
                        lines[i] = lines[i].replace("g++", "clang++") 
                lines[i] = lines[i].replace("-shared", "-dynamiclib")
            else: # linux
                if lines[i][0:7] == "clang++":
                        lines[i] = lines[i].replace("clang++", "g++") 
                lines[i] = lines[i].replace("-dynamiclib", "-shared")
    # All done -- Write Lines
    try:
        with open(swigit_path, 'w') as handle:
            handle.writelines(lines)
    except Exception:
        print(RED_START+"Could not write to {}! System permissive settings limited! Please fix write capabilities and try again! Exiting!".format(swigit_path) + RED_END)
        exit(1)
    print(GREEN_START+"Updated\n {}\nwith new paths!".format(swigit_path))

    # Download the correct numpy file matching their numpy version 
    numpy_maj_version = get_numpy_version(maj_only=True)
    numpy_i_local_path = auto_config_path + "/scripts/swig/cauchy/numpy_{}.i".format(numpy_maj_version)
    numpy_i_local_default = auto_config_path + "/scripts/swig/cauchy/numpy.i"
    print(GREEN_START+"--- Auto Configuration Script Downloading numpy.i to:\n {}\n For numpy version={}: ---".format(swigit_path, numpy_maj_version) + GREEN_END)
    # If not downloaded, download and then softlink as numpy.i
    if not os.path.isfile(numpy_i_local_path):
        url_numpy_i = "https://raw.githubusercontent.com/numpy/numpy/maintenance/{}.x/tools/swig/numpy.i".format(numpy_maj_version)
        download_file(url_numpy_i, numpy_i_local_path)
    # Regardless, just re-symlink 
    if os.path.isfile(numpy_i_local_default):
        os.remove(numpy_i_local_default)
    os.symlink(numpy_i_local_path, numpy_i_local_default)
    print(GREEN_START+"---------------------------------------------------"+GREEN_END)

    # swigit_unix.sh configured -- now download swig+pcre versions
    swig_local_dir = auto_config_path + "/scripts/swig/swig_download"
    print(GREEN_START + "--- Auto Configuration Script Downloading swig, pcre and installing... ---" + GREEN_END)
    if not os.path.isdir(swig_local_dir):
        os.mkdir(swig_local_dir)
    swig_tar_path = auto_config_path + "/scripts/swig/swig_download/swig-4.1.0.tar.gz"
    if not os.path.isfile(swig_tar_path):
        url_swig = "https://sourceforge.net/projects/swig/files/swig/swig-4.1.0/swig-4.1.0.tar.gz/download"
        download_file(url_swig, swig_tar_path)
    pcre_tar_name = "pcre2-10.44.tar.gz"
    pcre_tar_path = auto_config_path + "/scripts/swig/swig_download/" + pcre_tar_name
    if not os.path.isfile(pcre_tar_path):
        url_pcre = "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.44/pcre2-10.44.tar.gz"
        download_file(url_pcre, pcre_tar_path)
    # Extract Swig
    swig_untar_path = auto_config_path + "/scripts/swig/swig_download/swig-4.1.0"
    swig_install_dir = auto_config_path + "/scripts/swig/swig_download/install_swig"
    if not os.path.isdir(swig_untar_path):
        print("Installing and configuring swig...")
        # create swig's install folder
        if not os.path.isdir(swig_install_dir):
            os.mkdir(swig_install_dir)
        # Untar 
        with tarfile.open(swig_tar_path, 'r:gz') as tar:
            tar.extractall(path=swig_local_dir)
        os.chdir(swig_untar_path)
        # copy pcre into untar swig
        pcre_mv_tar_path = swig_untar_path + "/{}".format(pcre_tar_name)
        shutil.copyfile(pcre_tar_path, pcre_mv_tar_path)
        
        # Call the swig-4.1.0/Tools/pcre-build.sh script
        swig_pcre_helper_path = swig_untar_path + "/Tools/pcre-build.sh"
        result = subprocess.run([swig_pcre_helper_path], check=True)
        if result.returncode == 0:
            print("Swig's pcre helper script has executed successfully!")
        # Call swig's ./configure.sh script with install prefix
        swig_config_path = swig_untar_path + "/configure"
        result = subprocess.run([swig_config_path] + ["--prefix="+swig_install_dir], check=True)
        if result.returncode == 0:
            print("Swig's configure script has executed successfully!")
        # Call make, make install
        result = subprocess.run(["make"], check=True)
        if result.returncode == 0:
            print("make command for swig has executed successfully!")
        result = subprocess.run(["make", "install"], check=True)
        if result.returncode == 0:
            print("make install command for swig executed successfully!")
    # build path to swig executable
    swig_exec = swig_install_dir + "/bin/swig"
    # write out executable path to swigit_unix.sh
    with open(swigit_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
    # Change swig executable location
    count = 0
    while True:
        if "swig -c++ -python" in lines[count]:
            if lines[count][0:8] != "[ERROR:]":
                break
        count += 1
        if count == num_lines:
            print(RED_START+"[ERROR unix_setup_python_wrapper:] swig executable could not be found in swigit_{}.sh...indicating file corruption...please redownload this file and rerun auto_config.py...Exiting!".format(os_name) + RED_END)
            exit(1)
    swig_exec_swigit = swig_exec + " -c++ -python ${SWIG_FILE}\n"
    lines[count] = swig_exec_swigit
    with open(swigit_path, 'w') as handle:
        handle.writelines(lines)
    
    # Call swigit_unix.sh
    swigit_run_path = auto_config_path + "/scripts/swig/cauchy/"
    os.chdir(swigit_run_path)
    result = subprocess.run([swigit_path], check=True)
    os.chdir(cwd)
    if result.returncode == 0:
        print(GREEN_START+"swigit_unix.sh executed successfully!"+GREEN_END)
    else:
        print(RED_START + "swigit_unix.sh did not execute successfully!" + RED_END)
        return
    print(GREEN_START+"---------------------------------------------------"+GREEN_END)
    pycauchy_ce_path = auto_config_path + "/scripts/swig/cauchy/cauchy_estimator.py"
    pycauchy_tut1_path = auto_config_path + "/scripts/tutorial/lit_systems.ipynb"
    pycauchy_tut2_path = auto_config_path + "/scripts/tutorial/nonlin_systems.ipynb"
    print(GREEN_START+"Python Wrapper build script completed:\nThe module:\n {}\nCan be included in your projects. Checkout the tutorials:\n {}\n {}\nto see examples".format(pycauchy_ce_path, pycauchy_tut1_path, pycauchy_tut2_path) +GREEN_END)

def unix_setup_matlab_wrapper():
    # Get path to this file 
    os_name = get_operating_sys()
    assert(os_name in "mac linux")
    auto_config_path = get_auto_config_path()
    swig_cauchy_include_path = "-I" + auto_config_path + "/scripts/swig/cauchy" 
    swig_cauchy_include_path += " -I" + auto_config_path + "/include"
    matlab_build_path = auto_config_path + "/matlab/mex_files/build.m"
    print(GREEN_START + "--- Running Matlab Wrapper Configuration Script ---" + GREEN_END)
    with open(matlab_build_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
    # Change the include path, but also change deletion file paths based off of mac / linux type 
    include_path_line = -1
    for i in range(num_lines):
        if "includePath" == lines[i][0:11]:
            include_path_line = i
        else:
            if os_name == "linux":
                lines[i] = lines[i].replace("mexmaca", "mexa")
            else:
                lines[i] = lines[i].replace("mexa", "mexmaca")
    if include_path_line == -1:
        print(RED_START+"[ERROR def unix_setup_matlab_wrapper:] cannot find includePath in file {}...this indicates file corruption. Please replace this file with that found on our github. Exiting!" + RED_END)
        exit(1)
    lines[include_path_line] = "includePath = \'" + swig_cauchy_include_path + "\';\n"
    with open(matlab_build_path, 'w') as handle:
        handle.writelines(lines)
    print("Wrote updated contents to: ", matlab_build_path, "\ncalling matlab mex script!...")
    # Call matlab from python 
    matlab_tut1_path = auto_config_path + "/matlab/lti_systems.mlx"
    matlab_tut2_path = auto_config_path + "/matlab/nonlin_systems.mlx"
    matlab_mcauchy1_path = auto_config_path + "/matlab/matlab_pure/MCauchyEstimator.m"
    matlab_mcauchy2_path = auto_config_path + "/matlab/matlab_pure/MSlidingWindowManager.m"
    if run_matlab_script(matlab_build_path) == 0:
        print("Matlab build script completed: The modules:\n {}\n {}\nCan be included in your projects. Checkout the tutorials:\n {}\n {}\nto see examples".format(matlab_mcauchy1_path, matlab_mcauchy2_path, matlab_tut1_path, matlab_tut2_path) )
    # If unsuccessful to call matlab: 
    else:
        print(RED_START+"[Error unix_setup_matlab_wrapper:] Matlab executable not found from command line: you could add the matlab executable path to your PATH, or follow the below instructions:"+RED_END)
        print(YELLOW_START+"  1.) Open this workspace in the Matlab GUI and navigate to matlab/mex_files"+YELLOW_END)
        print(YELLOW_START+"  2.) Run the build.m file (i.e, type the word build) in the matlab command window", YELLOW_END)
        print(YELLOW_START+"The build process will generate the modules:\n {}\n {}\n and can be included in your projects... Checkout the tutorials:\n {}\n {}\nto see examples".format(matlab_mcauchy1_path, matlab_mcauchy2_path, matlab_tut1_path, matlab_tut2_path) + YELLOW_END)
    print(GREEN_START + "----------------------------------------" + GREEN_END)

def windows_setup_c_examples():
    assert(False)

def windows_setup_python_wrapper():
    assert(False)

def windows_setup_matlab_wrapper():
    assert(False)

if __name__ == '__main__':
    # Build C++ Examples
    while True:
        run_cpp = input("Would you like to compile the C++ build examples?\nThis is a good first step, as the Python/Matlab bindings will not work unless the C-code can compile:\nTo build C++ Examples, Enter: 'y' for yes or 'n' for no...")
        run_cpp = run_cpp.lower()
        if run_cpp in ['y', 'n']:
            break
        else:
            print("Unknown input {}, Enter y for yes or n for no. Try again!")
    if run_cpp == 'y' and (get_operating_sys() != "windows"): 
        unix_setup_c_examples()
    elif run_cpp == 'y' and (get_operating_sys() == "windows"):
        windows_setup_c_examples()
    
    # Build Python Wrapper
    while True:
        run_py = input("This script can also build the Python wrapper:\nWould you like to build it? Enter 'y' for yes or 'n' for no...")
        run_py = run_py.lower()
        if run_py in ['y', 'n']:
            break
        else:
            print("Unknown input {}, Enter y for yes or n for no. Try again!")
    if (run_py == 'y') and (get_operating_sys() != "windows"):
        unix_setup_python_wrapper()
    if (run_py == 'y') and (get_operating_sys() == "windows"):
        windows_setup_python_wrapper()
    
    # Build Matlab Wrapper
    while True:
        run_ml = input("\nWould you like to build the Matlab Wrapper? Enter 'y' for yes or 'n' for no...")
        run_ml = run_ml.lower()
        if run_ml in ['y', 'n']:
            break
        else:
            print("Unknown input {}, Enter y for yes or n for no. Try again!")
    if (run_ml == 'y') and (get_operating_sys() != "windows"):
        unix_setup_matlab_wrapper()
    if (run_ml == 'y') and (get_operating_sys() == "windows"):
        windows_setup_matlab_wrapper()
