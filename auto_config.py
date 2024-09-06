import os, sys, platform
import subprocess, shutil, sysconfig
import distutils.sysconfig as dusc
import tarfile, zipfile, importlib

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
    elif "cygwin" in os_name.lower():
        return "windows"
    else:
        print(RED_START+"[ERROR get_operating_sys:] the operatingg system {} does not match Darwin (macOS), Windows, or Linux. Please Debug Here!".format(os_name)+RED_END)
        exit(1)

def get_python_exe_name():
    pyver = get_python_version(True)
    pyver_test = ["python{}".format(pyver), "python{}".format(pyver.replace(".", "")), "python3", "python"]
    pyver_exe = None
    for pvt in pyver_test:
        try:
            result = subprocess.run([pvt, "--version"], check=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Access the output from stdout
                version_output = result.stdout
                # Check whether this matches the pyver above 
                if pyver in version_output:
                    pyver_exe = pvt
            else:
                pass
        except Exception:
            pass
    return pyver_exe

def get_auto_config_path():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    return file_dir

# Requests library
try:
    import requests #,numpy, matplotlib, scipy, urllib3
except Exception:
    _os_name = get_operating_sys()
    print(RED_START+"Python requests module is not installed!"+RED_END)
    print(GREEN_START+"Would you like to install this module? Enter y or n for yes or no"+GREEN_END)
    requests_installed = False
    while True:
        response = input("y (yes, install) or n (no dont auto-install):")
        if response == 'y':
            pyexe_tag = get_python_exe_name()
            if pyexe_tag is not None:
                # To install all of requirements.txt
                #if _os_name == "windows":
                #    path2reqs = get_auto_config_path() + "\\scripts\\requirements.txt"
                #else:
                #    path2reqs = get_auto_config_path() + "/scripts/requirements.txt"
                #result = subprocess.run([pyexe_tag, "-m", "pip", "install", "-r", path2reqs])
                # To install only requests
                result = subprocess.run([pyexe_tag, "-m", "pip", "install", "requests"])
                if result.returncode == 0:
                    print(GREEN_START+"Pip has run, see pip output above, seems to have been installed..."+GREEN_END)
                    importlib.invalidate_caches()  # Invalidate any cache in case pip installed requests
                    requests = importlib.import_module("requests")  # Re-import requests module
                    requests_installed = True
                else:
                    print(YELLOW_START+"Pip has run, but something may have gone wrong, see above pip output."+YELLOW_END)
                    requests_installed = False
            else:
                print(YELLOW_START+"Cannot determine python exe name! You will need to install requests on your own"+YELLOW_END)
                print(YELLOW_START+"...you can do so with:"+YELLOW_END)
            break
        elif response == 'n':
            print(YELLOW_START+"Not auto-installing...you can do so with:"+YELLOW_END)
            break
        else:
            print("Unknown response, please re-enter!")
    if not requests_installed:
        print(YELLOW_START+"  -> YOUR_PYTHON_VERSION -m pip install requests"+YELLOW_END)
        print(YELLOW_START+"Moreover, you could install all Python requirements for this repository using:"+YELLOW_END)
        print(YELLOW_START+"  -> YOUR_PYTHON_VERSION -m pip install -r scripts{}requirements.txt".format("\\" if _os_name == "windows" else "/") +YELLOW_END)
        print(YELLOW_START+"If any other modules above cannot be imported, please repeat this process for them."+YELLOW_END)
        print(YELLOW_START+"If using a Python package manager other than pip, please see external installation instructions for installating the requests module..."+YELLOW_END)
        print(RED_START+"Exiting!"+RED_END)
        exit(1)
import requests
        
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
    elif os_type == 'mac':
        libpython_path = sysconfig.get_config_var('LIBDIR')
        if not os.path.isdir(libpython_path):
            libpython_path = sysconfig.get_paths()["stdlib"]
            if libpython_path[-3:] != "lib":
                libpython_path = libpython_path.rsplit("/", 1)[0]
            if not os.path.isdir(libpython_path):
                print(RED_START+"Cannot Locate libpython{}.dylib or a closely named library. Please debug here! Exiting!"+RED_END)
                exit(1)
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
    else: # windows
        python_dir_path = get_python_include_path()
        if "include" == python_dir_path[-7:]:
            python_dir_path = python_dir_path.rsplit('\\', 1)[0]
        libpython_path = python_dir_path + "\\libs"
        if not os.path.isdir(libpython_path):
            libpython_path = python_dir_path
        libpython_so_name = None
        for dirpath, dirnames, filenames in os.walk(libpython_path):
            for filename in filenames:
                if ("python" in filename) and ("lib" in filename):
                    libpython_so_name = filename
                    break
            break        
        if not libpython_so_name:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name could not be found. Exiting!"+RED_END)
            exit(1)
        if ".so" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.so')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        elif ".a" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.a')[0]
            libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
        elif ".lib" in libpython_so_name:
            libpython_so_tag_prefix = libpython_so_name.split('.lib')[0]
            if libpython_so_tag_prefix[3:] == "lib":
                libpython_so_tag = '-l' + libpython_so_tag_prefix[3:]
            else:
                libpython_so_tag = '-l' + libpython_so_tag_prefix
        else:
            print(RED_START+"[ERROR get_python_lib_path:] libpython_so_name extension {} not (.a,.so,.dylib). Debug here. Exiting!"+RED_END)
            exit(1)
        return libpython_path, libpython_so_name, libpython_so_tag

def download_file(url, local_filename, allow_redirects=True):
    # Send a GET request to the URL
    with requests.get(url, stream=True, allow_redirects=allow_redirects) as r:
        r.raise_for_status()  # Check for HTTP errors
        # Write the response content to a local file
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return True

def walk_until(root_dir, file_name, dir_name, dir_filters=None):
    # assert only file name or dir name is entered
    if not( bool(file_name) ^ bool(dir_name) ):
        print("walk_until: only file_name or dir_name can be given, not both!")
        exit(1)
    # from start, walk
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check for file 
        if file_name is not None:
            if file_name in filenames:
                return dirpath + "\\" + file_name
        # check for dir
        if dir_name is not None:
            if dir_name in dirnames:
                return dirpath + "\\" + dir_name
        # check dir filters
        if dir_filters is not None:
            for df in dir_filters:
                if df in dirnames:
                    dir_filters_new = list(dir_filters)
                    dir_filters_new.remove(df)
                    ret = walk_until(dirpath+"\\"+df, file_name, dir_name, dir_filters_new)
                    if ret:
                        return ret
    return None           

# Finds file_name or dir_name making sure dir_filters are in the path
def walk_all_filtered(root_dir, file_name, dir_name, dir_filters, results = []):
    os_name = get_operating_sys()
    # assert only file name or dir name is entered
    if not( bool(file_name) ^ bool(dir_name) ):
        print("walk_until: only file_name or dir_name can be given, not both!")
        exit(1)
    # from start, walk
    used_dir_filters = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check results against dir_filters
        cont_loop = False
        for df in used_dir_filters:
            if os_name == "windows":
                if "\\{}".format(df) in dirpath:
                    cont_loop = True
                    break
            else:
                if "/{}".format(df) in dirpath:
                    cont_loop = True
                    break
        if cont_loop:
            continue
        # Check for file 
        if file_name is not None:
            if file_name in filenames:
                result = dirpath + "\\" + file_name
                is_append = True
                for df in dir_filters:
                    if os_name == "windows":
                        if "\\{}".format(df) not in result:
                            is_append = False
                    else:
                        if "/{}".format(df) not in result:
                            is_append = False
                if is_append:
                    results.append(result)
                #return results
        # check for dir
        if dir_name is not None:
            if dir_name in dirnames:
                result = dirpath + "\\" + dir_name
                is_append = True
                for df in dir_filters:
                    if os_name == "windows":
                        if "\\{}".format(df) not in result:
                            is_append = False
                    else:
                        if "/{}".format(df) not in result:
                            is_append = False
                if is_append:
                    results.append(result)
                #return results
        # check dir filters
        if dir_filters is not None:
            for df in dir_filters:
                if df in dirnames:
                    dir_filters_new = list(dir_filters)
                    dir_filters_new.remove(df)
                    if df not in used_dir_filters:
                        used_dir_filters.append(df)
                    results = walk_all_filtered(dirpath+"\\"+df, file_name, dir_name, dir_filters_new, results)
    return results     

# Walks down the provided root directory to find file_name or dir_name
def walk_all_unfiltered(root_dir, file_name, dir_name):
    os_name = get_operating_sys()
    # assert only file name or dir name is entered
    if not( bool(file_name) ^ bool(dir_name) ):
        print("walk_until: only file_name or dir_name can be given, not both!")
        exit(1)
    # from start, walk
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check for file 
        if file_name is not None:
            if file_name in filenames:
                result = dirpath + "\\" + file_name
                results.append(result)
                #return results
        # check for dir
        if dir_name is not None:
            if dir_name in dirnames:
                result = dirpath + "\\" + dir_name
                results.append(result)
                #return results
    return results
  
def win_reponse_or_exit():
    response = input("Enter path or q: ")
    while True:
        if response == 'q':
            print(RED_START+"Exiting! Goodbye!"+RED_END)     
            exit(1)
        elif "C:" in response:
            if os.path.isdir(response):
                print(GREEN_START+"Located {} ... Continuing!".format(response)+GREEN_END)
                return True
            else:
                print(YELLOW_START+"Path {} does not exist! Please recheck path!"+YELLOW_END)
                #return False
        else:
            print(RED_START+"Undefined Choice "+RED_END)

def win_get_ucrt_shared_um_paths(wk_root_dir, exit_early = False):
    wk_versions = []
    # Get the possible version numbers and from which determine if they have the um shared and ucrt sub-directories 
    for dirpath, dirnames, filenames in os.walk(wk_root_dir):
        for dirname in dirnames:
            try:
                wk_versions.append( float(dirname) )
            except Exception:
                pass
        break
    # Cannot find Win Kit Versions
    if len(wk_versions) == 0:
        if exit_early:
            return None, None, None
        else:
            print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find Versions of Windows Kits:\n  1.) please manually enter PATH\\TO\\Windows Kits\\YOUR_VERSION,\n  2.) enter q to quit script" + YELLOW_END)
            wk_versions = [win_reponse_or_exit()]
    # Look for um shared and ucrt
    wk_options = []
    for wkv in wk_versions:
        if (wkv % 1.0) == 0:
            wkv = int(wkv)
        wk_query = wk_root_dir + "\\" + str(wkv)
        include_dir_um = walk_until(wk_query, None, "um", ["Include"])
        if not include_dir_um:
            continue 
        include_dir_shr = include_dir_um[:include_dir_um.find("\\um")] + "\\shared"
        include_dir_ucrt = include_dir_um[:include_dir_um.find("\\um")] + "\\ucrt"
        if not os.path.isdir(include_dir_shr):
            include_dir_shr = None 
            continue
        if not os.path.isdir(include_dir_ucrt):
            include_dir_ucrt = None 
            continue
        wk_options.append( (wkv,include_dir_um, include_dir_shr, include_dir_ucrt))
    if len(wk_options) == 0:
        if exit_early:
            return None, None, None
        else:
            print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find Versions of Windows Kits which contain headers directorys 'um' 'shared' and 'ucrt':\n  1.) please manually enter PATH\\THAT\\CONTAINS -> {UCRT,UM,SHARED},\n  2.) enter q to quit script" + YELLOW_END)
            include_dir_um = win_reponse_or_exit()
        # make sure um/shared/ucrt is not ending
        suffix_idx = include_dir_um.rfind("\\\\")
        if suffix_idx > -1:
            suffix = include_dir_um[suffix_idx+2:]
            if suffix in ["um", "shared", "ucrt"]:
                include_dir_um = include_dir_um[:suffix_idx]
            include_dir_ucrt = include_dir_um + "\\ucrt"
            include_dir_shared = include_dir_um + "\\shared"
            include_dir_um = include_dir_um + "\\um"
            if not os.path.isdir(include_dir_shared):
                print(RED_START +"[error win_get_ucrt_shared_um_paths:]\n {} does not exist! Please check paths and re-follow instructions! Exiting!".format(include_dir_shared)+RED_END)
                exit(1)
            if not os.path.isdir(include_dir_ucrt):
                print(RED_START +"[error win_get_ucrt_shared_um_paths:]\n {} does not exist! Please check paths and re-follow instructions! Exiting!".format(include_dir_ucrt)+RED_END)
                exit(1)
            if not os.path.isdir(include_dir_um):
                print(RED_START +"[error win_get_ucrt_shared_um_paths:]\n {} does not exist! Please check paths and re-follow instructions! Exiting!".format(include_dir_um)+RED_END)
                exit(1)
    else:
        print(GREEN_START+"Found Windows Kits versions:"+GREEN_END)
        for wko in wk_options:
            print("Version: ", wko[0])
        if len(wk_options) == 1:
            include_dir_um = wk_options[0][1]
            include_dir_shr = wk_options[0][2]
            include_dir_ucrt = wk_options[0][3]
        else:
            while True:
                response = input("Enter 1 to {} to indicate the option correpsonding to your version, or q to quit: ".format(len(wk_options)))
                if response == 'q':
                    print(RED_START+"Exiting! Goodbye!"+RED_END)
                    exit(1)
                try:
                    int_resp = int(response)
                    print("Entered Response {} for include headers: {}\n{}\n{}\n".format(int_resp, *wk_options[response][1:]))
                    break
                except Exception:
                    print("{} is not accepted as a valid integer, try again!\n".format(response))
    return include_dir_ucrt, include_dir_shr, include_dir_um

def win_get_line_idx(file_lines, str_query):
    for i, line in enumerate(file_lines):
        if str_query in line:
            return i
    return -1

def win_get_cl_exe():
    cl_exe = walk_until("C:\\Program Files (x86)\\Microsoft Visual Studio", "cl.exe", None, dir_filters=["VC", "Tools", "bin"])
    if cl_exe is None:
        cl_exe = walk_until("C:\\Program Files\\Microsoft Visual Studio", "cl.exe", None, dir_filters=["VC", "Tools", "bin"])
        if cl_exe is None:
            print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find Microsoft Visual Studio Directory:\n  1.) please manually enter full path to cl.exe, which is located in the bin folder of a subdirectory of Microsoft Visual Studio,\n  2.) enter q to quit script" + YELLOW_END)
            cl_exe = win_reponse_or_exit()
    return cl_exe

def win_get_lib_ucrt(include_dir_ucrt):
    _lib_ucrt_path = include_dir_ucrt
    _lib_ucrt_path = _lib_ucrt_path.replace("Include", "Lib")
    lib_ucrt_path = walk_until(_lib_ucrt_path, "libucrt.lib", None, ["x64", "x86"])
    if not os.path.isfile(lib_ucrt_path):
        print(YELLOW_START+"[WARN win_get_lib_ucrt:] Cannot find 'Library path for the library UCRT.\n  1.) Please enter the directory that contains libucrt.lib,\n  2.) enter q to quit script" + YELLOW_END)
        lib_ucrt_path = win_reponse_or_exit()
        if not os.path.isdir(lib_ucrt_path):
            print(RED_START+"Path {} is still not valid! Please check path and restart this script! Exiting!".format(lib_ucrt_path)+RED_END)
            exit(1)
    else:
        lib_ucrt_path = lib_ucrt_path.rsplit("\\", 1)[0]
    return lib_ucrt_path

def win_get_lib_um(include_dir_um):
    _lib_um_path = include_dir_um
    _lib_um_path = _lib_um_path.replace("Include", "Lib")
    lib_um_path = walk_until(_lib_um_path, "Uuid.Lib", None, ["x64", "x86"])
    if not os.path.isfile(lib_um_path):
        print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find 'Library path for the library UM.\n  1.) Please enter the directory that contains libuuid.lib,\n  2.) enter q to quit script" + YELLOW_END)
        lib_um_path = win_reponse_or_exit()
        if not os.path.isdir(lib_um_path):
            print(RED_START+"Path {} is not valid! Please check path and restart this script! Exiting!".format(lib_um_path)+RED_END)
            exit(1)
    else:
        lib_um_path = lib_um_path.rsplit("\\", 1)[0]
    return lib_um_path

def win_get_lib_cpmt(include_dir_mscv):
    _lib_cpmt_path = include_dir_mscv
    _lib_cpmt_path = _lib_cpmt_path.replace("include", "lib")
    lib_cpmt_path = walk_until(_lib_cpmt_path, "libcpmt.lib", None, ["x64", "x86"])
    if not os.path.isfile(lib_cpmt_path):
        print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find 'Library path for the library UM.\n  1.) Please enter the directory that contains libuuid.lib,\n  2.) enter q to quit script" + YELLOW_END)
        lib_cpmt_path = win_reponse_or_exit()
        if not os.path.isdir(lib_cpmt_path):
            print(RED_START+"Path {} is not valid! Please check path and restart this script! Exiting!".format(lib_cpmt_path)+RED_END)
            exit(1)
    else:
        lib_cpmt_path = lib_cpmt_path.rsplit("\\", 1)[0]
    return lib_cpmt_path

def win_get_lib_msvcprt(include_dir_mscv, x64_or_x86):
    _lib_msvcprt_path = include_dir_mscv
    _lib_msvcprt_path = _lib_msvcprt_path.replace("include", "lib")
    lib_msvcprt_path = walk_until(_lib_msvcprt_path, "msvcprt.lib", None, [x64_or_x86])
    if not os.path.isfile(lib_msvcprt_path):
        print(YELLOW_START+"[WARN win_get_lib_msvcprt:] Cannot find 'Library path for the library msvcprt.lib ...\n  1.) Please enter the directory that contains msvcrpt.lib,\n  2.) enter q to quit script" + YELLOW_END)
        lib_msvcprt_path = win_reponse_or_exit()
        if not os.path.isdir(lib_msvcprt_path):
            print(RED_START+"Path {} is not valid! Please check path and restart this script! Exiting!".format(lib_msvcprt_path)+RED_END)
            exit(1)
    else:
        lib_msvcprt_path = lib_msvcprt_path.rsplit("\\", 1)[0]
    return lib_msvcprt_path

# Path to your .m file
def run_matlab_script(matlab_script):
    os_name = get_operating_sys()
    # Command to run MATLAB with the .m file
    print(YELLOW_START+"Running Matlab Wrapper Build Script...this may take a while..."+YELLOW_END)
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
        matlab_versions = []
        if os_name == "mac":
            app_dir = "/Applications"
            for dirpath, dirnames, filenames in os.walk(app_dir):
                for dirname in dirnames:
                    dn = dirname.lower()
                    if ("matlab" in dn) and (".app" in dn):
                        matlab_versions.append(dirname)
                        print(GREEN_START+"  Found matlab version {}".format(dirname) + GREEN_END)
                break
        elif os_name == "linux":
            app_dir = "/usr/opt"
            for dirpath, dirnames, filenames in os.walk(app_dir):
                for dirname in dirnames:
                    dn = dirname.lower()
                    if ("matlab" in dn):
                        matlab_versions.append(dirname)
                        print(GREEN_START+"  Found matlab version {}".format(dirname) + GREEN_END)
        else: # windows
            app_dir1 = "C:\\Program Files (x86)"
            app_dir2 = "C:\\Program Files"
            matlab_exec_path = walk_until(app_dir1, "matlab.exe", None, ["MATLAB", "bin"])
            if matlab_exec_path is not None:
                matlab_versions.append(matlab_exec_path)
            else:
                 matlab_exec_path = walk_until(app_dir2, "matlab.exe", None, ["MATLAB", "bin"])
                 if matlab_exec_path is not None:
                    matlab_versions.append(matlab_exec_path)
                 else:
                     matlab_versions = []
        if len(matlab_versions) == 0:
            print(RED_START+"Cannot auto-locate matlab!..."+RED_END)
            # Ask user to specify the matlab executable folder
            while True:
                if os_name != "windows":
                    response = input("Specify the path to the matlab executable below (for example, /YOUR/PATH/TO/MATLAB_2022a/.../bin/matlab), otherwise enter q to quit and see the simple manual Matlab Wrapper installation steps.\nEnter path or quit:")
                else:
                    response = input("Specify the path to the matlab executable below (for example, C:\\YOUR\\PATH\\TO\\MATLAB\\...\\bin\\matlab.exe), otherwise enter q to quit and see the simple manual Matlab Wrapper installation steps.\nEnter path or quit:")
                if response == 'q':
                    return 1
                elif "/" in response:
                     command = [response, '-batch', f"run('{matlab_script}')"]
                     break
                else:
                    print("Invalid input:")
        elif len(matlab_versions) == 1:
            if os_name == "mac":
                matlab_exec_path = "{}/{}/{}/matlab".format(app_dir, matlab_versions[0], "bin")
            elif os_name == "linux":
                matlab_exec_path = "{}/{}/{}/matlab".format(app_dir, matlab_versions[0], "bin")
            else: #windows
                matlab_exec_path = matlab_versions[0]
                
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
            print(YELLOW_START+"Running Matlab Wrapper Build Script...this may take a while..."+YELLOW_END)
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

def symlink_files(files_dir, files_list, link_dirs):
    print("Sym-linking Cauchy Python Modules:")
    for sym_file in files_list:
        print("  ", sym_file)
    print("Located in: ", files_dir)
    print("To Directories:")
    sym_link_errors = False
    for link_dir in link_dirs:
        print("  ", link_dir)
        for sym_file in files_list:
            src = files_dir + sym_file
            dst = link_dir + sym_file
            # Regardless, just re-symlink
            try:
                os.remove(dst)
            except FileNotFoundError:
                pass 
            except PermissionError:
                print(RED_START+"Permission denied, {} could not be removed".format(dst)+RED_END)
                sym_link_errors = True
            except Exception:
                print(YELLOW_START+"Caught Exception {}, file {} could not be removed".format(Exception, dst)+YELLOW_END)
                sym_link_errors = True
            os.symlink(src, dst)         
    if sym_link_errors:
        print(YELLOW_START+"Symbolic linking errors may have possibly occured. Manually linking or copying the above files may solve your problem!"+YELLOW_END)
        
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
    print(YELLOW_START+"Running: make cauchy window D=0"+YELLOW_END)
    result = subprocess.run(["make", "cauchy", "window", "D=0"], check=True)
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
    try:
        os.remove(numpy_i_local_default)
    except FileNotFoundError:
        pass 
    except PermissionError:
        print("Permission denied, {} could not be removed".format(numpy_i_local_default))
    except Exception:
        pass
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
    print("Linking Cauchy Estimator Python Module to scripts/tutorials, scripts/filter_compare, scripts/leo")
    symlink_files(swigit_run_path, 
        ["pycauchy.py", "_pycauchy.so", "cauchy_estimator.py", "gaussian_filters.py"], 
        [auto_config_path+"/scripts/tutorial/", auto_config_path+"/scripts/swig/filter_compare/", auto_config_path+"/scripts/swig/leo/"])
    
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
        print(GREEN_START + "Matlab build script completed: The modules:\n {}\n {}\nCan be included in your projects. Checkout the tutorials:\n {}\n {}\nto see examples".format(matlab_mcauchy1_path, matlab_mcauchy2_path, matlab_tut1_path, matlab_tut2_path) + GREEN_END)
    # If unsuccessful to call matlab: 
    else:
        print(RED_START+"[Error unix_setup_matlab_wrapper:] Matlab executable not found from command line: you could add the matlab executable path to your PATH, or follow the below instructions:"+RED_END)
        print("  1.) Open this workspace in the Matlab GUI and navigate to matlab/mex_files")
        print("  2.) Run the build.m file (i.e, type the word build) in the matlab command window")
        print("The build process will generate the modules:\n {}\n {}\n and can be included in your projects... Checkout the tutorials:\n {}\n {}\nto see examples".format(matlab_mcauchy1_path, matlab_mcauchy2_path, matlab_tut1_path, matlab_tut2_path))
    print(GREEN_START + "----------------------------------------" + GREEN_END)

def windows_setup_c_examples():
    print(GREEN_START+"--- Building Cauchy Estimator C++ Examples ---"+GREEN_END)
    os_name = get_operating_sys()
    cwd = os.getcwd()
    auto_config_path = get_auto_config_path()
    os.chdir(auto_config_path)
    c_examples_src_path = auto_config_path + "\\src"
    c_examples_bin_path =  auto_config_path + "\\bin"
    # Change g++ to clang++ for mac and clang++ to g++ for linux
    count = 0
    make_path = auto_config_path + "\\win_cpp_make.bat" 
    with open(make_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
        
    # Find the compiler under MSVC
    cl_exe = win_get_cl_exe()
    # Get the MSVC include folder 
    bin_idx = cl_exe.find("\\bin")
    include_dir_mscv = cl_exe[:bin_idx] + "\\include"
    # Now Get the UCRT/SHARED/UM PATH
    include_dir_ucrt = None
    # Check x86 Program files
    if os.path.isdir("C:\\Program Files (x86)\\Windows Kits"):
        include_dir_ucrt, include_dir_shared, include_dir_um = win_get_ucrt_shared_um_paths("C:\\Program Files (x86)\\Windows Kits", exit_early=True)
    # Check Program files
    if include_dir_ucrt is None:
        if os.path.isdir("C:\\Program Files\\Windows Kits"):
            include_dir_ucrt, include_dir_shared, include_dir_um = win_get_ucrt_shared_um_paths("C:\\Program Files\\Windows Kits", exit_early=True)
    # Ask user, cant find
    if include_dir_ucrt is None: 
        print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find 'Windows Kit' Directory:\n  1.) please manually enter full path to Windows Kits,\n  2.) enter q to quit script" + YELLOW_END)
        win_kit_path = win_reponse_or_exit()
        include_dir_ucrt, include_dir_shared, include_dir_um = win_get_ucrt_shared_um_paths(win_kit_path, exit_early=False)
    
    # Set LIB_UUID (UM) and LIB_UCRT paths 
    # figure out if we have x64 compile capabilities otherwise use x86
    # ucrt, um
    lib_ucrt_path =  win_get_lib_ucrt(include_dir_ucrt)
    lib_um_path = win_get_lib_um(include_dir_um)
        
    # Lib pthread
    isx64 = "x64" if  "x64" in lib_um_path else "x86" 
    lib_pthread_path = auto_config_path + "\\scripts\\windows\\pthread-win\\Pre-built.2\\lib\\" + isx64
    # Lib cpmt
    lib_cpmt_path = win_get_lib_cpmt(include_dir_mscv)
    # Write Results out to windows bat file:
    win_bat_file = auto_config_path + "\\win_cpp_make.bat"
    with open(win_bat_file, 'r') as handle:
        lines = handle.readlines()
        # Compiler
        idx_exe = win_get_line_idx(lines, "set EXE")
        if idx_exe == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_exe] = "set EXE=\"" + cl_exe + "\"\n"
        # INCLUDES
        # msvc
        idx_msvc = win_get_line_idx(lines, "set INC_MSVC")
        if idx_msvc == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_msvc] = "set INC_MSVC=-I\"" + include_dir_mscv + "\"\n"
        # ucrt 
        idx_ucrt = win_get_line_idx(lines, "set INC_UCRT")
        if idx_ucrt == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_ucrt] = "set INC_UCRT=-I\"" + include_dir_ucrt + "\"\n"
        # um
        idx_um = win_get_line_idx(lines, "set INC_WINUM")
        if idx_um == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_um] = "set INC_WINUM=-I\"" + include_dir_um + "\"\n"
        # shared 
        idx_shared = win_get_line_idx(lines, "set INC_WINSHR")
        if idx_shared == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_shared] = "set INC_WINSHR=-I\"" + include_dir_shared + "\"\n"
        # LIBRARIES
        #pthread 
        idx_pthread = win_get_line_idx(lines, "set LIB_PTHREAD")
        if idx_pthread == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_pthread] = "set LIB_PTHREAD=/LIBPATH:\"" + lib_pthread_path + "\"" + " pthreadVC2.lib\n"
        #uuid 
        idx_uuid = win_get_line_idx(lines, "set LIB_UUID")
        if idx_uuid == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_uuid] = "set LIB_UUID=/LIBPATH:\"" + lib_um_path + "\"" + " uuid.lib\n"
        #cpmt 
        idx_cpmt = win_get_line_idx(lines, "set LIB_CPMT")
        if idx_cpmt == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_cpmt] = "set LIB_CPMT=/LIBPATH:\"" + lib_cpmt_path + "\"" + " libcpmt.lib\n"
        #ucrt 
        idx_ucrt = win_get_line_idx(lines, "set LIB_UCRT")
        if idx_ucrt == -1:
            print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(win_bat_file)+RED_END)
            exit(1)
        lines[idx_ucrt] = "set LIB_UCRT=/LIBPATH:\"" + lib_ucrt_path + "\"" + " libucrt.lib\n"
    # Write Lines Out 
    with open(win_bat_file, 'w') as handle:
        handle.writelines(lines)
    # make sure pthread.dll is in bin 
    cauchy_bin_path = auto_config_path + "\\bin\\"
    pthread_dll_path = lib_pthread_path
    pthread_dll_path = pthread_dll_path.replace("\\lib\\", "\\dll\\")
    try:
        os.remove(cauchy_bin_path + "pthreadVC2.dll")
    except Exception:
        pass 
    shutil.copy(pthread_dll_path + "\\pthreadVC2.dll", cauchy_bin_path + "pthreadVC2.dll")
    # Now call .bat file and make 
    try:
        os.remove(cauchy_bin_path + "cauchy_estimator.exe")
    except Exception:
        pass 
    try:
        os.remove(cauchy_bin_path + "cauchy_estimtor.obj")
    except Exception:
        pass 
    os.chdir(cwd)
    result = subprocess.run([win_bat_file], check=True)
    if result.returncode == 0:
        print(GREEN_START+"C++ examples from\n {} have build successful in\n {}".format(c_examples_src_path, c_examples_bin_path)+GREEN_END)
    else:
        print(RED_END+"C++ examples did not compile successfully! Please look at print out above and assess problem." + RED_END)
        exit(1)       
    print(GREEN_START+"-----------------------------------------"+GREEN_END) 

def windows_setup_python_wrapper():
    cwd = os.getcwd()
    os_name = get_operating_sys()
    assert(os_name == 'windows')
    # Check Python Version >= 3
    py_ver = get_python_version()
    py_ver_maj = py_ver[:3]
    if int(py_ver[0]) < 3:
        print(RED_START+"[ERROR windows_setup_python_wrapper:] Python version {} is less than version 3...This is untested, and a very deprectated python version (Maybe it will work? Comment this block if you want to try). To prevent possible unforseen errors, use python >= 3.0.0 ... Exiting!" + RED_END)
        exit(1)
    # Check Numpy Installed
    if not is_numpy_installed():
        print(RED_START+"[ERROR windows_setup_python_wrapper:] Numpy is not installed. Please install numpy through your package manager...i.e., pip3 install numpy" + RED_END)
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
    # Get Compiler and linker paths 
    cl_exe = win_get_cl_exe()
    link_exe = cl_exe.replace("\\cl.exe", "\\link.exe")
    # Get Swig Exe Path and setup naming convention for local windows swig dir
    swig_zip_dirname = "swigwin-4.2.1"
    win_path = auto_config_path + "\\scripts\\windows"
    swig_zip_path = "{}\\{}.zip".format(win_path, swig_zip_dirname)
    swig_exe = "{}\\{}\\swig.exe".format(win_path,swig_zip_dirname)
    # Get MSVC Include path 
    bin_idx = cl_exe.find("\\bin")
    include_dir_mscv = cl_exe[:bin_idx] + "\\include"
    # Get UCRT, SHARED, UM include paths 
    if os.path.isdir("C:\\Program Files (x86)\\Windows Kits"):
        include_dir_ucrt, include_dir_shared, include_dir_um = win_get_ucrt_shared_um_paths("C:\\Program Files (x86)\\Windows Kits")
    elif os.path.isdir("C:\\Program Files\\Windows Kits"):
        include_dir_ucrt, include_dir_shared, include_dir_um = win_get_ucrt_shared_um_paths("C:\\Program Files\\Windows Kits") 
    else:
        print(YELLOW_START+"[WARN windows_setup_c_examples:] Cannot find 'Windows Kit' Directory:\n  1.) please manually enter full path to Windows Kits,\n  2.) enter q to quit script" + YELLOW_END)
        include_dir_ucrt = win_reponse_or_exit()
    # Libraries
    # Set LIB_UUID (UM) and LIB_UCRT paths 
    # figure out if we have x64 compile capabilities otherwise use x86
    # ucrt, um
    lib_ucrt_path =  win_get_lib_ucrt(include_dir_ucrt)
    lib_um_path = win_get_lib_um(include_dir_um)
    # Lib pthread
    x64_or_x86 = "x64" if  "x64" in lib_um_path else "x86" 
    lib_pthread_path = auto_config_path + "\\scripts\\windows\\pthread-win\\Pre-built.2\\lib\\" + x64_or_x86
    # Lib msvcrpt
    lib_msvcrpt_path = win_get_lib_msvcprt(include_dir_mscv, x64_or_x86)
    
    # Configure swigit_windows.bat
    swigit_file = "swigit_windows.bat"
    swigit_path = auto_config_path + "\\scripts\\swig\\cauchy\\" + swigit_file
    print(YELLOW_START+"Configuring Swig Install File:\n Located: {}".format(swigit_path)+YELLOW_END)
    with open(swigit_path, 'r') as handle:
        lines = handle.readlines()
        num_lines = len(lines)
    # Change EXE 
    idx_exe = win_get_line_idx(lines, "set MY_EXE")
    if idx_exe == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    cl_exe_swigit = "set MY_EXE=\"{}\"\n".format(cl_exe)
    lines[idx_exe] = cl_exe_swigit
    # Change LINK 
    idx_link = win_get_line_idx(lines, "set MY_LINK")
    if idx_link == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    link_exe_swigit = "set MY_LINK=\"{}\"\n".format(link_exe)
    lines[idx_link] = link_exe_swigit
    # Change SWIG_EXE
    idx_swig = win_get_line_idx(lines, "set MY_SWIG_EXE")
    if idx_swig == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    swig_exe_swigit = "set MY_SWIG_EXE=\"{}\"\n".format(swig_exe)
    lines[idx_swig] = swig_exe_swigit
    # Change INC_PYTHON
    idx_py = win_get_line_idx(lines, "set INC_PYTHON")
    if idx_py == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    py_include_swigit = "set INC_PYTHON=-I\"{}\"\n".format(py_include_path)
    lines[idx_py] = py_include_swigit
    # Change INC_NUMPY
    idx_np = win_get_line_idx(lines, "set INC_NUMPY")
    if idx_np == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    np_include_swigit = "set INC_NUMPY=-I\"{}\"\n".format(np_include_path)
    lines[idx_np] = np_include_swigit
    # Change INC_MSVC
    idx_mcsv = win_get_line_idx(lines, "set INC_MSVC")
    if idx_mcsv == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    msvc_include_swigit = "set INC_MSVC=-I\"{}\"\n".format(include_dir_mscv)
    lines[idx_mcsv] = msvc_include_swigit
    # Change INC_UCRT
    idx_ucrt = win_get_line_idx(lines, "set INC_UCRT")
    if idx_ucrt == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    ucrt_include_swigit = "set INC_UCRT=-I\"{}\"\n".format(include_dir_ucrt)
    lines[idx_ucrt] = ucrt_include_swigit
    # Change INC_WINSHR
    idx_winshr = win_get_line_idx(lines, "set INC_WINSHR")
    if idx_winshr == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    winshr_include_swigit = "set INC_WINSHR=-I\"{}\"\n".format(include_dir_shared)
    lines[idx_winshr] = winshr_include_swigit
    # Change INC_WINUM
    idx_winum = win_get_line_idx(lines, "set INC_WINUM")
    if idx_winum == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    winum_include_swigit = "set INC_WINUM=-I\"{}\"\n".format(include_dir_um)
    lines[idx_winum] = winum_include_swigit
    # Change LIB_PYTHON
    idx_libpy = win_get_line_idx(lines, "set LIB_PYTHON")
    if idx_libpy == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    python_lib_swigit = "set LIB_PYTHON=/LIBPATH:\"{}\" {}\n".format(py_lib_path, py_lib_so_name)
    lines[idx_libpy] = python_lib_swigit
    # Change LIB_UCRT
    idx_libucrt = win_get_line_idx(lines, "set LIB_UCRT")
    if idx_libucrt == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    ucrt_lib_swigit = "set LIB_UCRT=/LIBPATH:\"{}\" {}\n".format(lib_ucrt_path, "libucrt.lib")
    lines[idx_libucrt] = ucrt_lib_swigit
    # Change LIB_UM
    idx_libum = win_get_line_idx(lines, "set LIB_UM")
    if idx_libum == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    um_lib_swigit = "set LIB_UM=/LIBPATH:\"{}\" {}\n".format(lib_um_path, "uuid.lib")
    lines[idx_libum] = um_lib_swigit
    # Change LIB_PTHREAD
    idx_libpthread = win_get_line_idx(lines, "set LIB_PTHREAD")
    if idx_libpthread == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    pthread_lib_swigit = "set LIB_PTHREAD=/LIBPATH:\"{}\" {}\n".format(lib_pthread_path, "pthreadVC2.lib")
    lines[idx_libpthread] = pthread_lib_swigit
    # Change LIB_MSVC
    idx_libmsvc = win_get_line_idx(lines, "set LIB_MSVC")
    if idx_libmsvc == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(swigit_file)+RED_END)
        exit(1)
    msvc_lib_swigit = "set LIB_MSVC=/LIBPATH:\"{}\" {}\n".format(lib_msvcrpt_path, "msvcprt.lib")
    lines[idx_libmsvc] = msvc_lib_swigit
    # All done -- Write Lines
    with open(swigit_path, 'w') as handle:
        handle.writelines(lines)
    print(GREEN_START+"Updated\n {}\nwith new paths!".format(swigit_path)+GREEN_END)

    # Download the correct numpy file matching their numpy version 
    numpy_maj_version = get_numpy_version(maj_only=True)
    numpy_i_local_path = auto_config_path + "\\scripts\\swig\\cauchy\\numpy_{}.i".format(numpy_maj_version)
    numpy_i_local_default = auto_config_path + "\\scripts\\swig\\cauchy\\numpy.i"
    print(GREEN_START+"--- Auto Configuration Script Downloading numpy.i to:\n {}\n For numpy version={}: ---".format(swigit_path, numpy_maj_version) + GREEN_END)
    # If not downloaded, download and then softlink as numpy.i
    if not os.path.isfile(numpy_i_local_path):
        url_numpy_i = "https://raw.githubusercontent.com/numpy/numpy/maintenance/{}.x/tools/swig/numpy.i".format(numpy_maj_version)
        download_file(url_numpy_i, numpy_i_local_path)
    # Regardless, just re-symlink
    try:
        os.remove(numpy_i_local_default)
    except FileNotFoundError:
        pass 
    except PermissionError:
        print("Permission denied, {} could not be removed".format(numpy_i_local_default))
    except Exception:
        pass
    shutil.copy(numpy_i_local_path, numpy_i_local_default)
    print(GREEN_START+"---------------------------------------------------"+GREEN_END)

    # swigit_swindows.bat configured -- now download swig and unpack
    swig_local_dir = auto_config_path + "\\scripts\\windows"
    print(GREEN_START + "--- Auto Configuration Script Downloading swig, unpacking... ---" + GREEN_END)
    if not os.path.isfile(swig_zip_path):
        url_swig = "http://prdownloads.sourceforge.net/swig/" + swig_zip_dirname + ".zip"
        download_file(url_swig, swig_zip_path)
    # Extract Swig
    if not os.path.isdir(win_path + "\\" + swig_zip_dirname):
        print("Installing and configuring swig...")
        # create swig's install folder
        # Open the zip file
        with zipfile.ZipFile(swig_zip_path, 'r') as zip_ref:
            # Extract all contents
            zip_ref.extractall(win_path)
    
    # Move pthreadVC2.dll to swig/cauchy
    dll_pthread_path_src = lib_pthread_path.replace("\\lib", "\\dll") + "\\pthreadVC2.dll"
    dll_pthread_path_dst = auto_config_path + "\\scripts\\swig\\cauchy\\pthreadVC2.dll"
    if not os.path.isfile(dll_pthread_path_dst):
        shutil.copy(dll_pthread_path_src, dll_pthread_path_dst)
    
    # Call swigit_windows.bat
    swigit_run_path = auto_config_path + "\\scripts\\swig\\cauchy\\"
    os.chdir(swigit_run_path)
    result = subprocess.run([swigit_file], check=True)
    os.chdir(cwd)
    if result.returncode == 0:
        print(GREEN_START+"swigit_windows.bat executed successfully!"+GREEN_END)
    else:
        print(RED_START + "swigit_windows.bat did not execute successfully!" + RED_END)
        return
    print(GREEN_START+"---------------------------------------------------"+GREEN_END)
    pycauchy_ce_path = auto_config_path + "\\scripts\\swig\\cauchy\\cauchy_estimator.py"
    pycauchy_tut1_path = auto_config_path + "\\scripts\\tutorial\\lit_systems.ipynb"
    pycauchy_tut2_path = auto_config_path + "\\scripts\\tutorial\\nonlin_systems.ipynb"
    print(GREEN_START+"Python Wrapper build script completed:"+GREEN_END)
    print("The module:\n {}\nCan be included in your projects. Checkout the tutorials:\n {}\n {}\nto see examples".format(pycauchy_ce_path, pycauchy_tut1_path, pycauchy_tut2_path) +GREEN_END)
    print("Linking Cauchy Estimator Python Module to scripts/tutorials, scripts/filter_compare, scripts/leo")
    link_dirs = [auto_config_path+"\\scripts\\tutorial\\", auto_config_path+"\\scripts/swig\\filter_compare\\", auto_config_path+"\\scripts\\swig\\leo\\"]
    pthread_dll_dir = lib_pthread_path.replace("\\lib", "\\dll") 
    symlink_files(swigit_run_path, ["pycauchy.py", "_pycauchy.pyd", "_pycauchy.lib", "_pycauchy.exp", "cauchy_estimator.py", "gaussian_filters.py"], link_dirs)
    symlink_files(pthread_dll_dir, ["pthreadVC2.dll"], link_dirs)
    
def windows_setup_matlab_wrapper():
    
    swig_cauchy_path = get_auto_config_path() + "\\scripts\\swig\\cauchy"
    cpp_include_path = get_auto_config_path() + "\\include"
    lib_pthread_path = get_auto_config_path() + "\\scripts\\windows\\pthread-win\\Pre-built.2\\lib\\x64"
    # Copy pthread DLL to mex files
    dll_pthread_path_dst = get_auto_config_path() + "\\matlab\\mex_files\\pthreadVC2.dll"
    if not os.path.isfile(dll_pthread_path_dst):
        dll_pthread_path_src = lib_pthread_path.replace("\\lib", "\\dll") + "\\pthreadVC2.dll"
        shutil.copy(dll_pthread_path_src, dll_pthread_path_dst)
        
    includePath = "includePath = \'-I\"{}\" -I\"{}\"\';\n".format(swig_cauchy_path, cpp_include_path)
    libraryPath = "libraryPath = \'-L\"{}\" pthreadVC2.lib\';\n".format(lib_pthread_path)
    
    build_m_path = dll_pthread_path_dst.replace("pthreadVC2.dll", "build.m")
    with open(build_m_path, 'r') as handle:
        lines = handle.readlines()
    # Find include path 
    inc_idx = win_get_line_idx(lines, "includePath = ")
    if inc_idx == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(build_m_path)+RED_END)
        exit(1)
    lines[inc_idx] = includePath
    # Find Library path 
    lib_idx = win_get_line_idx(lines, "libraryPath = ")
    if lib_idx == -1:
        print(RED_START+"The {} file has been corrupted! Redownload in its entirety...Setting needed environment variable has failed! Exiting!".format(build_m_path)+RED_END)
        exit(1)
    lines[lib_idx] = libraryPath
    # Write out 
    with open(build_m_path, 'w') as handle:
        handle.writelines(lines)
    # Run
    run_matlab_script(build_m_path)
    print("Matlab Built!")

def windows_setup_msvc_pthread_dll():
    print(GREEN_START + " --------- Placing pthread dll into appropriate CauchyWindows MSVC directories ---------" + GREEN_END)
    pthread_dll_name = "pthreadVC2.dll"
    root_dir = get_auto_config_path()
    cauchy_win_path_dbg = root_dir + "\\scripts\\windows\\CauchyWindows\\Debug\\" + pthread_dll_name
    cauchy_win_path_rls = root_dir + "\\scripts\\windows\\CauchyWindows\\Release\\" + pthread_dll_name
    cauchy_win_path_x64dbg = root_dir + "\\scripts\\windows\\CauchyWindows\\x64\\Debug\\" + pthread_dll_name
    cauchy_win_path_x64rls = root_dir + "\\scripts\\windows\\CauchyWindows\\x64\\Release\\" + pthread_dll_name
    pthrean_dll_path = root_dir + "\\scripts\\windows\\pthread-win\\Pre-built.2\\dll\\x64\\" + pthread_dll_name
    if not os.path.isfile(cauchy_win_path_dbg):
        shutil.copy(pthrean_dll_path, cauchy_win_path_dbg)
    if not os.path.isfile(cauchy_win_path_rls):
        shutil.copy(pthrean_dll_path, cauchy_win_path_rls)
    if not os.path.isfile(cauchy_win_path_x64dbg):
        shutil.copy(pthrean_dll_path, cauchy_win_path_x64dbg)
    if not os.path.isfile(cauchy_win_path_x64rls):
        shutil.copy(pthrean_dll_path, cauchy_win_path_x64rls)
    print("Copied Successfully!")
    print(GREEN_START + " ---------------------------------------------------------------" + GREEN_END)

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
        windows_setup_msvc_pthread_dll()
    
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
