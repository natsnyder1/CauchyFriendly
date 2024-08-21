import sys, os 
import pickle 


def change_run_dic(path, runs):
    if ".pickle" in path:
        fpath_prefix, fpath_desc = path.rsplit(".", 1)
    else:
        fpath_desc = path 
    if fpath_desc == ".pickle":
        pickle_path = path
    else:
        if path[-1] != "/":
            path += "/"
        pickle_path = path + "run_dic.pickle"
    
    if not os.path.isfile(pickle_path):
        print(pickle_path, " not a valid directory or path to 'run_dic.pickle' Goodbye!")
        exit(1)

    # Now load run dic and change
    with open(pickle_path, "rb") as handle:
        run_dic = pickle.load(handle)
    print("Read Run Dic!")

    run_dic["days_ensemble"] = runs 
    run_dic["ensembles_previously_logged"] = runs
    print("Changed Run Dic!")

    # Now load run dic and change
    with open(pickle_path, "wb") as handle:
        pickle.dump(run_dic, handle)
    print("Re-Wrote Run Dic!")

    # Change Human Readable
    hum_path, _ = pickle_path.rsplit(".", 1)
    hum_path += ".txt"
    with open(hum_path, "r") as handle:
        lines = handle.readlines()
    count = 0
    while not ("days_ensemble" in lines[count]):
        count += 1
    assert(count < len(lines))
    lines[count] = "days_ensemble : {},\n".format(runs)
    count = 0
    while not ("ensembles_previously_logged" in lines[count]):
        count += 1
    assert(count < len(lines))
    lines[count] = "ensembles_previously_logged : {},\n".format(runs)
    print("Changed Human Readable Run Dic!")

    with open(hum_path, "w") as handle:
        handle.writelines(lines)
    print("Re-Wrote Human Readable Run Dic!")


print(sys.argv)
if len(sys.argv) < 3:
    print("Must Enter Path to Run Dictionary 'run_dic.pickle' and the number of runs to change run_dic_to")

path = sys.argv[1]
runs = int(sys.argv[2])

#run_dic_path = "/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/CauchyFriendly/scripts/swig/leo/gmat_data/gps_2_11_23/pred_def_ensemble/run_foo"
#runs = 8 
change_run_dic(path, runs)