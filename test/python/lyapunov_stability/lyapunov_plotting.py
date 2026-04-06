import numpy as np
import matplotlib.pyplot as plt
import os


def plot_nocontrol_from_file(file,file_expect):
    data=np.loadtxt(file,delimiter=' ')
    lyapk = data[:,0]
    lyapk1 = data[:,1]
    lyap_diff = data[:,2]

    data_expec = np.loadtxt(file_expect,delimiter= ' ')
    expec_k = data_expec[:,0]
    expec_k1 = data_expec[:,1]
    expec_diff = data_expec[:,2]

    time = data.shape[0]

    delk = lyapk-expec_k[:-1]
    delk1= lyapk1-expec_k1[:-1]

    if False:
        plt.figure()
        plt.title("Difference in Lyapunov Functions")
        plt.plot(range(0,time),delk,label="difference at k")
        plt.plot(range(0,time),delk1,label = "difference at k+1")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.xlim(0,100)

    lw = 7
    lfz = 20
    lfzl =15
    lt = 20
    plt.figure(figsize=(13,20))
    #plt.title("Averaged Lyapunov Functions")
    plt.subplot(311)
    #plt.gca().set_title('Lyapunov Expectation at timestep k')
    plt.plot(range(0,time),lyapk,color='midnightblue',label="Expectation averaged over Measurements",lw=lw)
    plt.plot(range(0,time),expec_k[:-1],color='sandybrown',label="Unconditional Expectation",linestyle="dashed",lw=lw-2)
    plt.legend(fontsize= lfzl,frameon=False)
    plt.minorticks_on()
    plt.grid(which="minor",color="gainsboro")
    plt.grid(which="major")
    plt.xlim(0,100)
    plt.ylabel(r"$\bar{V}_{k|k}$",fontsize=lfz)
    #plt.rcParams.update({'font.size': 10})
    plt.tick_params(axis='y', which='major', labelsize=lt)
    plt.tick_params(axis='x',labelbottom='off')
    plt.gca().axes.xaxis.set_ticklabels([])

    plt.subplot(312)
    #plt.gca().set_title('Projected Lyapunov Expectation at timestep k+1')
    plt.plot(range(0,time),lyapk1,color='midnightblue',label="Projected Expectation averaged over measurements",lw=lw)
    plt.plot(range(0,time),expec_k1[:-1],color='sandybrown',label="Uncondtional Projected Expectation",linestyle="dashed",lw=lw-2)
    plt.legend(fontsize= lfzl,frameon=False)
    plt.minorticks_on()
    plt.grid(which="minor",color="gainsboro")
    plt.grid(which="major")
    plt.xlim(0,100)
    plt.ylabel(r"$\bar{V}_{k+1|k}$",fontsize=lfz)
    #plt.rcParams.update({'font.size': 10})
    plt.tick_params(axis='y', which='major', labelsize=lt)
    plt.gca().axes.xaxis.set_ticklabels([])

    plt.subplot(313)
    #plt.gca().set_title('Lyapunov Expectation Drift')
    plt.plot(range(0,time),lyap_diff,color='midnightblue',label="Drift averaged over measurements",lw=lw)
    plt.plot(range(0,time),expec_diff[:-1],color='sandybrown',label="Unconditional Drift",linestyle="dashed",lw=lw-2)
    plt.legend(fontsize= lfzl,frameon=False)
    plt.minorticks_on()
    plt.grid(which="minor",color="gainsboro")
    plt.grid(which="major")
    plt.xlabel('Time steps (k)',fontsize=lt)
    plt.xlim(0,100)
    plt.ylabel(r"$\bar{V}_{k+1|k} - \bar{V}_{k|k}$",fontsize=lfz)
    #plt.rcParams.update({'font.size': 10})
    plt.tick_params(axis='both', which='major', labelsize=lt)

    plt.show()

def plot_from_file(file):
    data=np.loadtxt(file,delimiter=' ')
    lyapk = data[:,0]
    lyapk1 = data[:,1]
    lyap_diff = data[:,2]

    avg_lyapk = round(np.average(lyapk[20:]),4)

    avg_lyapk1 = round(np.average(lyapk1[20:]),4)
    
    plt.figure()
    plt.title("Averaged Lyapunov Functions")
    plt.subplot(311)
    plt.plot(range(0,len(lyapk)),lyapk,color='green',label="E[V(xk)|y(k)]")
    plt.plot(range(0,len(lyapk)),avg_lyapk*np.ones(len(range(0,len(lyapk)))),color='red',linestyle="dashed",label=f"avg is {avg_lyapk}")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')
    plt.ylim(0,1)

    plt.subplot(312)
    plt.plot(range(0,len(lyapk1)),lyapk,color='green',label="E[V(xk+1)|y(k)]")
    plt.plot(range(0,len(lyapk1)),avg_lyapk1*np.ones(len(range(0,len(lyapk)))),color='red',linestyle="dashed",label=f"avg is {avg_lyapk1}")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')
    plt.ylim(0,1)

    plt.subplot(313)
    plt.plot(range(0,len(lyap_diff)),np.zeros(len(range(0,len(lyapk)))),color='red',linestyle="dashed")
    plt.plot(range(0,len(lyap_diff)),lyap_diff,color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')

    plt.show()

def access_all_runs(files,plot_all_runs=True,plot_avg = True, runs_to_del =[]):
    if not runs_to_del:
        saved_lyap_all = np.load(files[0])
    else: 
        saved_lyap_all = np.delete(np.load(files[0]),runs_to_del[0],axis=0)

    for file_ind,file in enumerate(files[1:]):
        if not runs_to_del:
            nextnp = np.load(file)
        else:
            nextnp = np.delete(np.load(file),runs_to_del[file_ind+1],axis=0)
        saved_lyap_all = np.append(saved_lyap_all,nextnp,axis=0)

    print(saved_lyap_all.shape)
    lyap_avg = np.mean(saved_lyap_all,axis=0)


    lyapk = lyap_avg[:,0]
    lyapk1 = lyap_avg[:,1]
    lyap_diff = lyapk1-lyapk

    avg_lyapk = np.average(lyapk[20:])
    avg_lyapk1 = np.average(lyapk1[20:])
    avg_diff = avg_lyapk1-avg_lyapk

    lw=4
    if plot_all_runs:
        plt.figure()
        plt.title("Lyapunov Functions for all runs")
        plt.subplot(211)
        plt.plot(range(0,len(saved_lyap_all[0,:,0])),np.transpose(saved_lyap_all[:,:,0]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk)|y(k)]")
    
        plt.subplot(212)
        plt.plot(range(0,len(saved_lyap_all[0,:,1])),np.transpose(saved_lyap_all[:,:,1]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk+1)|y(k)]")

        plt.show()
    
    if plot_avg:
        lw = 7
        lfz = 20
        lfzl =15
        lt = 20
        plt.figure(figsize=(13,20))
        plt.subplot(311)
        plt.plot(range(0,len(lyapk)),lyapk,color='midnightblue',lw=lw)
        plt.plot(range(0,len(lyapk)),round(avg_lyapk,5)*np.ones(len(range(0,len(lyapk)))),color='red',linestyle="dashed",label=f"avg = {round(avg_lyapk,5)}",lw=lw-2)
        plt.legend(fontsize=lfzl)
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.ylim(0,1)
        plt.xlim(0,100)
        plt.ylabel(r"$\bar{V}_{k|k}$",fontsize=lfz)
        plt.tick_params(axis='y', which='major', labelsize=lt)
        plt.tick_params(axis='x',labelbottom='off')
        plt.gca().axes.xaxis.set_ticklabels([])

        plt.subplot(312)
        plt.plot(range(0,len(lyapk1)),lyapk,color='midnightblue',lw=lw)
        plt.plot(range(0,len(lyapk1)),round(avg_lyapk1,5)*np.ones(len(range(0,len(lyapk)))),color='red',linestyle="dashed",label=f"avg = {round(avg_lyapk1,5)}",lw=lw-2)
        plt.legend(fontsize=lfzl)
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.ylim(0,1)
        plt.xlim(0,100)
        plt.ylabel(r"$\bar{V}_{k+1|k}$",fontsize=lfz)
        plt.tick_params(axis='y', which='major', labelsize=lt)
        plt.gca().axes.xaxis.set_ticklabels([])

        plt.subplot(313)
        plt.plot(range(0,len(lyap_diff)),lyap_diff,color='midnightblue',lw=lw)
        plt.plot(range(0,len(lyap_diff)),avg_diff*np.ones(len(range(0,len(lyapk)))),color='red',linestyle="dashed",label=f"avg = {round(avg_diff,5)}",lw=lw-2)
        plt.legend(fontsize=lfzl)
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('Time steps (k)',fontsize=lt)
        plt.xlim(0,100)
        plt.ylabel(r"$\bar{V}_{k+1|k} - \bar{V}_{k|k}$",fontsize=lfz)
        plt.tick_params(axis='both', which='major', labelsize=lt)

        plt.show()

def access_individ_runs(filepath,plot_all_runs=True,plot_bad_run = True):

    saved_lyap_all = np.load(filepath)

    if plot_all_runs:
        plt.figure()
        plt.title("Lyapunov Functions for all runs")
        plt.subplot(211)
        plt.plot(range(0,len(saved_lyap_all[0,:,0])),np.transpose(saved_lyap_all[:,:,0]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk)|y(k)]")
    
        plt.subplot(212)
        plt.plot(range(0,len(saved_lyap_all[0,:,1])),np.transpose(saved_lyap_all[:,:,1]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk+1)|y(k)]")

        plt.show()

    for checker in range(0,saved_lyap_all.shape[0]):
        if saved_lyap_all[checker,-2,0] > 8 and saved_lyap_all[checker,-1,0] > 8:
            print(checker)
            
            if plot_bad_run:
                plt.figure()
                plt.title("Lyapunov Functions for all runs")
                plt.subplot(211)
                plt.plot(range(0,len(saved_lyap_all[0,:,0])),np.transpose(saved_lyap_all[checker,:,0]),color='blue')
                plt.minorticks_on()
                plt.grid(which="minor")
                plt.grid(which="major")
                plt.xlabel('time')
                plt.ylabel("E[V(xk)|y(k)]")
            
                plt.subplot(212)
                plt.plot(range(0,len(saved_lyap_all[0,:,1])),np.transpose(saved_lyap_all[checker,:,1]),color='blue')
                plt.minorticks_on()
                plt.grid(which="minor")
                plt.grid(which="major")
                plt.xlabel('time')
                plt.ylabel("E[V(xk+1)|y(k)]")

                plt.show()
        

if __name__ == "__main__":
    # plot_from_file(file='saved_MC0.95.txt')
    # plot_from_file(file='saved_MC1.05.txt')

    plot_stable = False
    plot_unstable = True
    plot_nocontrol = False

    cd = os.getcwd()

    if plot_stable:
        files_stable = ["MC_1856stable_allruns.npy","MC_1000stable_1_allruns.npy","MC_579stable_2_allruns.npy","MC_1000stable_3_allruns.npy","MC_1000stable_4_allruns.npy","MC_852stable_5_allruns.npy","MC_1000stable_6_allruns.npy","MC_1000stable_7_allruns.npy","MC_1000stable_8_allruns.npy","MC_1000stable_9_allruns.npy"]
        folder = "stable10000"
        
        filepaths = [f"{cd}/{folder}/{file}" for file in files_stable]
        access_all_runs(filepaths,plot_all_runs=False)

    if plot_unstable:
        files_unstable = ["MC_1000unstable_0_allruns.npy","MC_2000unstable_1_allruns.npy","MC_2000unstable_2_allruns.npy","MC_2000unstable_3_allruns.npy","MC_3000unstable_4_allruns.npy","MC_3unstable_5_allruns.npy"]
        folder = "unstable10000"
        filepaths = [f"{cd}/{folder}/{file}" for file in files_unstable]
        runs_to_del = [[729],[],[],[181],[924],[]]
        access_all_runs(filepaths,plot_all_runs=False,runs_to_del=runs_to_del)
    
    if plot_nocontrol: 
        file = "MC_nocontrol_stable_avg.txt"
        file_expec = "MC_nocontrol_stable_expectation.txt"

        plot_nocontrol_from_file(file,file_expec)



    #access_individ_runs(filepaths[5])

    