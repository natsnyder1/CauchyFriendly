import re
import os
from re import findall
import numpy as np
from inv_cpdf_fullscript import Term
from inv_cpdf_fullscript import convertListGToTableG

def import_bigtxt():
    filename = "cauchy_run.txt"
    filename = "naina_run.txt"
    cd = os.getcwd()
    filepath = f"{cd}/test/python/{filename}"
    with open (filepath) as run:
        fileread = run.read()
        filesplit = re.split("Current Timestep",fileread)
    for MU_num,MUfile in enumerate(filesplit): 
        if MU_num ==0:
            pass
        else:
            text_file = open(f"MU{MU_num}.txt", "w")
            text_file.write(MUfile)
            text_file.close()
    return

def splitMUtxt(MUfilename):
    cd = os.getcwd()
    filepath = f"{cd}/{MUfilename}"
    with open (filepath) as run:
        MUfileread = run.read()
    numterms = int(find_in_string(MUfileread,"Current Number of Terms")[0][0])
    split_MU = re.split("-----------------------------------------------------------\n",MUfileread)
    if int(MUfilename[2]) == 6: # CHANGE THIS CHANGE THIS MAKE IT == 6
        list_of_terms = split_MU[1:-2]
    else:
        list_of_terms = split_MU[1:-4]
    if numterms != len(list_of_terms):
        print("Error in parsing file!")
        return
    term_obj_list = []
    for term in list_of_terms:
        b_split = re.split("b:",term)
        b_str = b_split[1]
        b_vec = [float(b) for b in re.split("\n",b_str)[:-1]]


        p_split = re.split("p: ",b_split[0])
        p_str = p_split[1]
        p_vec = [float(p) for p in re.split("\n",p_str)[:-1]]

        BG_split  = re.split("B: ", p_split[0])
        BG =BG_split[1]
        chamber_split = re.split("\nChamber ", BG[:-2])
        B_list =[]
        G_list = []
        for chamber in chamber_split[1:]:
            B_int,g_val=convert_chamber_to_int(chamber)    
            B_list.append(B_int)
            G_list.append(g_val)    

        A_split = BG_split[0]
        A_hyplane_list = re.split("\nHyperplane",A_split[:-2])[1:]
        full_A = []
        for a_hyplane in A_hyplane_list:
            a_vals = convert_stringa_to_arr(a_hyplane)
            full_A.append(a_vals)
        full_A_arr = np.array(full_A)

        # pattern = "(?<=%s)[0-9.]+"
        # newpattern = pattern % ("dim=")
        # findall(newpattern,A_split)
        m = len(p_vec)
        n = len(b_vec)
        Q = np.identity(n)
        term = Term(parent=0,ndim=n,m_hyperplanes=m,A_hplane_arr=full_A_arr,expnt_b = b_vec,expnt_Q = Q,expnt_p = p_vec,enumeration_B=B_list,enumeration_G = convertListGToTableG(G_list,n))
        term_obj_list.append(term)
    return term_obj_list

def convert_chamber_to_int(chamber):
    enum_gval = re.split("g_val: ",chamber)
    g_val_str= re.split(",",enum_gval[1][1:-1])
    g_val = float(g_val_str[0]) + float(g_val_str[1])*1j

    sign_seq = enum_gval[0].replace('[','').replace(']','').replace(' ','')
    
    seq =[]
    for s in sign_seq: 
        if s == '+':
            seq+='0'
        else: 
            seq += '1'
    seq_str = ''.join(seq)
    B_int = int(seq_str,2)
    return B_int,g_val

def convert_stringa_to_arr(stringa):
    a_vec_str = stringa.replace("(a = [",'').replace("])",'')
    a_vals = re.split(" ",a_vec_str)
    a_vals_flt = [float(a_i) for a_i in a_vals]
    return a_vals_flt


def find_in_string(wholestring,find):
    found=[]
    pattern = "(?<=%s)[0-9.]+"
    newpattern = pattern % (find + ": ")
    found.append(findall(newpattern,wholestring))
    return found

if __name__ == "__main__":
    import_bigtxt()

    print(len(splitMUtxt("MU1.txt")))
    print(len(splitMUtxt("MU2.txt")))
    print(len(splitMUtxt("MU3.txt")))
    print(len(splitMUtxt("MU6.txt")))

    #splitMUtxt("MU6.txt")

    #text = "cat: 4, mouse: 8"
    #find_in_string(text,"(?<=%s)[0-9.]+","cat")