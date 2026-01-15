# imports
import numpy as np
import math
pi = math.pi
comb = math.comb
copysign = math.copysign
from collections import Counter
import copy
import random
import matplotlib
import matplotlib.pyplot as plt


""" 
Term class
contains all the information in the exponential (hyperplane arrangements, p, b etc)
also contains B enumeration table and symbolic G table
"""
class Term:
    def __init__(self,parent,ndim,m_hyperplanes,A_hplane_arr,expnt_b,expnt_Q,expnt_p,enumeration_B,enumeration_G): #,coeff,num_past_integrations):
        """
        A_hplane_arr: hyperplane arrangements A

        """
        self.A_hplane_arr = A_hplane_arr
        self.parent = parent
        self.expnt_b = expnt_b
        self.ndim = ndim
        self.m_hyperplanes = m_hyperplanes
        self.expnt_Q = expnt_Q
        self.expnt_p = expnt_p
        self.enumeration_B = enumeration_B
        self.enumeration_G = enumeration_G
        

    def calcMuZero(self,zero_inds):
        mu=np.empty((self.m_hyperplanes,self.ndim-1))
        for l in range(self.m_hyperplanes):
            if l in zero_inds:
                mu[l,:] = self.A_hplane_arr[l,0:self.ndim-1]
            else:
                mu[l,:] = -1 * self.A_hplane_arr[l,0:self.ndim-1]/self.A_hplane_arr[l,-1]
        return mu

    def findChildArrangementZeros(self,child_t,zero_inds): # ind child is 1,2,3,4
        child_dim = self.ndim-1
        A_child = np.empty((self.m_hyperplanes-1,child_dim))
        mu = self.calcMuZero(zero_inds)
        mu_t = mu[child_t-1,:]
        for l in range(self.m_hyperplanes): # indexes from 0 to 1
            if l in zero_inds:
                A_child[l,:] = self.A_hplane_arr[l,0:-1]
            else:
                if l < child_t-1:   
                    A_child[l,:] = mu[l,:] - mu_t
                elif l > child_t-1:
                    A_child[l-1,:] = mu[l,:] - mu_t
        return A_child,mu_t
    
    
    def findAtildZeros(self):
        A_tild = self.A_hplane_arr[:,-1].tolist()
        zero_inds = [ind for ind,element in enumerate(A_tild) if element == 0]
        return A_tild,zero_inds
        

    def generateChildEnumerationWithZeros(self,child_t):
        """
        for i in enumeration length
        make g bundles
        combine g bundles with previous bundles by updating list of bundles associated with this enumeration
        
        
        """
        # define sign sequence and absolute value associated with A_tild 
        A_tild,zero_inds = self.findAtildZeros()

        child_m = self.m_hyperplanes - 1
        child_n = self.ndim - 1
        B = self.enumeration_B
        G = self.enumeration_G
        B_to_G_dict = dict(zip(B,G))
        
        if child_t-1 in zero_inds:
            return "child index cannot be a row that has a_tild = 0"
        
        childA,mu_t = self.findChildArrangementZeros(child_t,zero_inds)

        #A_tild_sign = [copysign(1,a_til) for a_til in A_tild]
        A_tild_sign = [copysign(1,a_til) if a_til != 0 else 0 for a_til in A_tild]
        A_tild_abs = [abs(a_tild) for a_tild in A_tild]
        
        # negate sign sequence associated with A tild
        A_tild_sign_min = [a*-1 for a in A_tild_sign]
        
        # make negated sign sequence into binary string
        # A_tild_bin_list = ['0' if s == 1  else '1' for s in A_tild_sign_min ] OLD. THIS HAD ERRROR
        A_tild_bin_list = ['1' if s == -1  else '0' for s in A_tild_sign_min ]
        A_tild_bin = ''.join(A_tild_bin_list)
        
        # gather constants that will be in denominator and stay in exponent
        b = self.expnt_b
        Q = self.expnt_Q
        p = self.expnt_p

        comp_b = b[-1]
        comp_q = Q[-1,:]

        new_b = b[:-1]+comp_b*mu_t
        new_Q = Q[0:-1,:]+np.outer(mu_t,comp_q)
        
        # childB
        childB = self.generateChildB(child_t=child_t,lambda_tild=A_tild_bin)

        # generate p's that will be in the denominator of the new g's 
        p_il = [p[i] * A_tild_abs[i] for i in range(len(A_tild_abs))]
        p_it = p_il[child_t-1]
        new_p = p_il[:child_t-1] + p_il[child_t:]

        new_p_ign0 = [p_il[i] if p_il[i] != 0 else p[i] for i in range(len(p_il[:child_t-1]))] + [p_il[child_t+i] if p_il[child_t+i] != 0 else p[child_t+i] for i in range(len(p_il[child_t:]))]
        
        # for this child, construct enumeration table
        G_child = []
        for lam_child in childB: 
            lam_child_bin = f'{{0:0{child_m}b}}'.format(lam_child)

            # find the sum of p's in the denominator of g using sign sequence
            p_sum = 0
            for index,c in enumerate(lam_child_bin):
                if c == '0':
                    p_sum +=new_p[index]
                else:
                    p_sum -= new_p[index]
                
            # find the sgn sequence for the numerators (for parents)
            sgn_plus = insert(lam_child_bin,'0',child_t-1)
            sgn_minus = insert(lam_child_bin,'1',child_t-1)
            
            # generate sgn sequence for numerators, and access the associated G row from parent
            parent_enum_plus_ind = int(sgn_plus,2) ^ int(A_tild_bin,2)
            #parent_enum_plus = self.enumeration_G[parent_enum_plus_ind]
            parent_enum_plus = B_to_G_dict[parent_enum_plus_ind]
            parent_enum_minus_ind = int(sgn_minus,2) ^ int(A_tild_bin,2)
            parent_enum_minus = B_to_G_dict[parent_enum_minus_ind]
            #parent_enum_minus = self.enumeration_G[parent_enum_minus_ind]

            if isinstance(parent_enum_plus[0][0],Bundle):
                # calculate components of new bundles
                plus_denom_qt0 = p_it+p_sum+1j*comp_b
                plus_denom_qt = -1j*comp_q

                minus_denom_qt0 = -(-p_it+p_sum+1j*comp_b)
                minus_denom_qt = 1j*comp_q

                # make these two into bundles to be added to the enumeration table
                plus_bundle = Bundle(1,plus_denom_qt0,plus_denom_qt)
                minus_bundle = Bundle(1,minus_denom_qt0,minus_denom_qt)

                # plus bundle should be added to the list of list of bundles from parent +
                # minus bundle should be added to the list of list of bundles from parent - 
                new_plus_list = copy.deepcopy(parent_enum_plus)
                [x.append(plus_bundle) for x in new_plus_list]
                new_minus_list = copy.deepcopy(parent_enum_minus)
                [x.append(minus_bundle) for x in new_minus_list]
            else: 
                #first integration step: g's are values, not bundles

                # calculate components of new bundles
                if parent_enum_plus[0][0] == 0: 
                    plus_bundle = Bundle(0,1,0*comp_q)
                else: 
                    plus_denom_qt0 = (p_it+p_sum+1j*comp_b)/parent_enum_plus[0][0]
                    plus_denom_qt = -1j*comp_q/parent_enum_plus[0][0]
                    # make this into bundle to be added to the enumeration table
                    plus_bundle = Bundle(1,plus_denom_qt0,plus_denom_qt)
                
                if parent_enum_minus[0][0] == 0: 
                    minus_bundle = Bundle(0,1,0*comp_q)
                else:
                    minus_denom_qt0 = -(-p_it+p_sum+1j*comp_b)/parent_enum_minus[0][0]
                    minus_denom_qt = 1j*comp_q/parent_enum_minus[0][0]
                    # make this into bundle to be added to the enumeration table
                    minus_bundle = Bundle(1,minus_denom_qt0,minus_denom_qt)

                # plus bundle should be added to the list of list of bundles from parent +
                # minus bundle should be added to the list of list of bundles from parent - 
                new_plus_list = [[]]
                [x.append(plus_bundle) for x in new_plus_list]
                new_minus_list = [[]]
                [x.append(minus_bundle) for x in new_minus_list]
            
            # add this list of lists of bundles to the child enumeration table
            G_child.append( new_plus_list+new_minus_list)
        
        # create new child term using childA, the newly calculated b, q, p, and the new enumeration table
        childTerm = Term(self,child_n,child_m,childA,new_b,new_Q,new_p_ign0,childB,G_child)
        coaligned_ind,p_adj_fullen = childTerm.coalignmentCheck()
        if len(coaligned_ind) != 0:
            newChildTerm = childTerm.coalignmentAdjustment(coaligned_ind,p_adj_fullen)
            return newChildTerm

        return childTerm
    
    def generateLastChild(self):
        # gather constants that will be in denominator and stay in exponent
        b = self.expnt_b
        Q = self.expnt_Q
        p = self.expnt_p

        A_tild = self.A_hplane_arr[:,-1].tolist()
        A_tild_sign = [copysign(1,a_til) for a_til in A_tild]
        A_tild_abs = [abs(a_tild) for a_tild in A_tild]

        B_to_G_dict = dict(zip(self.enumeration_B,self.enumeration_G))

        G_child = []

        for hplan_ind,a_til in enumerate(A_tild):
        # calculate components of new bundles
            sgn_a_til = A_tild_sign[hplan_ind]
            first_bundle_sign = -1*sgn_a_til
            second_bundle_sign = sgn_a_til
            
            parent_enum_first_ind = int(self.convertSignToBin(first_bundle_sign),2)
            parent_enum_plus = B_to_G_dict[parent_enum_first_ind]
            parent_enum_second_ind = int(self.convertSignToBin(second_bundle_sign),2)
            parent_enum_minus = B_to_G_dict[parent_enum_second_ind]

            if isinstance(parent_enum_plus[0][0],Bundle):
                first_denom_qt0 = p[0]*A_tild_abs[hplan_ind]+1j*b[0]
                first_denom_qt = -1j*Q[-1,:]

                second_denom_qt0 = -(-p[0]*A_tild_abs[hplan_ind] +1j*b[0])
                second_denom_qt = 1j*Q[-1,:]
                
                first_bundle = Bundle(1,first_denom_qt0,first_denom_qt)
                second_bundle = Bundle(1,second_denom_qt0,second_denom_qt)

                new_first_list = parent_enum_plus
                [x.append(first_bundle) for x in new_first_list]
                new_second_list = parent_enum_minus
                [x.append(second_bundle) for x in new_second_list]
                
            else:
                first_denom_qt0 = (p[0]*A_tild_abs[hplan_ind]+1j*b[0]) /parent_enum_plus[0][0]
                first_denom_qt = -1j*Q[-1,:] /parent_enum_plus[0][0]

                second_denom_qt0 = -(-p[0]*A_tild_abs[hplan_ind] +1j*b[0]) /parent_enum_minus[0][0]
                second_denom_qt = 1j*Q[-1,:] /parent_enum_minus[0][0]
                
                first_bundle = Bundle(1,first_denom_qt0,first_denom_qt)
                second_bundle = Bundle(1,second_denom_qt0,second_denom_qt)

                new_first_list = [[]]
                [x.append(first_bundle) for x in new_first_list]
                new_second_list = [[]]
                [x.append(second_bundle) for x in new_second_list]
            
            G_child.append( new_first_list+new_second_list)
        
        return G_child
    

    def convertSignToBin(self,sign):
        if sign == -1:
            return '1'
        elif sign == 1:
            return '0'

    def checkLastChild(self):
        if self.ndim == 1:
            return True
        else:
            return False
    
    def findChildrenIndices(self):
        all_inds = range(self.m_hyperplanes)
        A_tild,zero_inds = self.findAtildZeros()
        child_inds = [ind + 1 for ind in all_inds if ind not in zero_inds]
        return child_inds
    
    def generateChildB(self,child_t,lambda_tild):
        # CHECK CHILD IND
        ind_child = child_t-1

        B = self.enumeration_B
        m = self.m_hyperplanes
        B_bin = [f'{{0:0{m}b}}'.format(lam_base10) for lam_base10 in B]
        B_bin_child_removed = [lamda[0:ind_child]+lamda[ind_child+1:] for lamda in B_bin]
        newB_bin_try2 = []
        enum_used = []
        for enum_index,enum_t_removed in enumerate(B_bin_child_removed):
            if B_bin_child_removed.count(enum_t_removed) == 2 and enum_t_removed not in enum_used: 
                enum_used.append(enum_t_removed)
                lambda_a = B_bin[enum_index]
                lambda_hprod= int(lambda_a,2) ^ int(lambda_tild,2)
                lambda_hprod_bin = f'{{0:0{m}b}}'.format(lambda_hprod)
                lambda_plus = lambda_hprod_bin[0:ind_child] + '0' + lambda_hprod_bin[ind_child+1:]
                lambda_minus = lambda_hprod_bin[0:ind_child] + '1' + lambda_hprod_bin[ind_child+1:]
                lambda_bar = lambda_plus[:ind_child] + lambda_plus[ind_child+1:]
                newB_bin_try2.append(lambda_bar)

        newB_try2 = [int(lam_bin,2) for lam_bin in newB_bin_try2]

        counter_dict = Counter(B_bin_child_removed)
        newB_bin = [key for key in counter_dict if counter_dict[key]==2]
        newB = [int(lam_bin,2) for lam_bin in newB_bin]

        return newB_try2
    
    def coalignmentCheck(self):
        m = self.m_hyperplanes
        A = self.A_hplane_arr
        p = self.expnt_p

        coaligned_indices = []

        for i_hyplane in range(m):
            for j_hyplane in range(m):
                a_i = A[i_hyplane,:]
                a_j = A[j_hyplane,:]
                if i_hyplane != j_hyplane:
                    inner_prod = np.inner(a_i,a_j)
                    check = 1-abs(inner_prod)/(np.linalg.norm(a_i)*np.linalg.norm(a_j))
                    if check < 1e-6: #the planes are co-aligned
                        if i_hyplane not in coaligned_indices and j_hyplane not in coaligned_indices:
                            coaligned_indices.append(j_hyplane)
                            p[i_hyplane] = p[i_hyplane] + p[j_hyplane]
        return coaligned_indices,p

    def coalignmentAdjustment(self,coaligned_indices,p_adj_fulllen):
        # make binary string out of all enumerations 
        # remove all indices in coaligned pairs
        # regenerate 
        A = self.A_hplane_arr
        G = self.enumeration_G
        m = self.m_hyperplanes
        B = self.enumeration_B

        B_to_G_dict = dict(zip(B,G))

        # enum_seq = range(enum_length)
        # enum_seq_in_bin = [bin(enum_i)[2:] for enum_i in enum_seq]
        
        enum_seq_in_bin = [f'{{0:0{m}b}}'.format(lam_child) for lam_child in B]
        
        old_enum_list = []
        for index in coaligned_indices:
            enum_seq_in_bin = [enum_b[:index]+enum_b[index+1:] for enum_b in enum_seq_in_bin]
            A = np.delete(A, (index), axis=0)
            p_adj_fulllen = p_adj_fulllen[:index]+p_adj_fulllen[index+1:]

        short_to_ful_len_dict = dict(zip(enum_seq_in_bin,B))

        dedup_enum_seq_in_bin = list(dict.fromkeys(enum_seq_in_bin))
        dedup_enum_seq = [int(enum,2) for enum in dedup_enum_seq_in_bin]

        new_enum_leng = len(dedup_enum_seq_in_bin)
        # make new G with 
        new_G = []
        for ind in dedup_enum_seq_in_bin:
            new_G.append(B_to_G_dict[short_to_ful_len_dict[ind]])

        newTerm = Term(self.parent,self.ndim,self.m_hyperplanes-len(coaligned_indices),A,self.expnt_b,self.expnt_Q,p_adj_fulllen,dedup_enum_seq,new_G)
        return newTerm
    

""" Bundle class"""
class Bundle:
    def __init__(self,const_num,const_den,coeff_den):
        self.const_num = const_num
        self.const_den = const_den
        self.coeff_den = coeff_den
    
    def multiplyBundles(self,newBundle):
        pass

# General useful functions
def convertConstGToBundle(G_table,dim_x):
    newG = [[[Bundle(enum_list[0][0],1,[0]*dim_x)]] for enum_list in G_table]
    return newG

def convertListGToTableG(G_table,dim_x):
    newG = [[[enum_list]] for enum_list in G_table]
    return newG

def insert(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]

# figure out how to unpack these lists slightly better!!!
def findSymbolic_UCPDF(listofterms):
    finalIntegral = []
    for term in listofterms:

        # check coalignment
        coaligned_ind,p_adj_fullen = term.coalignmentCheck()
        if len(coaligned_ind) != 0:
            term = term.coalignmentAdjustment(coaligned_ind,p_adj_fullen)

        bundlelist = getPDFPerTerm(term)
        [finalIntegral.append(productbundle) for allproducts in bundlelist for productbundle in allproducts ]
    return finalIntegral 

def getPDFPerTerm(term):
    # recursion starts here
    if term.checkLastChild():
            return term.generateLastChild()
    else:
        child_inds = term.findChildrenIndices()
        res = []
        for child_num in child_inds:
            childTerm = term.generateChildEnumerationWithZeros(child_num)
            # res.append(*getPDFPerTerm(childTerm))
            res.append(getPDFPerTerm(childTerm)[0])
        return res

def evaluateAtX(symbolic_UCPDF,x_vec,fz):
    sumPDF = 0
    for productList in symbolic_UCPDF:
        product = 1
        for bundle in productList:
            dot_prod = sum([x*y for x,y in zip(x_vec,bundle.coeff_den)])
            val = bundle.const_num/ (bundle.const_den + dot_prod)
            product *= val
        
        sumPDF += product
    
    return sumPDF / (fz*(2*math.pi)**len(x_vec))

def print_UCPDF(symbolic_UCPDF):
    for r, sum_r in enumerate(symbolic_UCPDF):
        print(f"sum: {r}")
        print([f"{prod_p.const_num}/[{prod_p.const_den}+{prod_p.coeff_den}x_k]" for prod_p in sum_r])
    pass

def calcfz(listofterms):
    fz = 0
    n_dim = listofterms[0].ndim
    rand_x = np.array([random.uniform(-1,1) for i in range(n_dim)])
    for term in listofterms:
        A = term.A_hplane_arr
        sign_seq =  [copysign(1,np.inner(rand_x,A[i,:])) for i in range(term.m_hyperplanes)]
        sign_bin_list = [term.convertSignToBin(sign) for sign in sign_seq]
        sign_bin = ''.join(sign_bin_list)
        B_to_G_dict = dict(zip(term.enumeration_B,term.enumeration_G))
        g_coeff = B_to_G_dict[int(sign_bin,2)][0][0].const_num if isinstance(B_to_G_dict[int(sign_bin,2)][0][0], Bundle) else B_to_G_dict[int(sign_bin,2)][0][0]
        
        fz += g_coeff
    return fz

def calc_marginal(listofterms):

    pass

def plot2d_from_3d_pdf(symbolicPDF_3d):# set up a figure three times as wide as it is tall  
    # 2D Grid Params
    g2lx = -2
    g2hx = 2
    g2rx = 0.025
    g2ly = -2
    g2hy = 2
    g2ry = 0.025

    x_grid = range(g2lx,g2hx,g2rx)
    y_grid = range(g2ly, g2hy, g2ry)

    pdf_12 = np.empty((len(x_grid),len(y_grid)))
    pdf_13 = np.empty((len(x_grid),len(y_grid)))
    pdf_23 = np.empty((len(x_grid),len(y_grid)))

    x_grid_2d = np.empty((len(x_grid),len(y_grid)))
    y_grid_2d = np.empty((len(x_grid),len(y_grid)))

    for i_x,x in enumerate(x_grid):
        for i_y,y in enumerate(y_grid): 
            pdf_12[i_x,i_y] = evaluateAtX(symbolicPDF_3d,[x,y,0])
            pdf_13[i_x,i_y] = evaluateAtX(symbolicPDF_3d,[x,0,y])
            pdf_23[i_x,i_y] = evaluateAtX(symbolicPDF_3d,[0,x,y])
            x_grid_2d[i_x,i_y] = x
            y_grid_2d[i_x,i_y] = y        

    fig1 = plt.figure(figsize = (18,5))

    ax12 = fig1.add_subplot(1,3,1,projection='3d')
    ax13 = fig1.add_subplot(1,3,2,projection='3d')
    ax23 = fig1.add_subplot(1,3,3,projection='3d')

    # Marg (0,1)
    ax12.set_title("Marginal of States 1 and 2", pad=-15)
    ax12.plot_wireframe(x_grid_2d, y_grid_2d, pdf_12, zorder=2, color='b')
    ax12.set_xlabel("x-axis (State-1)")
    ax12.set_ylabel("y-axis (State-2)")
    ax12.set_zlabel("z-axis (CPDF Probability)")
    # Marg (0,2)
    ax13.set_title("Marginal of States 1 and 3", pad=-8)
    ax13.plot_wireframe(x_grid_2d, y_grid_2d, pdf_13, zorder=2, color='g')
    ax13.set_xlabel("x-axis (State-1)")
    ax13.set_ylabel("y-axis (State-3)")
    ax13.set_zlabel("z-axis (CPDF Probability)")
    # Marg (1,2)
    ax23.set_title("Marginal of States 2 and 3", pad=-8)
    ax23.plot_wireframe(x_grid_2d, y_grid_2d, pdf_23, zorder=2, color='r')
    ax23.set_xlabel("x-axis (State-2)")
    ax23.set_ylabel("y-axis (State-3)")
    ax23.set_zlabel("z-axis (CPDF Probability)")
    
def plot2d_from_4d(symbolic_UCPDF_4d,fz):
     # 2D Grid Params
    g2lx = -2
    g2hx = 2
    g2rx = 0.025
    g2ly = -2
    g2hy = 2
    g2ry = 0.025

    x_grid = np.arange(g2lx,g2hx,g2rx)
    y_grid = np.arange(g2ly, g2hy, g2ry)

    pdf = np.empty((len(x_grid),len(y_grid)))
    pdf1 = np.empty(len(x_grid))
    pdf2 = np.empty(len(y_grid))

    x_grid_2d = np.empty((len(x_grid),len(y_grid)))
    y_grid_2d = np.empty((len(x_grid),len(y_grid)))

    for i_x,x in enumerate(x_grid):
        for i_y,y in enumerate(y_grid): 
            pdf[i_x,i_y] = evaluateAtX(symbolic_UCPDF_4d,[x,y,0,0],fz)
            if abs(pdf[i_x,i_y].imag) > 1e-3:
                print(pdf[i_x,i_y])
            x_grid_2d[i_x,i_y] = x
            y_grid_2d[i_x,i_y] = y     

    fig1 = plt.figure(figsize = (5,5))
    ax = fig1.subplots(1,1,subplot_kw={'projection': '3d'})

    # Marg (0,1)
    ax.set_title("States 1 and 2", pad=-15)
    ax.plot_wireframe(x_grid_2d, y_grid_2d, pdf, zorder=2, color='b')
    ax.set_xlabel("x-axis (State-1)")
    ax.set_ylabel("y-axis (State-2)")
    ax.set_zlabel("z-axis (CPDF Probability)")

    plt.show()
    plt.close()

def plot2d_from_2d(symbolic_UCPDF_2d,fz):
     # 2D Grid Params
    g2lx = -2
    g2hx = 2
    g2rx = 0.025
    g2ly = -2
    g2hy = 2
    g2ry = 0.025

    x_grid = np.arange(g2lx,g2hx,g2rx)
    y_grid = np.arange(g2ly, g2hy, g2ry)

    pdf = np.empty((len(x_grid),len(y_grid)))
    pdf1 = np.empty(len(x_grid))
    pdf2 = np.empty(len(y_grid))

    x_grid_2d = np.empty((len(x_grid),len(y_grid)))
    y_grid_2d = np.empty((len(x_grid),len(y_grid)))

    for i_x,x in enumerate(x_grid):
        for i_y,y in enumerate(y_grid): 
            pdf[i_x,i_y] = evaluateAtX(symbolic_UCPDF_2d,[x,y],fz)
            if abs(pdf[i_x,i_y].imag) > 1e-3:
                print(pdf[i_x,i_y])
            x_grid_2d[i_x,i_y] = x
            y_grid_2d[i_x,i_y] = y  
            pdf2[i_y] = evaluateAtX(symbolic_UCPDF_2d,[0,y],fz)
        pdf1[i_x] = evaluateAtX(symbolic_UCPDF_2d,[x,0],fz)      

    fig1 = plt.figure(figsize = (5,5))
    ax = fig1.subplots(1,1,subplot_kw={'projection': '3d'})

    # Marg (0,1)
    ax.set_title("States 1 and 2", pad=-15)
    ax.plot_wireframe(x_grid_2d, y_grid_2d, pdf, zorder=2, color='b')
    ax.set_xlabel("x-axis (State-1)")
    ax.set_ylabel("y-axis (State-2)")
    ax.set_zlabel("z-axis (CPDF Probability)")

    # Marg 1D
    # set up a figure three times as wide as it is tall
    fig2 = plt.figure(figsize = (18,4))
    ax1 = fig2.add_subplot(1,3,1)
    ax2 = fig2.add_subplot(1,3,2)
    # Marg 1
    ax1.set_title("1D Marg of State 1")
    ax1.plot(x_grid,pdf1)
    ax1.set_xlabel("State 1")
    ax1.set_ylabel("CPDF Probability")
    # Marg 2
    ax2.set_title("1D Marg of State 2")
    ax2.plot(y_grid,pdf2)
    ax2.set_xlabel("State 2")
    ax2.set_ylabel("CPDF Probability")

    plt.show()
    plt.close()

def expand_BG_tables(m,B,G):
    newB = B.copy()
    newG = G.copy()
    for enum_ind,enum_B in enumerate(B): 

        enum_B_bin = f'{{0:0{m}b}}'.format(enum_B)
        enum_G = G[enum_ind]
        #enum_B_bin_flip = enum_B_bin
        enum_B_bin_flip = enum_B_bin.replace('1', '2').replace('0', '1').replace('2', '0')
        enum_G_flip = np.conjugate(enum_G)

        newB.append(int(enum_B_bin_flip,2))
        newG.append(enum_G_flip)
        
    return newB, newG

# Tests 
def test_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 1

    # term components
    # G1 = [
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2+gamma)-1/(1j*z1-alpha_1+alpha_2+gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2-gamma)-1/(1j*z1-alpha_1+alpha_2-gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2+gamma)-1/(1j*z1-alpha_1-alpha_2+gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2-gamma)-1/(1j*z1-alpha_1-alpha_2-gamma))]] 
    #   ]
    G1 = [
      [[1/(2*pi)*(1/(1j*z1+alpha_1+gamma+alpha_2)-1/(1j*z1-alpha_1+gamma+alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1+gamma-alpha_2)-1/(1j*z1-alpha_1+gamma-alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-gamma+alpha_2)-1/(1j*z1-alpha_1-gamma+alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-gamma-alpha_2)-1/(1j*z1-alpha_1-gamma-alpha_2))]] 
      ]
    A1 = np.array([[-1,0], [-1, 1]])
    b1 = [z1,0]
    p1 = [gamma,alpha_2]
    Q1 = np.identity(2)

    # child 2
    G2 = [
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1+gamma)-1/(1j*z1-alpha_2+alpha_1+gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1-gamma)-1/(1j*z1-alpha_2+alpha_1-gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1+gamma)-1/(1j*z1-alpha_2-alpha_1+gamma))]],
        [[ 1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1-gamma)-1/(1j*z1-alpha_2-alpha_1-gamma))]]
        ]
    A2 = np.array([[1,-1], [0, -1]])
    b2 = [0,z1]
    p2 = [alpha_1,gamma]
    Q2 = np.identity(2)

    G3 = [
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1+alpha_2)-1/(1j*z1-gamma+alpha_1+alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1-alpha_2)-1/(1j*z1-gamma+alpha_1-alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma-alpha_1+alpha_2)-1/(1j*z1-gamma-alpha_1+alpha_2))]],
        [[ 1/(2*pi)*(1/(1j*z1+gamma-alpha_1-alpha_2)-1/(1j*z1-gamma-alpha_1-alpha_2))]]
            ]
    A3 = np.array([[1,0], [0, 1]])
    b3 = [0,0]
    p3 = [alpha_1,alpha_2]
    Q3 = np.identity(2)

    B = [0,1,2,3]

    # # define terms
    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))
    
    # define terms
    # term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=G1 )    
    # term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=G2)
    # term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=G3)
    

    resultingint = findSymbolic_UCPDF([term1,term2,term3])
    #print(resultingint)
    fz = calcfz([term1,term2,term3])



    try1 = findSymbolic_UCPDF([term1])
    try2 = findSymbolic_UCPDF([term2])
    try3 = findSymbolic_UCPDF([term3])
    print(f"first:{evaluateAtX(try1,[0.2,0.1],fz)}, second: {evaluateAtX(try2,[0.2,0.1],fz)}, third: {evaluateAtX(try3,[0.2,0.1],fz)} ")

    # print_UCPDF(resultingint)

    print(evaluateAtX(resultingint,[0.2,0.1],fz))
    plot2d_from_2d(resultingint,fz)

def test_4d():
    n_dim = 4 # number of dimensions
    m_i = 5 # number of hyperplanes

    A1=np.array([ [0.479452,0,0.205479,0], 
    [0.731707,-1.21951,0.731707,-0], 
    [0.558659,-0.27933,0.111732,-0], 
    [0,0,0,-10], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p1=[ 0.292, 0.0656, 0.0895, 0.005, 0.027 ]
    b1=[ 0, 0, 0, 0 ]
    Enc_B1 = [ 9,  8,  11,  10,  13,  12,  15,  14,  1,  0,  3,  2,  4,  7,  6] 
    G1 = [0.130918 - 0.333187*1j, 0.351195 + 0.292181*1j, 0.273443 - 0.326724*1j, 0.198048 + 0.338787*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j]

    A2=np.array([ [0.612903,-0.645161,0.483871,-0], 
    [0.54755,-0.240154,0.12488,-0], 
    [0.729167,-0,0.3125,5.20833], 
    [0.479452,-0,0.205479,-0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p2=[ 0.124, 0.1041, 0.0096, 0.146, 0.027 ]
    b2=[ -0.2541, 0, -0.1089, 0 ]
    Enc_B2 = [ 15,  13,  12,  11,  10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0] 
    G2 = [0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j, 0.829152 - 0.141498*1j, 0.0163725 - 0.122822*1j, 0.189147 + 0.0865223*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0359991 - 0.133128*1j]

    A3= np.array([ [0.612903,-0.645161,0.483871,0], 
    [0.519836,-0.0683995,-0.0273598,-0], 
    [0.560748,-0.934579,0.560748,-2.33645], 
    [0.731707,-1.21951,0.731707,0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p3=[ 0.62, 0.0731, 0.0214, 0.164, 0.027 ]
    b3=[ 0.4356, -0.726, 0.4356, 0 ]
    Enc_B3 = [ 2,  0,  1,  6,  7,  4,  5,  10,  11,  8,  9,  14,  15,  12,  13] 
    G3 = [0.0163725 - 0.122822*1j, 0.189147 + 0.0865223*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0359991 - 0.133128*1j, 0.158897 + 0.116359*1j, 0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j]

    A4=np.array([ [0.54755,-0.240154,0.12488,0], 
    [0.519836,-0.0683995,-0.0273598,0], 
    [0.529101,-0.26455,0.10582,-0.529101], 
    [0.558659,-0.27933,0.111732,0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p4=[ 2.082, 0.2924, 0.0945, 0.895, 0.027 ]
    b4=[ 1.815, -0.9075, 0.363, 0 ]
    Enc_B4 = [ 15,  14,  13,  12,  11,  10,  9,  8,  7,  6,  5,  4,  2,  1,  0] 
    G4 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j]

    A5=np.array([ [0.729167,0,0.3125,5.20833], 
    [0.560748,-0.934579,0.560748,-2.33645], 
    [0.529101,-0.26455,0.10582,-0.529101], 
    [-0,-0,-0,-10], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p5=[ 0.192, 0.0856, 0.0945, 0.05, 0.027 ]
    b5=[ 0, 0, 0, 1.815 ]
    Enc_B5 = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11,  10,  12,  15,  14] 
    G5 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j]

    new_B1,new_G1 = expand_BG_tables(m_i,Enc_B1,G1)
    new_B2,new_G2 = expand_BG_tables(m_i,Enc_B2,G2)
    new_B3,new_G3 = expand_BG_tables(m_i,Enc_B3,G3)
    new_B4,new_G4 = expand_BG_tables(m_i,Enc_B4,G4)
    new_B5,new_G5 = expand_BG_tables(m_i,Enc_B5,G5)
    Q = np.identity(n_dim)

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q,expnt_p = p1,enumeration_B=new_B1,enumeration_G=convertListGToTableG(new_G1,n_dim))  
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q,expnt_p = p2,enumeration_B=new_B2,enumeration_G=convertListGToTableG(new_G2,n_dim))  
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q,expnt_p = p3,enumeration_B=new_B3,enumeration_G=convertListGToTableG(new_G3,n_dim))  
    term4 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A4,expnt_b=b4,expnt_Q = Q,expnt_p = p4,enumeration_B=new_B4,enumeration_G=convertListGToTableG(new_G4,n_dim))  
    term5 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A5,expnt_b=b5,expnt_Q = Q,expnt_p = p5,enumeration_B=new_B5,enumeration_G=convertListGToTableG(new_G5,n_dim))  

    pdf = findSymbolic_UCPDF([term1,term2,term3,term4,term5])
    fz = calcfz([term1,term2,term3,term4,term5])

    print(fz)

    plot2d_from_4d(pdf,fz)



def test_coalignment():
    A = np.array([[1,1],[1,1],[1,-1]])
    b1 = [0.3,0]
    Q1 = np.identity(2)
    p = [0.1,0.2,0.3]
    G = [[['a']],[['b']],[['g']],[['h']]]
    B = [0,1,6,7]
    term1 = Term(parent=0,ndim=2,m_hyperplanes=3,A_hplane_arr=A,expnt_b=b1,expnt_Q = Q1,expnt_p = p,enumeration_B=B,enumeration_G=G )    

    coaligned_ind,p_adjusted = term1.coalignmentCheck()
    newTerm = term1.coalignmentAdjustment(coaligned_ind,p_adjusted)

    print(newTerm.enumeration_G)
    print(newTerm.A_hplane_arr)

def test_evalAtX():
    bundle1 = Bundle(1,3,[1,1])
    bundle2 = Bundle(-15,1,[0,0])
    bundle3 = Bundle(-1,3,[1,2])
    bundle4 = Bundle(1,1,[-1,1])

    pdf = [[bundle1,bundle2],[bundle3,bundle4]]
    evaluateAtX(pdf,[1,2])

def test_1D():
    Phi = 1
    Gam = 1
    H = 1
    
    alpha1 = 0.1
    gamma = 0.1
    beta = 0.1
    z1 = 0.1

    A1 = np.array([[-Phi],[Gam]])
    p1 = [gamma, beta]
    b1 = [Phi*z1];  
    Q1 = np.identity(1)
    B1 = [0,1,2,3]
    G1 = [
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]]
        ]
    
    A2 = np.array([[Phi],[Gam]])
    p2 = [alpha1,beta]
    b2 = [0]
    Q2 = np.identity(1)
    B2 = [0,1,2,3]
    G2 = [
        [[1/(2*pi) *(1/(1j*z1+gamma+alpha1) - 1/(1j*z1-gamma+alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma+alpha1) - 1/(1j*z1-gamma+alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma-alpha1) - 1/(1j*z1-gamma-alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma-alpha1) - 1/(1j*z1-gamma-alpha1))]]
        ]
    
    n_dim = 1
    m_i = 2


    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B1,enumeration_G=G1 )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B2,enumeration_G=G2)

    resultingint = findSymbolic_UCPDF([term1,term2])
    #print(resultingint)

    print_UCPDF(resultingint)

    # for i in range(len(resultingint)):
    #     print(f"product {i}")
    #     for j in range(len(resultingint[i])):
    #         print(f"num: {resultingint[i][j].const_num}, const: {resultingint[i][j].const_den}, coeff: {resultingint[i][j].coeff_den}")
    
def test_fz_1D():
    
    Phi = 1
    Gam = 1
    H = 1
    
    alpha1 = 0.1
    gamma = 0.1
    beta = 0.1
    z1 = 0.1

    n_dim = 1
    m_i = 2
    A1 = np.array([[-Phi],[Gam]])
    p1 = [gamma, beta]
    b1 = [Phi*z1];  
    Q1 = np.identity(1)
    B1 = [0,1,2,3]
    G1 = [
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]]
        ]

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B1,enumeration_G=convertConstGToBundle(G1,2) )    
    
    print(calcfz([term1]))
    print(G1[0][3])


def test_fz_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 0.1

    # term components
    G1 = [
      [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2+gamma)-1/(1j*z1-alpha_1+alpha_2+gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2-gamma)-1/(1j*z1-alpha_1+alpha_2-gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2+gamma)-1/(1j*z1-alpha_1-alpha_2+gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2-gamma)-1/(1j*z1-alpha_1-alpha_2-gamma))]] 
      ]
    A1 = np.array([[-1,0], [-1, 1]])
    b1 = [z1,0]
    p1 = [gamma,alpha_2]
    Q1 = np.identity(2)

    # child 2
    G2 = [
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1+gamma)-1/(1j*z1-alpha_2+alpha_1+gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1-gamma)-1/(1j*z1-alpha_2+alpha_1-gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1+gamma)-1/(1j*z1-alpha_2-alpha_1+gamma))]],
        [[ 1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1-gamma)-1/(1j*z1-alpha_2-alpha_1-gamma))]]
        ]
    A2 = np.array([[1,-1], [0, -1]])
    b2 = [0,z1]
    p2 = [alpha_1,gamma]
    Q2 = np.identity(2)

    G3 = [
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1+alpha_2)-1/(1j*z1-gamma+alpha_1+alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1-alpha_2)-1/(1j*z1-gamma+alpha_1-alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma-alpha_1+alpha_2)-1/(1j*z1-gamma-alpha_1+alpha_2))]],
        [[ 1/(2*pi)*(1/(1j*z1+gamma-alpha_1-alpha_2)-1/(1j*z1-gamma-alpha_1-alpha_2))]]
            ]
    A3 = np.array([[1,0], [0, 1]])
    b3 = [0,0]
    p3 = [alpha_1,alpha_2]
    Q3 = np.identity(2)

    B = [0,1,2,3]

    # define terms
    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))
    
    print(calcfz([term1,term2,term3]))


if __name__ == "__main__":
    #test_1D()
    #test_2d()
    test_4d()
    