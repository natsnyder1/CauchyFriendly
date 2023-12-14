import numpy as np 
import os, sys
dir_enu = "/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/n_state_cuda5/scripts/pycauchy"
sys.path.append(dir_enu)
import enumeration as enu

def encode_b(b):
    enc_b = int(0)
    n = len(b)
    for i in range(n):
        if b[i] == -1:
            enc_b |= 1 << i
    return enc_b 

def remove_bit(b, bit):
    high = b >> (bit+1)
    low = b & ((1<<bit)-1)
    return (high << bit) | low

# This is the check of the integration formula for the measurement update step
def compare_mu_forms(mu, t):
    m,d = mu.shape
    assert(t < m)
    B_mu = enu.run_inc_enu_central(mu)
    At = np.array([ mu[i] - mu[t] if i != t else -mu[t] for i in range(m) ])
    B_At = enu.run_inc_enu_central(At)

    # Need to see if the t-column flip rule works for B_A
    enc_Bmu = [encode_b(b) for b in B_mu]
    set_enc_Bmu = set(enc_Bmu)
    # Store all rows of B_A which meet the t-column flip rule
    enc_Bmu_flip = []
    for enc_b in enc_Bmu:
        qenc_b = enc_b
        qenc_b ^= (1 << t)
        if qenc_b in set_enc_Bmu:
            enc_Bmu_flip.append( enc_b )
    set_enc_BA_flip = set(enc_Bmu_flip)
    # Create enc_BAt and set_enc_Bat
    enc_BAt = [encode_b(b) for b in B_At]
    set_enc_BAt = set(enc_BAt)
    # See if this set is equal to set_enc_BAt
    return set_enc_BAt == set_enc_BA_flip

# This is the check of the integration formula for the lowering step
def compare_lowered_forms(A, t):
    m,d = A.shape
    assert( t < m )
    B_A = enu.run_inc_enu_central(A) # B matrix of the parent arrangement A_i
    mu_denom = -A[:,-1]
    sgn_mu_denom = np.sign(mu_denom) # sign sequence of last dimension
    enc_sgn_mu_denom = encode_b( sgn_mu_denom )
    mu = A[:,:-1] / mu_denom.reshape((m,1)) 
    B_mu = enu.run_inc_enu_central(mu) 
    t_idxs = [i for i in range(m) if i != t]
    At = mu[t_idxs] - mu[t] # the m-1 x d-1 arrangement A_t
    B_At = enu.run_inc_enu_central(At) # B matrix of A_t
    
    # Need to see if the t-column flip rule works for B_A
    enc_BA = [encode_b(b) for b in B_A]
    set_enc_BA = set(enc_BA)
    # Store all rows of B_A which meet the t-column flip rule
    enc_BA_flip = []
    for enc_b in enc_BA:
        qenc_b = enc_b
        qenc_b ^= (1 << t)
        if qenc_b in set_enc_BA:
            enc_BA_flip.append( remove_bit(enc_b ^ enc_sgn_mu_denom, t) )
    set_enc_BA_flip = set(enc_BA_flip)
    # Create enc_BAt and set_enc_Bat
    enc_BAt = [encode_b(b) for b in B_At]
    set_enc_BAt = set(enc_BAt) 
    # See if this set is equal to set_enc_BAt
    return set_enc_BAt == set_enc_BA_flip


if __name__ == "__main__":
    # Some manual tests
    #A0 = np.array([
    #            [-1.08626927, -2.35122018, 0.37101918],
    #            [-1.94363186, 1.20159343, 1.0159622],
    #            [-1.74350472, -0.62333604, 0.68103499],
    #            [0.86240403, 0.17449126, 0.03934143]
    #            ])
    #A0 = np.array([[-0.80225246,  0.12980083,  1.72880519],
    #            [ 0.07555972, -1.08822043,  1.17110152],
    #            [-0.89682258, -0.68455098,  1.59735365],
    #            [-1.29077915,  0.95713706,  0.04679146],
    #            [-1.16817933, -1.55380787,  0.66786884]])
    #m = A0.shape[0]
    #d = A0.shape[1]
    #H = np.array([1,-0.5, 0.2])

    # Arbitrary Generation
    m = 6
    d = 3
    A0 = np.random.randn(m,d)
    H = np.random.randn(d)
    
    AH = A0 @ H
    mu = A0 / AH.reshape((A0.shape[0],1)) 
    print("Comparing B matrix generation of A_t (at k|k) from mu_i (of k|k-1)!")
    for t in range(m):
        print( "  t={} : {}".format(t+1, compare_mu_forms(mu, t)) ) 

    print("\nNOTE: The A_t's (at k|k) above are now refered to as A_i, as they ALL create lower dimentional children (now referred to at A_t)!\n")
    print("Comparing B matrix generation of (lowered dimensional) child A_t from parent A_i!\n")
    for i in range(m+1):
        print("Testing lowered children of parent A_i={}:".format(i+1))
        if i < m:
            Ai = np.array([ mu[j] - mu[i] if j != i else -mu[i] for j in range(m) ])
        else:
            Ai = mu.copy()
        ct = [ _ for _ in range(m) ]
        for t in ct:
            result = compare_lowered_forms(Ai, t=t)
            if result:
                print("  Child t={}, Results match".format(t+1))
            else:
                print("  Child t={}, Results do not match...error".format(t+1))