import _enu
import numpy as np

def inc_enu(A, encode = False, sort = False):
    assert(A.ndim == 2)
    assert( not ((not encode) and sort) )
    m,n = A.shape
    _A = A.copy().reshape(-1)
    B = _enu.call_inc_enu(_A, m, n, encode, sort)
    if not encode:
        ccA = int(B.size / m)
        B = B.reshape((ccA, m))
    return B

def nat_enu(A, encode = False, sort = False):
    assert(A.ndim == 2)
    assert( not ((not encode) and sort) )
    m,n = A.shape
    # Dimensionality reduction of A
    mrA = np.linalg.matrix_rank(A)
    if mrA < n:
        print("Nat Enu: Projecting HPA from original dim={} to dim={} as its rank {}<{}".format(n,mrA,mrA,n))
        _,_,Vt = np.linalg.svd(A)
        T = Vt[:mrA,:].T
        proj_A = A @ T # project A down to subspace it spans with T 
        print(proj_A.shape)
        _A = proj_A.copy().reshape(-1)
        n = int(mrA)
    else:
        _A = A.copy().reshape(-1)
    B = _enu.call_nat_enu(_A, m, n, encode, sort)
    if not encode:
        ccA = int(B.size / m)
        B = B.reshape((ccA, m))
    return B

#'''
if __name__ == "__main__":
    A = np.array([ 
    [1,1,0,0],
    [1,2,0,0],
    [1,3,0,0],
    [1,4,0,0],
    [1,5,0,0],
    [1,1,1,0],
    [1,1,2,0],
    [1,1,3,0],
    [1,1,4,0],
    [1,1,5,0],
    [7,2,9,3.0] ])
    A2 = A[0:5,:]
    B = inc_enu(A2, True, True)
    B2 = nat_enu(A2, True, True)
    print("Same?:", np.all(B2 == B))

    A = np.array([ 
    [1,1,1,0],
    [1,1,1,2],
    [2,2,2,3],
    [3,3,3,5],
    [5,5,5,8],
    [1,6,0,0]])
    A2 = A#[0:5,:]
    B = inc_enu(A2, True, True)
    B2 = nat_enu(A2, True, True)
    print("Same?:", np.all(B2 == B))
#'''