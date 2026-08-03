import numpy as np

def mult_test():
    A1 = np.array([[-1,0],[-0.2,0.8]])
    A1 = np.array([[-0.5, 0],[-0.5,2]])
    A2 = np.array([[0.2,-0.8],[0,-1]])
    A3 = np.array([[1,0],[0,1]])

    e1 = np.array([1,0])
    e2 = np.array([0,1])

    Gamma = np.array([[0.1],[0.3]])
    Phi = np.array([[1.4,-0.6],[-0.2,1]])
    H = np.array([[2,0.5]])

    newA1 = TP(A1,Gamma,Phi)
    print(newA1)
    
    newA1A1,newA1A1_norm = MU(newA1,0,H)
    print(newA1A1)

    alpha1 = - np.inner(H,np.matmul(e1,Phi.T))
    alpha2 = np.inner(H,np.matmul(e2/np.inner(H,e2) - e1/np.inner(H,e1),Phi.T))
    alpha3 = np.inner(H,Gamma.T)

    second_coeff = 1/(alpha2 * np.inner(H,e1)) + 1/alpha1

    first_coeff = 1/(alpha2 * np.inner(H,e2))


    bottom_row = Gamma.T/alpha3 - np.matmul(e1,Phi.T)/alpha1
    mid_row = np.matmul((e2/np.inner(H,e2) - e1/np.inner(H,e1)),Phi.T) * 1/alpha2 - np.matmul(e1,Phi.T)/alpha1

    tp_top_row = np.matmul(-e1/np.inner(H,e1),Phi.T)
    tp_mid_row = np.matmul(e2/np.inner(H,e2),Phi.T) + np.matmul(-e1/np.inner(H,e1),Phi.T)
    tp_bot_row = Gamma.T[0,:]

    mu_mu_top_row = tp_top_row/np.inner(H,tp_top_row)
    mu_mu_mid_row = tp_mid_row/np.inner(H,tp_mid_row)
    mu_mu_bot_row = tp_bot_row/np.inner(H,tp_bot_row)

    mu_mid_row = mu_mu_mid_row - mu_mu_top_row
    mu_bot_row = mu_mu_bot_row - mu_mu_top_row
    alpha4 = 1/-0.7405

    coeff_left = alpha4/alpha1 - 1/(np.inner(H,e1)*alpha2) - 1/alpha1
    right_vec = alpha4*(Gamma.T/np.inner(H,Gamma.T) - np.matmul(e2,Phi.T)/(np.inner(H,e2)*alpha2))
    
    mu1 = np.matmul(e1,Phi.T)/np.inner(H,np.matmul(e1,Phi.T))
    mu2 = np.matmul(e2/np.inner(H,e2)-e1/np.inner(H,e1),Phi.T)/np.inner(H,np.matmul(e2/np.inner(H,e2)-e1/np.inner(H,e1),Phi.T))
    mu3 = Gamma.T/np.inner(H,Gamma.T)

    print(alpha3)




def TP(A,Gamma,Phi):
    newA = np.zeros((A.shape[0]+1,A.shape[1]))
    for row in range(A.shape[0]):
        new_row = np.matmul(A[row,:],Phi.T)
        newA[row,:] = new_row
    newA[-1,:] = Gamma.T
    return newA

def MU(A,t,H):
    mu = np.zeros((A.shape))
    for row in range(A.shape[0]):
        mu[row,:] = A[row,:] / np.inner(H,A[row,:])
    
    newA = np.array([-mu[0,:],mu[1,:]-mu[0,:],mu[2,:]-mu[0,:]])

    norm = np.linalg.norm(newA,axis=1,ord=1)
    A_normed = newA / norm[:,None]
    return newA,A_normed


if __name__ == "__main__":
    mult_test()