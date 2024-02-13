from re import S
import numpy as np 
import matplotlib.pyplot as plt

zeros2 = np.zeros(2)
zeros1 = np.zeros(1)

xk_true = np.array([.3, .5])
xhat = np.array([0.2, 0.4])
P = np.array([0.50, 0.2, 0.2, 0.5]).reshape((2,2))
W = np.array([1.0, 0.5, 0.5, 1.0]).reshape((2,2))
V = np.eye(1) * 0.02
Phi = np.array([0.9, 0.1, -0.2, 1.1]).reshape((2,2))
H = np.array([1.0, 0.5])

xk1_true = Phi @ xk_true + np.random.multivariate_normal(zeros2, W)
zk1 = H @ xk1_true + np.random.multivariate_normal(zeros1, V)

xbar = Phi @ xhat
zbar = H @ Phi @ xhat 

# Form Joint
J = np.zeros((5,5))
J[0:2,0:2] = Phi @ P @ Phi.T + W 
J[2:4,0:2] = P @ Phi.T
J[4,  0:2] = H @ Phi @ P @ Phi.T  + H @ W

J[0:2,2:4] = J[2:4,0:2].T
J[2:4,2:4] = P
J[4,  2:4] = H @ Phi @ P

J[0:2, 4] = J[4,  0:2].T 
J[2:4, 4] = J[4,  2:4].T 
J[4,   4] = H @ Phi @ P @ Phi.T @ H.T + H @ W @ H.T + V

# Form Conditionals from Joint
_A = J[0:2,0:2]
_B = J[0:2,2:]
_D = J[2:,2:]
Cond_Mean = xbar + _B @ np.linalg.inv(_D) @ np.concatenate((np.zeros(2), zk1 - zbar))
Cond_Covar = _A - _B @ np.linalg.inv(_D) @ _B.T

# Berkely Notes -> But comes out identical to above
H = H.reshape((1,2))
from numpy.linalg import inv 
Sig = inv( inv(W) + H.T @ inv(V) @ H )
m = Sig @ (inv(W) @ xbar + H.T @ inv(V) @ zk1 )

pws = np.array([np.random.multivariate_normal(xbar, W) for _ in range(200)])
better_pws = np.array([np.random.multivariate_normal(Cond_Mean, Cond_Covar) for _ in range(200)])

plt.scatter(pws[:,0], pws[:,1], color='b')
plt.scatter(better_pws[:,0], better_pws[:,1], color='g')
plt.scatter(xk_true[0], xk_true[1], color='m', marker='*', s=100)
plt.scatter(xbar[0], xbar[1], color='m', marker='o', s=100)
plt.scatter(xk1_true[0], xk1_true[1], color='r', marker='*', s=100)
plt.scatter(Cond_Mean[0], Cond_Mean[1], color='r', marker='o', s=100)
plt.show()
foobar=2


