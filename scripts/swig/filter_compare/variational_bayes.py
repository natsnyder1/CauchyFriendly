import numpy as np 


class VBayes():

    def __init__(self, x0, P0, f, F, h, H, v, u, Qhat = 10*np.eye(4), V = 100 * np.eye(4), rho_R = 1-np.ep(-4), tau_P = 3, is_lin = True):
        self.f = f 
        self.F = F 
        self.h = h 
        self.H = H 
        
        self.x = x0.copy() 
        self.P = P0.copy()
        self.v = v.copy()
        self.u = u.copy()
        self.Qhat = Qhat.copy()
        self.V = V.copy()
        self.rho_R = rho_R
        self.tau_P = tau_P
        self.is_lin = is_lin
        self.eps = 1e-5

    # Time Prop
    def pxk_g_ykm1(self):
        Phi = self.F(self.x)
        self.x = self.f(self.x)
        self.P = Phi @ self.P @ Phi + self.Qhat
        self.u = self.n + self.tau_P + 1
        self.U = self.tau_P * self.P
        self.v = self.rho_R*(self.v - self.m - 1) + self.m + 1 
        self.V = self.rho_R*self.V 

    def lin_nat_param_update(self, yk):
        # Get lam_R 
        H = self.H(self.x) 

        it = 0
        xbar = self.x.copy()
        vbar = self.v.copy()
        ubar = self.u.copy()
        Ubar = self.U.copy()
        Vbar = self.V.copy()

        for it in range(6):
            EPkk_I = (self.u - self.n - 1) * np.linalg.inv(self.U)
            ERk_I = (self.v - self.m - 1) * np.linalg.inv(self.V)
            # Form \lambda^x_k(1) and \lambda^x_k(2)
            lxk1 = EPkk_I @ self.x + H.T @ ERk_I @ yk 
            lxk2 = -0.5 * EPkk_I - 0.5 * H.T @ ERk_I @ H
            # recover Pkk and xkk 
            self.P = np.linalg.inv(lxk2 / -0.5)
            self.x = self.P @ lxk1 
            # Form \lambda^P_k(1) and \lambda^P_k(2)
            ex = (self.x - xbar).reshape((self.x.size, 1))
            Ck = ex @ ex.T + self.P 
            lPk1 = -0.5 * (ubar + self.n + 2)
            lPk2 = -0.5 * (Ubar + Ck)
            # recover uk and Uk 
            self.u = lPk1 /-0.5 - self.n - 1
            self.U = lPk2 / -0.5 
            # Form \lambda^R_k(1) and \lambda^R_k(2)
            resid = (yk - H @ self.x ).reshape((yk.size,1))
            Ak = resid @ resid.T + H @ self.P @ H.T
            lRk1 = -0.5 * (vbar + self.m + 2)
            lRk2 = -0.5 * (Vbar + Ak)
            # recover 
            self.v = lRk1 / -0.5 - self.m - 1
            self.V = lRk2 / -0.5 

    def nonlin_nat_param_update(self): 
        pass
    
    def step(self, yk, with_tp = True):
        # Time Prop
        if with_tp:
            self.pxk_g_ykm1()
        if self.is_lin:
            self.lin_nat_param_update(self, yk)
        else:
            self.nonlin_nat_param_update(self, yk)