import numpy as np
from scipy.stats import cauchy 
import matplotlib.pyplot as plt

def advance_simulation():
    pass

def define_simulation():
    phi = 1
    B = 1
    gamma = 1
    H = 1

    alpha = 0.1 # 
    gamma = 0.1 # meas noise
    beta = 0.1 # process noise

    P = 1 # Weighting constant in Lyapunov function 

    x0 = 0 
    x = np.linspace(-10,10,1000)
    pdf_val = cauchy.pdf(x,loc=x0,scale=alpha)
    x_init = cauchy.rvs(loc=x0,scale=alpha,size=1)

    plt.plot(x,pdf_val,color = 'blue')
    plt.plot(x,cauchy.pdf(x,loc=x0,scale=0.4),color ='red')
    plt.xlabel('x')
    plt.ylabel('Probability Density')

    plt.show()




if __name__ == "__main__":
    define_simulation()