import numpy as np
import scipy.io

num_steps = 10
W_shape = (2, 2)  # Replace with actual shape
V_shape = (1, 1)  # Replace with actual shape

# Set the seed
np.random.seed(0)

# Generate the random noise vectors for W and V
W_noise = np.array([np.random.multivariate_normal(np.zeros(W_shape[0]), np.eye(W_shape[0])) for _ in range(num_steps)])
V_noise = np.array([np.random.multivariate_normal(np.zeros(V_shape[0]), np.eye(V_shape[0])) for _ in range(num_steps)])

# Save to a .mat file for MATLAB to read
scipy.io.savemat('random_noise_vectors.mat', {'W_noise': W_noise, 'V_noise': V_noise})