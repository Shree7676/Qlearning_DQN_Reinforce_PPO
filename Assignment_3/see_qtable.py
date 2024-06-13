import numpy as np

# Load the .npy file
q_table = np.load("q_table.npy")

# Flatten the array
q_table_flat = q_table.reshape(-1, q_table.shape[-1])

# Save the flattened array to a text file
np.savetxt("q_table.txt", q_table_flat)
