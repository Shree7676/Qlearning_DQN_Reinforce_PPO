import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Load the .npy file
q_table = np.load("q_table_V2.npy")

# Check the shape of the Q-table
print(q_table.shape)  # Should be (1, 20, 4) based on the error message

# Reshape the Q-table if needed
if q_table.shape[0] == 1:
    q_table = q_table.reshape((20, 4))

# Create the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    q_table,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=["Shoot & ML", "Shoot & MR", "Left", "Right"],
    yticklabels=[f"State {i+1}" for i in range(20)],
    cbar=False,
)

# Highlight the maximum value in each row
for i in range(q_table.shape[0]):
    j = np.argmax(q_table[i])
    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3))

plt.title("Version_2")
plt.xlabel("Actions")
plt.ylabel("States")
plt.show()
