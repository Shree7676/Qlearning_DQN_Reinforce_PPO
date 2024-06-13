# Imports:
# --------
from Assignment_1.Assignment1_ import create_env
from Assignment_3.q_learn_ import train_q_learning

# User definitions:
# -----------------
train = True
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1_000  # Number of episodes

# goal_coordinates = (4, 4)
# Define all hell state coordinates as a tuple within a list
# hell_state_coordinates = [(2, 1), (0, 4)]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(
        env=env,
        no_episodes=no_episodes,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=learning_rate,
        gamma=gamma,
    )

# if visualize_results:
#     # Visualize the Q-table:
#     # ----------------------
#     visualize_q_table(
#         hell_state_coordinates=hell_state_coordinates,
#         goal_coordinates=goal_coordinates,
#         q_values_path="q_table.npy",
#     )
