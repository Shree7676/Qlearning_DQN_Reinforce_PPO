import numpy as np


def train_q_learning(
    env,
    render,
    no_episodes,
    epsilon,
    epsilon_min,
    epsilon_decay,
    alpha,
    discount_factor,
    q_table_save_path="q_table.npy",
):
    q_table = np.zeros((1, 20, env.action_space.n))

    goal = 0
    hell = 0
    for episode in range(no_episodes):
        state, _ = env.reset()
        # print(state)

        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[0][state])  # Exploit

            # print(q_table)
            next_state, reward, done, _ = env.step(action)
            print(next_state, reward, done, _, end="\t")
            if render:
                env.render()

            total_reward += reward

            q_table[0][state][action] = q_table[0][state][action] + alpha * (
                reward
                + discount_factor * np.max(q_table[0][next_state])
                - q_table[0][state][action]
            )

            state = next_state

            if done == "Goal":
                goal += 1
                break
            elif done == "Hell":
                hell += 1
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"goal {goal}, hell {hell}", end="\t")

        print(f"Episode {episode + 1}: Total Reward: {total_reward}", end="\t")

    env.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")


#
# # Function 2: Visualize the Q-table
# # -----------
# def visualize_q_table(
#     hell_state_coordinates=[(2, 1), (0, 4)],
#     goal_coordinates=(4, 4),
#     actions=["Up", "Down", "Right", "Left"],
#     q_values_path="q_table.npy",
# ):
#     # Load the Q-table:
#     # -----------------
#     try:
#         q_table = np.load(q_values_path)
#
#         # Create subplots for each action:
#         # --------------------------------
#         _, axes = plt.subplots(1, 4, figsize=(20, 5))
#
#         for i, action in enumerate(actions):
#             ax = axes[i]
#             heatmap_data = q_table[:, :, i].copy()
#
#             # Mask the goal state's Q-value for visualization:
#             # ------------------------------------------------
#             mask = np.zeros_like(heatmap_data, dtype=bool)
#             mask[goal_coordinates] = True
#             mask[hell_state_coordinates[0]] = True
#             mask[hell_state_coordinates[1]] = True
#
#             sns.heatmap(
#                 heatmap_data,
#                 annot=True,
#                 fmt=".2f",
#                 cmap="viridis",
#                 ax=ax,
#                 cbar=False,
#                 mask=mask,
#                 annot_kws={"size": 9},
#             )
#
#             # Denote Goal and Hell states:
#             # ----------------------------
#             ax.text(
#                 goal_coordinates[1] + 0.5,
#                 goal_coordinates[0] + 0.5,
#                 "G",
#                 color="green",
#                 ha="center",
#                 va="center",
#                 weight="bold",
#                 fontsize=14,
#             )
#             ax.text(
#                 hell_state_coordinates[0][1] + 0.5,
#                 hell_state_coordinates[0][0] + 0.5,
#                 "H",
#                 color="red",
#                 ha="center",
#                 va="center",
#                 weight="bold",
#                 fontsize=14,
#             )
#             ax.text(
#                 hell_state_coordinates[1][1] + 0.5,
#                 hell_state_coordinates[1][0] + 0.5,
#                 "H",
#                 color="red",
#                 ha="center",
#                 va="center",
#                 weight="bold",
#                 fontsize=14,
#             )
#
#             ax.set_title(f"Action: {action}")
#
#         plt.tight_layout()
#         plt.show()
#
#     except FileNotFoundError:
#         print(
#             "No saved Q-table was found. Please train the Q-learning agent first or check your path."
#         )
#
