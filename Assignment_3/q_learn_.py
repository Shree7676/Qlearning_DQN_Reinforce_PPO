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
            # print(next_state, reward, done, _, end="\t")
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

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")
