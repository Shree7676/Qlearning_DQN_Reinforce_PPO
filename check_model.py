import numpy as np

q_table = np.load("q_table.npy")

# Flatten the array
q_table_flat = q_table.reshape(-1, q_table.shape[-1])

# Save the flattened array to a text file
np.savetxt("q_table.txt", q_table_flat)


def test_q_learning(
    env,
    no_episodes,
):
    goal = 0
    hell = 0
    for episode in range(no_episodes):
        state, _ = env.reset()

        total_reward = 0

        print(f"episode {episode},\t goal {goal} ,\t hell {hell}")
        while True:
            action = np.argmax(q_table[0][state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()

            total_reward += reward
            state = next_state

            if done == "Goal":
                goal += 1
                break
            if done == "Hell":
                hell += 1
                break
    env.close()
    print("Testing finished.\n")
