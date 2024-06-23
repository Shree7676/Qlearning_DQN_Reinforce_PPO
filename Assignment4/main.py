# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from DQN_model import Qnet
from game_4_dqn import create_env
from utils import ReplayBuffer, train

# User definitions:
# -----------------
train_dqn = False
test_dqn = True
render = True

# Define env attributes (environment specific)
num_actions = 4
num_states = 4

# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 50_000
max_steps = 10_000

epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate

# Main:
# -----
if train_dqn:
    env = create_env()

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(num_actions=num_actions, num_states=num_states)
    q_target = Qnet(num_actions=num_actions, num_states=num_states)
    q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0

    # what is the parameters
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards = []
    goal = 0
    hell = 0

    for n_epi in range(num_episodes):
        if n_epi < 45_000:
            epsilon = epsilon_min + (1.0 - epsilon_min) * (1 - n_epi / num_episodes)
        else:
            epsilon = epsilon_min

        state_, _ = env.reset()
        done = False

        # Define maximum steps per episode, here 1,000
        for _ in range(max_steps):
            action_ = q_net.sample_action(torch.from_numpy(state_).float(), epsilon)
            s_prime, reward_, done, _ = env.step(action_)
            # print(s_prime, reward_)
            if render:
                env.render()

            done_mask = 0.0 if done else 1.0

            # Save the trajectories
            memory.put((state_, action_, reward_, s_prime, done_mask))
            state_ = s_prime

            episode_reward += reward_

            if done == "Goal":
                goal += 1
                break
            elif done == "Hell":
                hell += 1
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon} ##### goal={goal} and hell={hell}"
            )

        rewards.append(episode_reward)
        episode_reward = 0.0

        # Define a stopping condition for the game:
        if rewards[-10:] == [max_steps] * 10:
            break

    env.close()

    # Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn.pth")

    # Plot the training curve
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env()

    dqn = Qnet(num_actions=num_actions, num_states=num_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    goal = 0
    hell = 0

    for _ in range(1000):
        state_, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # Completely exploit
            action_ = dqn(torch.from_numpy(state_).float())
            s_prime, reward_, done_, _ = env.step(action_.argmax().item())
            s = s_prime
            if render:
                env.render()

            episode_reward += reward_
            if done_ == "Goal":
                goal += 1
                print(f"goal {goal} \t hell {hell}")
                break
            elif done_ == "Hell":
                hell += 1
                print(f"goal {goal} \t hell {hell}")
                break

        print(f"Episode reward: {episode_reward}")

    env.close()
