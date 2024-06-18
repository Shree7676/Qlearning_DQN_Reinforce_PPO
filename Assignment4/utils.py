# Imports:
# --------
import random
from collections import deque

import torch
import torch.nn.functional as F


# Repla Buffer:
# -------------
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


# Train function:
# ---------------
def train(q_net, q_target, memory, optimizer, batch_size, gamma):
    for _ in range(10):
        (
            s_tensor_lst,
            a_tensor_lst,
            r_tensor_lst,
            s_prime_tensor_lst,
            done_mask_tensor_lst,
        ) = memory.sample(batch_size)

        q_out = q_net(s_tensor_lst)  # q_net.forward(input_state)

        q_a = q_out.gather(1, a_tensor_lst)
        max_q_prime = q_target(s_prime_tensor_lst).max(1)[0].unsqueeze(1)
        target = r_tensor_lst + gamma * max_q_prime * done_mask_tensor_lst
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
