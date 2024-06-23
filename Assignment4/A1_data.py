import random

import numpy as np

possible_values_heroX = np.arange(35, 1366, 70)
heroX = possible_values_heroX[random.randint(0, len(possible_values_heroX) - 1)]
data_list = [
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (770, 35),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (700, 300),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (70, 350),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (1050, 350),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (560, 300),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (1120, 150),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (630, 350),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Enemy_N",
        "Reward": 10,
        "Position": (210, 150),
        "Status": "Alive",
        "Image": "alien.png",
    },
    {
        "Type": "Friend",
        "Reward": -30,
        "Position": (1330, 450),
        "Status": "Alive",
        "Image": "astronaut.png",
    },
    {
        "Type": "Friend",
        "Reward": -30,
        "Position": (980, 35),
        "Status": "Alive",
        "Image": "astronaut.png",
    },
    {
        "Type": "Friend",
        "Reward": -30,
        "Position": (350, 400),
        "Status": "Alive",
        "Image": "astronaut.png",
    },
    {
        "Type": "King",
        "Reward": 50,
        "Position": (1295, 100),
        "Status": "Alive",
        "Image": "king.png",
    },
    {
        "Type": "Hero",
        "Reward": 50,
        "Position": (heroX, 650),
        "Status": "Alive",
        "Image": "hero.png",
    },
]

score_board = {
    "Score": 0,
    "Target Hit": 0,
    "Hostage Loss": 0,
    "Self Hit": 0,
    "Bullet Loss": 0,
    "Bullet Left": 1000,
}
