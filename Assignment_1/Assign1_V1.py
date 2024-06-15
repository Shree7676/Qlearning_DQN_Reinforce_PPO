import os
import tkinter as tk

import gymnasium as gym
import numpy as np
from gym import spaces
from PIL import Image, ImageTk


class SpaceWarEnv(gym.Env):
    def __init__(self, game_width=1400, game_height=700, space_size=70):
        super(SpaceWarEnv, self).__init__()
        self.game_width = game_width
        self.game_height = game_height
        self.space_size = space_size

        self.previous_direction = 0
        self.king_dead = False

        self.enemy_list = [
            (770, 35),
            (700, 300),
            (70, 350),
            (560, 500),
            (1050, 350),
            (560, 300),  # 560
            (1120, 150),
            (630, 350),
            (420, 500),
            (210, 150),
        ]

        self.astronaut_list = [
            (840, 450),
            (980, 35),
            (350, 400),
        ]
        self.king_cord = (1295, 100)

        self.bullet = 10_000

        self.space_cord_X = 735  # agent location x
        self.space_cord_Y = 650  # agent location y

        self.state_reward = 0
        # self.move = 0
        self.target_hit = 0  # +ve reward
        self.astronaut_hit = 0  # -ve reward
        self.waste_bullet = 0  # -ve reward
        self.dead_list = []  # to add dead enemies to mark cross symbol on that

        self.window = tk.Tk()  # creating window
        self.window.title("Space War")
        self.window.resizable(False, False)

        # Construct the absolute path to the image file
        current_dir = os.path.dirname(__file__)
        hero = os.path.join(current_dir, "Assets", "hero.png")
        enemy = os.path.join(current_dir, "Assets", "alien.png")
        astro = os.path.join(current_dir, "Assets", "astronaut.png")
        king = os.path.join(current_dir, "Assets", "king.png")

        # loading images
        self.SPACE_SHIP = ImageTk.PhotoImage(Image.open(hero))
        self.ENEMY_SHIP = ImageTk.PhotoImage(Image.open(enemy))
        self.ASTRONAUT = ImageTk.PhotoImage(Image.open(astro))
        self.KING = ImageTk.PhotoImage(Image.open(king))

        self.reward = 0  # sumation of all +ve and -ve rewards
        self.done = False  # to check if reached terminal state

        # creating label to display the values/points
        self.label = tk.Label(
            self.window, text="Score:{}".format(self.reward), font=("consolas", 40)
        )
        self.label.grid(row=0, column=1)

        # creating canvas == environment
        self.canvas = tk.Canvas(
            self.window, bg="black", height=self.game_height, width=self.game_width
        )
        self.canvas.grid(row=1, column=1)

        # displaying all the images in canvas
        self.space_id = self.canvas.create_image(
            self.space_cord_X, self.space_cord_Y, image=self.SPACE_SHIP
        )

        for n in range(len(self.enemy_list)):
            self.canvas.create_image(
                self.enemy_list[n][0] - 32, self.enemy_list[n][1], image=self.ENEMY_SHIP
            )

        for n in range(len(self.astronaut_list)):
            self.canvas.create_image(
                self.astronaut_list[n][0] - 32,
                self.astronaut_list[n][1],
                image=self.ASTRONAUT,
            )
        self.canvas.create_image(self.king_cord[0], self.king_cord[1], image=self.KING)

        # frame to see the dashboard / to see the score
        self.frame = tk.LabelFrame(
            self.window,
            text="Details",
            font=("consolas", 20),
            width=400,
            height=700,
            bg="white",
        )
        self.frame.grid(row=1, column=2, rowspan=2)
        self.frame.pack_propagate(False)

        self.label1 = tk.Label(
            self.frame,
            text="Target  points:\n{}".format(self.target_hit),
            font=("consolas", 20),
            bg="white",
        )
        self.label1.pack(padx=10, pady=30)

        self.label2 = tk.Label(
            self.frame,
            text="Astronaut  points:\n{}".format(self.astronaut_hit),
            font=("consolas", 20),
            bg="white",
        )
        self.label2.pack(padx=10, pady=30)

        self.label3 = tk.Label(
            self.frame,
            text="Bullet  points:\n{}".format(self.waste_bullet),
            font=("consolas", 20),
            bg="white",
        )
        self.label3.pack(padx=10, pady=30)

        self.label4 = tk.Label(
            self.frame,
            text="Bullet  left:\n{}".format(self.bullet),
            font=("consolas", 20),
            bg="white",
        )
        self.label4.pack(padx=10, pady=30)

        # Gymnasium spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=max(20, 1),
            shape=(1, 20),
            dtype=np.int32,
        )

    def reset(self):
        self.previous_direction = 0

        self.king_dead = False
        self.update_score()
        self.space_cord_X = 735
        self.space_cord_Y = 650
        # self.move = 0
        self.state_reward = 0
        self.target_hit = 0
        self.astronaut_hit = 0
        self.waste_bullet = 0
        self.bullet = 10_000
        self.king_cord = (1295, 100)
        self.dead_list = []
        self.done = False
        self.reward = 0
        self.enemy_list = [
            (770, 35),
            (700, 300),
            (70, 350),
            (560, 500),
            (1050, 350),
            (560, 300),
            (1120, 150),
            (630, 350),
            (420, 500),
            (210, 150),
        ]
        self.astronaut_list = [
            (840, 450),
            (980, 35),
            (350, 400),
        ]

        self.canvas.coords(self.space_id, self.space_cord_X, self.space_cord_Y)
        self.canvas.delete("all")
        self.update_score()
        self.frame.config(bg="white")

        # loading all the images
        for n in range(len(self.enemy_list)):
            self.canvas.create_image(
                self.enemy_list[n][0] - 32, self.enemy_list[n][1], image=self.ENEMY_SHIP
            )

        for n in range(len(self.astronaut_list)):
            self.canvas.create_image(
                self.astronaut_list[n][0] - 32,
                self.astronaut_list[n][1],
                image=self.ASTRONAUT,
            )
        self.space_id = self.canvas.create_image(
            self.space_cord_X, self.space_cord_Y, image=self.SPACE_SHIP
        )
        self.canvas.create_image(self.king_cord[0], self.king_cord[1], image=self.KING)

        return (
            int(self.space_cord_X / 70),
            # np.array([self.space_cord_X, self.space_cord_Y]),
            {},
        )  # returns present location of agent and env info(at begining no data is passed in info)

    def step(self, action):
        self.state_reward = 0
        if action == 0:  # shoot and Move left
            self.shoot((self.space_cord_X, self.space_cord_Y), self.enemy_list)
            self.move_spaceship("left")
        elif action == 1:  # shoot and Move right
            self.shoot((self.space_cord_X, self.space_cord_Y), self.enemy_list)
            self.move_spaceship("right")
        elif action == 2:  # Just Move left
            self.move_spaceship("left")
        elif action == 3:  # Just Move right
            self.move_spaceship("right")

        # going back in previous direction - ve reward
        if len(self.astronaut_list) == 0 or self.bullet == 0:
            self.frame.config(bg="red")
            self.done = "Hell"

        if len(self.enemy_list) == 0:
            self.frame.config(bg="green")
            self.done = "Goal"

        if self.king_dead == True and len(self.enemy_list) <= 7:
            self.frame.config(bg="green")
            self.done = "Goal"

        # state = np.array([self.space_cord_X, self.space_cord_Y])
        state = int(self.space_cord_X / 70)
        info = {"bullets_left": self.bullet}

        return state, self.state_reward, self.done, info

    def render(self):
        self.window.update()

    def close(self):
        self.window.destroy()

    # below is the helper function for step function
    def move_spaceship(self, direction):
        self.erase_line()
        if direction == "left":
            if self.previous_direction == "right":
                self.state_reward -= 5
            else:
                self.state_reward += 5
            self.space_cord_X -= self.space_size
            self.previous_direction = "left"
        elif direction == "right":
            if self.previous_direction == "left":
                self.state_reward -= 5
            else:
                self.state_reward += 5
            self.space_cord_X += self.space_size
            self.previous_direction = "right"

        # Ensure the spaceship stays within canvas bounds
        self.space_cord_X = max(35, min(self.game_width - 35, self.space_cord_X))
        self.canvas.coords(self.space_id, self.space_cord_X, self.space_cord_Y)

        self.update_score()

    def shoot(self, ship_cord, enemy_list):
        fired = False
        # check if shot enemy
        for enemy_cord_X, enemy_cord_Y in enemy_list:
            if enemy_cord_X - 35 == ship_cord[0]:
                self.waste_bullet += 5
                self.canvas.create_line(
                    ship_cord, (enemy_cord_X - 35, enemy_cord_Y), width=4, fill="red"
                )
                self.dead_list.append([ship_cord, (enemy_cord_X, enemy_cord_Y)])
                enemy_list.remove((enemy_cord_X, enemy_cord_Y))
                self.target_hit += 10
                self.state_reward += 10
                fired = True
                break
        # checks if shot friendly astronaut
        for aX, aY in self.astronaut_list:
            if aX - 35 == ship_cord[0]:
                self.canvas.create_line(ship_cord, (aX - 35, aY), width=4, fill="red")
                self.dead_list.append([ship_cord, (aX, aY)])
                self.astronaut_list.remove((aX, aY))
                self.astronaut_hit -= 30
                self.state_reward -= 30
                fired = True
                break
        # checks if king is dead which has high reward
        if not fired and not self.king_dead:
            if self.king_cord[0] == ship_cord[0]:
                self.canvas.create_line(
                    ship_cord,
                    (self.king_cord[0], self.king_cord[1]),
                    width=4,
                    fill="red",
                )
                self.target_hit += 50
                self.state_reward += 50
                self.king_dead = True
                self.dead_list.append([ship_cord, self.king_cord])

        if self.king_cord[0] == ship_cord[0] and not fired:
            self.cross(king_down=True)
        else:
            self.cross()

        self.waste_bullet -= 5
        self.bullet -= 1
        self.update_score()

    def update_score(self):
        # self.bullet *
        self.reward = self.target_hit + self.astronaut_hit
        self.label.config(text="Score:{}".format(self.reward))
        self.label1.config(text="Target points:\n{}".format(self.target_hit))
        self.label2.config(text="Astronaut points:\n{}".format(self.astronaut_hit))
        self.label3.config(text="Bullet points:\n{}".format(self.waste_bullet))
        self.label4.config(text="Bullet left:\n{}".format(self.bullet))

    # visually cross mark on the dead spaceship
    def cross(self, king_down=False):
        if self.dead_list:
            ship_cord, (enemy_cord_X, enemy_cord_Y) = self.dead_list[0]
            if len(self.dead_list) > 1:
                ship_cord, (enemy_cord_X, enemy_cord_Y) = self.dead_list[1]
            if king_down:
                self.canvas.create_line(
                    (enemy_cord_X - 35, enemy_cord_Y - 35),
                    (enemy_cord_X + 35, enemy_cord_Y + 35),
                    width=4,
                    fill="red",
                )
                self.canvas.create_line(
                    (enemy_cord_X - 35, enemy_cord_Y + 35),
                    (enemy_cord_X + 35, enemy_cord_Y - 35),
                    width=4,
                    fill="red",
                )
            else:
                self.canvas.create_line(
                    (enemy_cord_X - 70, enemy_cord_Y - 35),
                    (enemy_cord_X, enemy_cord_Y + 35),
                    width=4,
                    fill="red",
                )
                self.canvas.create_line(
                    (enemy_cord_X - 70, enemy_cord_Y + 35),
                    (enemy_cord_X, enemy_cord_Y - 35),
                    width=4,
                    fill="red",
                )

    def erase_line(self):
        if self.dead_list:
            if len(self.dead_list) > 1:
                ship_cord, (enemy_cord_X, enemy_cord_Y) = self.dead_list[1]
            else:
                ship_cord, (enemy_cord_X, enemy_cord_Y) = self.dead_list[0]
            if (enemy_cord_X, enemy_cord_Y) == self.king_cord:
                self.canvas.create_line(
                    ship_cord, (enemy_cord_X, enemy_cord_Y), width=4, fill="black"
                )
                self.dead_list.pop()
            else:
                self.canvas.create_line(
                    ship_cord, (enemy_cord_X - 35, enemy_cord_Y), width=4, fill="black"
                )
                self.dead_list.pop()


# for q learning (not neccessary)
def create_env():
    env = SpaceWarEnv()
    return env


# below code runs only if you directly run this file without importing it somewhere else
if __name__ == "__main__":
    env = SpaceWarEnv()
    for x in range(10):
        while True:
            action = int(
                input(
                    "\n\nPlease select the action \n To move Left : 0 \n To move right : 1 \n To shoot : 2 "
                )
            )
            while action not in [0, 1, 2, 3]:
                action = int(input("PLEASE ENTER VALID DATA"))
            env_state, env_reward, env_done, env_info = env.step(action)
            env.render()
            if env_done:
                env.reset()
                break
    env.window.mainloop()   env.window.mainloop()
