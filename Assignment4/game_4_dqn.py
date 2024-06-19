import copy
import os
import random
import tkinter as tk

import gymnasium as gym
import numpy as np
from gym import spaces
from PIL import Image, ImageTk

from A1_data import data_list, score_board


class space_war(gym.Env):
    def __init__(self, game_width=1400, game_height=700, space_size=70) -> None:
        super(space_war, self).__init__()
        self.game_width = game_width
        self.game_height = game_height
        self.space_size = space_size

        self.env_details = copy.deepcopy(data_list)
        self.score_board = copy.deepcopy(score_board)

        self.enemy_len = 0
        self.previous_score = 0

        self.agent_health = 0
        self.enemy_shoot_line = []
        self.done = False  # hitting terminal state

        self.temp_list = []

        self.window = tk.Tk()
        self.window.title("Space War")
        self.window.resizable(False, False)

        self.canvas = tk.Canvas(
            self.window, bg="black", height=self.game_height, width=self.game_width
        )
        self.canvas.grid(row=1, column=1)

        # Horizontal line
        self.canvas.create_line((0, 575), (1400, 575), width=2, fill="white")

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

        # displaying labels on screen
        self.label_list = []
        for key, value in self.score_board.items():
            text_content = key
            content_value = value
            if key == "Score":
                label = tk.Label(
                    self.window,
                    text=f"{text_content}:{content_value}",
                    font=("consolas", 20),
                )
                label.grid(row=0, column=1)
            else:
                label = tk.Label(
                    self.frame,
                    text=f"{text_content}:\n{content_value}",
                    font=("consolas", 15),
                    bg="white",
                )
                label.pack(padx=10, pady=30)
            self.label_list.append(label)

        # Construct the absolute path to the image file
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        hero = os.path.join(parent_dir, "Assets", "hero.png")
        enemy = os.path.join(parent_dir, "Assets", "alien.png")
        astro = os.path.join(parent_dir, "Assets", "astronaut.png")
        king = os.path.join(parent_dir, "Assets", "king.png")

        # loading images
        self.SPACE_SHIP = ImageTk.PhotoImage(Image.open(hero))
        self.ENEMY_SHIP = ImageTk.PhotoImage(Image.open(enemy))
        self.ASTRONAUT = ImageTk.PhotoImage(Image.open(astro))
        self.KING = ImageTk.PhotoImage(Image.open(king))

        self.image_load()

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.game_width, self.game_height),
            shape=(2,),
            dtype=np.int32,
        )

    def image_load(self):
        self.img_id = []
        for index in range(len(self.env_details)):
            x = self.env_details[index]["Position"][0]
            y = self.env_details[index]["Position"][1]
            if self.env_details[index]["Type"] == "Enemy_N":
                img_var = self.ENEMY_SHIP
                id = self.canvas.create_image(x - 32, y, image=img_var)
            elif self.env_details[index]["Type"] == "Hero":
                img_var = self.SPACE_SHIP
                id = self.canvas.create_image(x, y, image=img_var)
            elif self.env_details[index]["Type"] == "Friend":
                img_var = self.ASTRONAUT
                id = self.canvas.create_image(x - 32, y, image=img_var)
            elif self.env_details[index]["Type"] == "King":
                img_var = self.KING
                id = self.canvas.create_image(x, y, image=img_var)

            self.img_id.append(id)

    def reset(self):
        self.env_details = copy.deepcopy(data_list)
        self.score_board = copy.deepcopy(score_board)

        self.enemy_len = 0
        self.previous_score = 0

        self.agent_health = 0
        self.enemy_shoot_line = []
        self.done = False  # hitting terminal state

        self.temp_list = []

        self.canvas.delete("all")

        # displaying images on canvas
        self.image_load()
        self.update_score()

        # Horizontal line
        self.canvas.create_line((0, 575), (1400, 575), width=2, fill="white")

        return self.state_features(), self.env_details

    def enemy_status(self, agentX):
        for detail in self.env_details:
            if detail["Type"] != "Hero":
                enemyX = detail["Position"][0] - 35
                status = detail["Status"]
                if enemyX == agentX:
                    return 0 if status == "Alive" else 1
        return 1

    def step(self, action):
        self.erase_line()
        hero_x, hero_y = self.env_details[-1]["Position"]
        if action == 0:  # move left and shoot
            hero_x -= self.space_size
            self.shoot(hero_x, hero_y)
        elif action == 1:  # move right and shoot
            hero_x += self.space_size
            self.shoot(hero_x, hero_y)
        elif action == 2:  # just move left
            hero_x -= self.space_size
        elif action == 3:  # just move right
            hero_x += self.space_size
        elif action == 4:  # just shoot
            hero_x += self.space_size

        self.done = self.check_game_status()
        # if action == 3:
        #     if self.score_board["Score"] >= 100:
        #         self.done = "Goal"

        # updating the position of the spaceshp

        # To Ensure the spaceship stays within canvas bounds
        # hero_x = max(35, min(self.game_width - 35, hero_x))
        # teleporting feature
        if hero_x > self.game_width - 35:
            hero_x = 35
        elif hero_x < 35:
            hero_x = self.game_width - 35
        self.canvas.coords(self.img_id[-1], hero_x, hero_y)
        self.env_details[-1]["Position"] = (
            hero_x,
            hero_y,
        )
        self.enemy_shoot()
        self.update_score()

        # important
        # reward of each step has to send not cummulative reward

        reward = self.score_board["Score"] - self.previous_score
        self.previous_score = self.score_board["Score"]
        return (
            self.state_features(),
            reward,
            self.done,
            self.env_details,
        )

    def state_features(self):
        agentX = self.env_details[-1]["Position"][0]
        left_alive = 0
        right_alive = 0
        for dic in self.env_details:
            if dic["Status"] == "Alive" and dic["Type"] != "Hero":
                if dic["Position"][0] < agentX:
                    left_alive += 1
                elif dic["Position"][0] > agentX:
                    right_alive += 1

        position_x = int(self.env_details[-1]["Position"][0] / 70)
        enemy_status = self.enemy_status(self.env_details[-1]["Position"][0])
        state_value = np.array([position_x, enemy_status, left_alive, right_alive])

        return state_value

    def render(self):
        self.window.update()

    def close(self):
        self.window.destroy()

    def enemy_shoot(self):
        # erase line
        if self.enemy_shoot_line != []:
            oldX, oldY = self.enemy_shoot_line[0]
            self.canvas.create_line((oldX, oldY), (oldX, 700), width=2, fill="black")
            self.enemy_shoot_line = []

        if self.enemy_len >= len(self.env_details):
            self.enemy_len = 0
        flag = False
        # randomly select the pairs
        if random.randint(0, 2) == 0:
            if (
                self.env_details[self.enemy_len]["Type"] not in ["Friend", "Hero"]
                and self.env_details[self.enemy_len]["Status"] == "Alive"
            ):
                if self.env_details[self.enemy_len]["Type"] == "King":
                    flag = True
                else:
                    X, Y = self.env_details[self.enemy_len]["Position"]
                    self.canvas.create_line(
                        (X - 32, Y), (X - 32, 700), width=2, fill="blue"
                    )
                    self.enemy_shoot_line.append((X - 32, Y))
                    if self.env_details[-1]["Position"][0] in range(X - 70, X):
                        self.score_board["Self Hit"] -= 1
        else:
            index = len(self.env_details) - self.enemy_len - 1
            if (
                self.env_details[index]["Type"] not in ["Friend", "Hero"]
                and self.env_details[index]["Status"] == "Alive"
            ):
                if self.env_details[index]["Type"] == "King":
                    flag = True
                else:
                    X, Y = self.env_details[index]["Position"]
                    self.canvas.create_line(
                        (X - 32, Y), (X - 32, 700), width=2, fill="blue"
                    )
                    self.enemy_shoot_line.append((X - 32, Y))
                    if self.env_details[-1]["Position"][0] in range(X - 70, X):
                        self.score_board["Self Hit"] -= 1
        if flag:
            X, Y = self.env_details[-2]["Position"]
            self.canvas.create_line((X, Y), (X, 700), width=2, fill="blue")
            self.enemy_shoot_line.append((X, Y))
            if self.env_details[-1]["Position"][0] == X:
                self.score_board["Self Hit"] -= 1

        self.enemy_len += 1

    def erase_line(self):
        # erase line for previous shot
        if self.temp_list != []:
            agentCord, enemyCord = self.temp_list[0]
            self.canvas.create_line(agentCord, enemyCord, width=4, fill="black")
            self.temp_list = []

    def shoot(self, hero_x, hero_y):
        i = 0
        for detail in self.env_details:
            pos_x, pos_y = detail["Position"]
            status = detail["Status"]

            if pos_x - 35 == hero_x and status == "Alive":
                self.canvas.create_line(
                    (hero_x, hero_y), (pos_x - 35, pos_y), width=4, fill="red"
                )
                self.temp_list.append([(hero_x, hero_y), (pos_x - 35, pos_y)])
                self.env_details[i]["Status"] = "Dead"
                self.draw_cross_mark(pos_x, pos_y)
                if detail["Type"] == "Enemy_N":
                    self.score_board["Target Hit"] += 15
                elif detail["Type"] == "Friend":
                    self.score_board["Hostage Loss"] -= 10
                    break
            elif pos_x == hero_x and status == "Alive":  # king
                self.canvas.create_line(
                    (hero_x, hero_y), (pos_x, pos_y), width=4, fill="red"
                )
                if detail["Type"] == "King":
                    self.score_board["Target Hit"] += 100
                self.temp_list.append([(hero_x, hero_y), (pos_x, pos_y)])
                self.env_details[i]["Status"] = "Dead"
                self.draw_cross_mark(pos_x + 35, pos_y)

            i += 1
        else:
            self.score_board["Bullet Left"] -= 1
            self.score_board["Bullet Loss"] -= 1

    def draw_cross_mark(self, pos_x, pos_y):
        if pos_y == self.env_details[-1]["Position"][1]:
            pass
        else:
            self.canvas.create_line(
                (pos_x - 64, pos_y - 32),
                (pos_x, pos_y + 32),
                width=4,
                fill="red",
            )
            self.canvas.create_line(
                (pos_x - 64, pos_y + 32),
                (pos_x, pos_y - 32),
                width=4,
                fill="red",
            )

    def check_game_status(self):
        astro = 3
        total_enemy = 8
        for index in range(len(self.env_details)):
            if (
                self.env_details[index]["Type"] == "Enemy_N"
                and self.env_details[index]["Status"] == "Dead"
            ):
                total_enemy -= 1
            elif (
                self.env_details[index]["Type"] == "Friend"
                and self.env_details[index]["Status"] == "Dead"
            ):
                astro -= 1

        # Hell state
        if astro == 0 or self.score_board["Bullet Left"] == 0:
            return "Hell"
        # goal state
        elif self.score_board["Score"] > 130 or total_enemy == 0:
            return "Goal"
        return False

    def update_score(self):
        i = 0
        total_score = 0
        for key, value in self.score_board.items():
            if key == "Score":
                pass
            else:
                if key == "Bullet Left":
                    pass
                else:
                    total_score += value
                self.label_list[i].config(text=f"{key}:\n{value}")
            i += 1
        self.score_board["Score"] = total_score
        self.label_list[0].config(text=f"Score:{total_score}")


def create_env():
    env = space_war()
    return env


if __name__ == "__main__":
    env = space_war()
    for x in range(10):
        while True:
            action = input(
                "\n\nPlease select the action \n To move Left : 0 \n To move right : 1 \n To shoot : 2 "
            )
            while action not in ["0", "1", "2", "3"]:
                action = input("PLEASE ENTER VALID DATA")
            env_state, env_reward, env_done, env_info = env.step(int(action))
            env.render()
            if env_done:
                env.reset()
                break
    env.window.mainloop()
