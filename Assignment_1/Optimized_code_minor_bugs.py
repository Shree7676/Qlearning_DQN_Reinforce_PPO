import tkinter as tk

import gymnasium as gym
import numpy as np
from A1_data import data_list, score_board
from gym import spaces
from PIL import Image, ImageTk


class space_war(gym.Env):
    def __init__(self, game_width=1400, game_height=700, space_size=70) -> None:
        super(space_war, self).__init__()
        self.game_width = game_width
        self.game_height = game_height
        self.space_size = space_size

        self.env_details = data_list
        self.score_board = score_board

        self.done = False  # hitting terminal state

        self.temp_list = []

        self.window = tk.Tk()
        self.window.title("Space War")
        self.window.resizable(False, False)

        self.canvas = tk.Canvas(
            self.window, bg="black", height=self.game_height, width=self.game_width
        )
        self.canvas.grid(row=1, column=1)

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
                    text=f"{text_content}:\n{content_value}",
                    font=("consolas", 20),
                    bg="white",
                )
                label.grid(row=0, column=1)
            else:
                label = tk.Label(
                    self.frame,
                    text=f"{text_content}:\n{content_value}",
                    font=("consolas", 20),
                    bg="white",
                )
                label.pack(padx=10, pady=30)
            self.label_list.append(label)

        # displaying images on canvas

        self.SPACE_SHIP = ImageTk.PhotoImage(Image.open("./Assets/hero.png"))
        self.ENEMY_SHIP = ImageTk.PhotoImage(Image.open("./Assets/alien.png"))
        self.ASTRONAUT = ImageTk.PhotoImage(Image.open("./Assets/astronaut.png"))
        self.KING = ImageTk.PhotoImage(Image.open("./Assets/king.png"))

        self.image_load()

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(4)
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
        self.env_details = data_list
        self.score_board = score_board

        self.done = False  # hitting terminal state

        self.canvas.delete("all")

        # displaying images on canvas
        self.image_load()
        self.update_score()

        return self.env_details[-1]["Position"], self.env_details

    def step(self, action):
        hero_x, hero_y = self.env_details[-1]["Position"]
        if action == 0:
            hero_x -= self.space_size
        elif action == 1:
            hero_x += self.space_size
        elif action == 2:
            self.shoot(hero_x, hero_y)
        elif action == 3:
            if self.score_board["Target Hit"] >= 70:
                self.done = True

        self.done = self.check_game_status()
        # updating the position of the spaceshp

        # To Ensure the spaceship stays within canvas bounds
        hero_x = max(35, min(self.game_width - 35, hero_x))
        self.canvas.coords(self.img_id[-1], hero_x, hero_y)
        self.env_details[-1]["Position"] = (
            hero_x,
            hero_y,
        )

        return (
            self.env_details[-1]["Position"],
            self.score_board["Score"],
            self.done,
            self.env_details,
        )

    def render(self):
        self.window.update()

    def close(self):
        self.window.destroy()

    def shoot(self, hero_x, hero_y):
        bullet_loss, bullet_left = 0, 0

        for detail in self.env_details:
            pos_x, pos_y = detail["Position"]
            status = detail["Status"]

            if pos_x - 35 == hero_x and status == "Alive":
                self.canvas.create_line(
                    (hero_x, hero_y), (pos_x - 35, pos_y), width=4, fill="red"
                )
                detail["Status"] = "Dead"
                bullet_left -= 1

                if detail["Type"] == "Enemy_N":
                    self.score_board["Target Hit"] += 10
                elif detail["Type"] == "Friend":
                    self.score_board["Hostage Loss"] -= 20
                elif detail["Type"] == "King":
                    self.score_board["Target Hit"] += 100

                self.draw_cross_mark(pos_x, pos_y)
                self.temp_list.append([(hero_x, hero_y), (pos_x, pos_y)])
            else:
                bullet_loss -= 5
                bullet_left -= 1

        self.score_board["Bullet Loss"] += bullet_loss
        self.score_board["Bullet Left"] += bullet_left
        self.update_score()

    def draw_cross_mark(self, pos_x, pos_y):
        offset = 35
        self.canvas.create_line(
            (pos_x - offset, pos_y - offset),
            (pos_x + offset, pos_y + offset),
            width=4,
            fill="red",
        )
        self.canvas.create_line(
            (pos_x - offset, pos_y + offset),
            (pos_x + offset, pos_y - offset),
            width=4,
            fill="red",
        )

    def check_game_status(self):
        astro = 3
        total_enemy = 10
        for index in range(len(self.env_details)):
            if (
                self.env_details[index]["Type"] == "Enemy_N"
                and self.env_details[index]["Status"] == "Alive"
            ):
                total_enemy -= 1
            elif (
                self.env_details[index]["Type"] == "Friend"
                and self.env_details[index]["Status"] == "Dead"
            ):
                astro -= 1

        if astro == 0 or self.score_board["Bullet Left"] == 0:
            return True
        elif self.score_board["Score"] > 700 or total_enemy == 0:
            return True

        return False

    def update_score(self):
        i = 0
        for key, value in self.score_board.items():
            if key == "Score":
                self.label_list[i].config(text=f"{key}:{value}")
            else:
                self.label_list[i].config(text=f"{key}:\n{value}")

    def erase_line(self):
        for hero, space in self.temp_list:
            self.canvas.create_line(
                hero, (space[0] - 35, space[1]), width=4, fill="black"
            )
            self.temp_list.remove([hero, space])


if __name__ == "__main__":
    env = space_war()
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
    env.window.mainloop()
