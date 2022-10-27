import os
from typing import Tuple
import pygame
from pygame import sprite


PATH_SEP = os.path.sep
BASE_PATH = os.path.split(os.path.abspath(__file__))[0]
ASSETS_PATH = os.path.join(BASE_PATH, 'assets')


class GoalSprite(sprite.Sprite):
    # TODO: Add images
    _GOAL_IMAGES = {
        "desirable": "blue_square.png",
        "undesirable": "beige_square.png",
    }

    def __init__(self, sprite_size: int, is_desirable: bool, pos: Tuple[int, int] = None):
        is_desirable_str = "desirable" if is_desirable else "undesirable"
        self.name = f"goal_{is_desirable_str}"
        self._is_desirable = is_desirable
        self._sprite_size = sprite_size
        sprite.Sprite.__init__(self)

        image = load_image(self._GOAL_IMAGES[is_desirable_str])
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()
        self.position = pos
        self.rect.topleft = calculate_topleft_position(self.position, self._sprite_size)

    def reset(self, position: Tuple[int, int]):
        self.position = position
        self.rect.topleft = calculate_topleft_position(position, self._sprite_size)


class AgentSprite(sprite.Sprite):
    _IMAGE_BASE_NAME = "character"  # TODO
    _COLORS = ["red", "blue", "green", "yellow"]

    def __init__(self, sprite_size: int, agent_no: int, start_pos: Tuple[int,int]):
        self.name = f"agent_{agent_no}"
        self._sprite_size = sprite_size
        self._agent_no = agent_no

        sprite.Sprite.__init__(self)
        assert agent_no < len(self._COLORS), f"agent_no too high (agent_no={agent_no}>={len(self._COLORS)})"
        img_name = f"{self._IMAGE_BASE_NAME}_{self._COLORS[agent_no]}.png"
        image = load_image(img_name)
        self.image = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.rect = self.image.get_rect()

        self.position = None
        self.update_pos(start_pos)

    def update_pos(self, pos):
        self.position = pos
        x, y = calculate_topleft_position(self.position, self._sprite_size)
        self.rect.update(x, y, self._sprite_size, self._sprite_size)
        # print(f"Agent {self._agent_no}: pos={pos}, x,y={(x,y)}")

        # self.rect.x, self.rect.y = _calculate_topleft_position(self.position, self._sprite_size)

    def reset(self, position: Tuple[int, int]):
        self.update_pos(position)
        # self.position = position
        # self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def step(self, move: Tuple[int, int]):
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        self.rect.topleft = calculate_topleft_position(self.position, self._sprite_size)


def load_image(name):
    img_path = os.path.join(ASSETS_PATH, name)
    try:
        image = pygame.image.load(img_path)
    except pygame.error:
        print('Cannot load image:', img_path)
        raise SystemExit()
    image = image.convert_alpha()
    return image


def calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]

