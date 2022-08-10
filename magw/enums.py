from enum import Enum, IntEnum


class Action(IntEnum):
    WAIT = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_UP = 4
    NORTH = 5
    EAST = 6
    SOUTH = 7
    WEST = 8


class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3