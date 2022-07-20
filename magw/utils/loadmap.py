import numpy as np
from typing import List, Tuple, Optional, Dict, Union, IO
import os


PATH_SEP = os.path.sep
MAPS_PATH = os.path.abspath(__file__).split(PATH_SEP)[:-2]
MAPS_PATH.append("maps")
MAPS_PATH = PATH_SEP.join(MAPS_PATH)


def load(grid, grid_input_type):
    if grid_input_type == "map_name":
        return _load_from_txt(grid + ".txt")
    elif grid_input_type == "file_path":
        return _load_from_txt(grid, in_maps_folder=False)


def _load_from_txt(file_name, in_maps_folder=True):
    if in_maps_folder:
        file_name = MAPS_PATH + PATH_SEP + file_name
    with open(file_name) as f:
        grid_arr = []
        for line in f:
            curr_arr = line.split(" ")
            curr_arr = [int(el) for el in curr_arr]
            grid_arr.append(curr_arr)

        grid = np.asarray(grid_arr)

    return grid


if __name__ == "__main__":
    # print(load("corridors_alt", grid_input_type="map_name"))
    pass
