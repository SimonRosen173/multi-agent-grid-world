import json
import os

PATH_SEP = os.path.sep
MAPS_PATH = os.path.abspath(__file__).split(PATH_SEP)[:-2]
MAPS_PATH.append("maps")
MAPS_PATH = PATH_SEP.join(MAPS_PATH)

def load_config(file_path=None, file_name=None):
    assert not (file_path is None and file_name is None), "Either file_path or file_name must be set"

    if file_path is None:
        file_path = os.path.join(MAPS_PATH, file_name)
        # file_path = f"{MAPS_PATH}\\{file_name}"

    with open(file_path) as f:
        json_data = json.load(f)

        # named_goals = json_data["named_goals"]
        # named_goals = {key: tuple(val) for (key, val) in named_goals.items()}
        #
        # start_locs = json_data["start_locs"]
        # start_locs = [[tuple(loc) for loc in joint_start] for joint_start in start_locs]

        config = {
            "goals": [tuple(goal) for goal in json_data["goals"]],
            "named_goals": {key: tuple(val) for (key, val) in json_data["named_goals"].items()},
            "start_locs": [tuple(loc) for loc in json_data["start_locs"]],
            "named_start_locs": {key: tuple(val) for (key, val) in json_data["named_start_locs"].items()},
        }

        return config


def main():

    config = load_config(f"{MAPS_PATH}\\corridors.config")
    print(config)


if __name__ == "__main__":
    main()
