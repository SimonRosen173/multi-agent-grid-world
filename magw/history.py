from magw.utils.general import copy_to_dict
from magw.enums import Direction, Action

import cv2
import os

###########
# HISTORY #
###########
class EnvHistory:
    _DEFAULT_LOGGING_CONFIG = {
        "joint_action": True,
        "joint_state": True,
        "reward": True,
        "info": True,
        "is_done": True,
        "frames": False  # RGB frames
    }

    def __init__(self, n_agents, actions_type="cardinal", logging_config=None,
                 joint_start_state=None, episode_no=-1):
        self.n_agents = n_agents
        self.actions_type = actions_type
        self.episode_no = episode_no

        self.curr_step = 0

        self.joint_action_history = None
        self.joint_action_str_history = None

        self.joint_state_history = None
        self.reward_history = None
        self.info_history = None
        self.is_done_history = None
        self.frame_history = None

        self._init_dicts()

        self.eps_return = 0

        self.logging_config = copy_to_dict(logging_config, self._DEFAULT_LOGGING_CONFIG)

        if self.logging_config["joint_state"] and joint_start_state is not None:
            self.joint_state_history[0] = joint_start_state

        if actions_type != "cardinal":
            raise NotImplementedError("Only actions_type='cardinal' is supported right now.")

    def _init_dicts(self):
        self.joint_action_history = {"given": {}, "taken": {}}
        self.joint_action_str_history = {"given": {}, "taken": {}}

        self.joint_state_history = {}
        self.reward_history = {}
        self.info_history = {}
        self.is_done_history = {}
        self.frame_history = {}

    def step(self, joint_action_given=None, joint_action_taken=None, next_joint_state=None,
             reward=None, is_done=None, info=None):
        curr_step = self.curr_step

        if self.logging_config["joint_action"]:
            self.joint_action_history["given"][curr_step] = joint_action_given
            self.joint_action_history["taken"][curr_step] = joint_action_taken

            action_str_map = {
                Action.WAIT: "WAIT",
                Action.NORTH: "UP",
                Action.EAST: "RIGHT",
                Action.SOUTH: "DOWN",
                Action.WEST: "LEFT"
            }
            action_given_str = [action_str_map[action] for action in joint_action_given]
            self.joint_action_str_history["given"][curr_step] = action_given_str

            action_taken_str = [action_str_map[action] for action in joint_action_taken]
            self.joint_action_str_history["taken"][curr_step] = action_taken_str

        if self.logging_config["joint_state"]:
            self.joint_state_history[curr_step + 1] = next_joint_state

        if self.logging_config["reward"]:
            self.reward_history[curr_step] = reward
            self.eps_return += reward

        if self.logging_config["info"]:
            self.info_history[curr_step] = info

        if self.logging_config["is_done"] and is_done:  # Only logs when is_done
            self.is_done_history[curr_step] = is_done

        self.curr_step += 1

    def reset(self, episode_no, start_joint_state):
        self.curr_step = 0
        self.episode_no = episode_no

        self.eps_return = 0

        self._init_dicts()

        self.joint_state_history[0] = start_joint_state

    # Log RGB frames
    def log_frame(self, frame, step=None):
        if step is None:
            step = self.curr_step

        self.frame_history[step] = frame

    def to_dict(self):
        out_dict = {
            "episode": self.episode_no
        }

        if self.logging_config["joint_action"]:
            out_dict["joint_action"] = self.joint_action_history

        if self.logging_config["joint_state"]:
            out_dict["joint_state"] = self.joint_state_history

        if self.logging_config["reward"]:
            out_dict["reward"] = self.reward_history

        if self.logging_config["info"]:
            out_dict["info"] = self.info_history

        return out_dict

    def save_video(self, base_path=None, video_name=None, fps=30):
        if not self.logging_config["frames"]:
            raise Exception("save_video cannot be called if frames aren't being logged "
                            "(logging_config['frames']=False)")

        if video_name is None:
            video_name = f"video_{self.episode_no}"

        if base_path is None:
            file_path = video_name + ".mp4"
        else:
            file_path = base_path + os.path.sep + video_name + ".mp4"

        frame_hist = self.frame_history
        steps_frames = [(key, frame_hist[key]) for key in frame_hist.keys()]
        steps_frames.sort(key=lambda x: x[0])

        height, width, channels = steps_frames[0][1].shape

        fourcc = cv2.VideoWriter_fourcc(*'MP42')  # FourCC is a 4-byte code used to specify the video codec.
        video = cv2.VideoWriter(file_path, fourcc, float(fps), (width, height))

        for step, frame in steps_frames:
            frame = frame[:, :, ::-1]  # change from BGR to RBG image
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
        # video = cv2.VideoWriter(video_name+".mp4")