import numpy as np
import torch
from collections import deque
import time

from r2d2.data_processing.timestep_processing import TimestepProcesser
from robomimic.utils.lang_utils import LangEncoder
import cv2 

def converter_helper(data, batchify=True):
    if torch.is_tensor(data):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise ValueError

    if batchify:
        data = data.unsqueeze(0)
    return data


def np_dict_to_torch_dict(np_dict, batchify=True):
    torch_dict = {}

    for key in np_dict:
        curr_data = np_dict[key]
        if isinstance(curr_data, dict):
            torch_dict[key] = np_dict_to_torch_dict(curr_data)
        elif isinstance(curr_data, np.ndarray) or torch.is_tensor(curr_data):
            torch_dict[key] = converter_helper(curr_data, batchify=batchify)
        elif isinstance(curr_data, list):
            torch_dict[key] = [converter_helper(d, batchify=batchify) for d in curr_data]
        else:
            raise ValueError

    return torch_dict


class PolicyWrapper:
    def __init__(self, policy, timestep_filtering_kwargs, image_transform_kwargs, eval_mode=True):
        self.policy = policy

        if eval_mode:
            self.policy.eval()
        else:
            self.policy.train()

        self.timestep_processor = TimestepProcesser(
            ignore_action=True, **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )

    def forward(self, observation):
        timestep = {"observation": observation}
        processed_timestep = self.timestep_processor.forward(timestep)
        torch_timestep = np_dict_to_torch_dict(processed_timestep)
        action = self.policy(torch_timestep)[0]
        np_action = action.detach().numpy()

        # a_star = np.cumsum(processed_timestep['observation']['state']) / 7
        # print('Policy Action: ', np_action)
        # print('Expert Action: ', a_star)
        # print('Error: ', np.abs(a_star - np_action).mean())

        # import pdb; pdb.set_trace()
        return np_action
    

class PolicyWrapperRobomimic:
    def __init__(self, policy, timestep_filtering_kwargs, image_transform_kwargs, eval_mode=True):
        self.policy = policy

        assert eval_mode is True

        self.fs_wrapper = FrameStackWrapper(num_frames=10)
        self.fs_wrapper.reset()
        self.policy.start_episode()
        self.lang_encoder = LangEncoder(device=torch.device("cpu"))
        self.lang_emb = self.lang_encoder.get_lang_emb("pick the sponge from the counter and place it in the sink")

        # if eval_mode:
        #     self.policy.eval()
        # else:
        #     self.policy.train()

        self.timestep_processor = TimestepProcesser(
            ignore_action=True, **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )

        # import os
        # import imageio
        # video_path = os.path.expanduser("~/tmp/debug.mp4")
        # self.video_writer = imageio.get_writer(video_path, fps=5)

    def forward(self, observation):
        t1 = time.time()
        timestep = {"observation": observation}
        # print(type(observation))
        # print(observation["image"].keys())
        # print(observation["camera_type"])
        # cv2.imwrite("handcam0.jpg", observation["image"]["hand_camera"][0])
        # cv2.imwrite("handcam1.jpg", observation["image"]["hand_camera"][1])
        # cv2.imwrite("variedcam0.jpg", observation["image"]["varied_camera"][0])
        # cv2.imwrite("variedcam1.jpg", observation["image"]["varied_camera"][1])
        # cv2.imwrite("variedcam2.jpg", observation["image"]["varied_camera"][2])
        # cv2.imwrite("variedcam3.jpg", observation["image"]["varied_camera"][3])
        processed_timestep = self.timestep_processor.forward(timestep)
        # torch_timestep = np_dict_to_torch_dict(processed_timestep)
        # print("obs process time1:", time.time() - t1)

        t2 = time.time()

        wrist_camera = processed_timestep["observation"]["camera"]["image"]["hand_camera"][0] #observation["image"]["25047636_left"],
        #switch these back
        varied_cam_1_left = processed_timestep["observation"]["camera"]["image"]["varied_camera"][2] #observation["image"]["25047636_left"],
        varied_cam_2_left = processed_timestep["observation"]["camera"]["image"]["varied_camera"][0] #observation["image"]["25047636_left"],
        # print(type(im1))
        # print(im1)

        #varied camera 2 and 3 are the left cameras

        # cv2.imwrite("handcam0.jpg", processed_timestep["observation"]["camera"]["image"]["hand_camera"][0])
        # cv2.imwrite("handcam1.jpg", processed_timestep["observation"]["camera"]["image"]["hand_camera"][1])
        # cv2.imwrite("variedcam0.jpg", processed_timestep["observation"]["camera"]["image"]["varied_camera"][0])
        # cv2.imwrite("variedcam1.jpg", processed_timestep["observation"]["camera"]["image"]["varied_camera"][1])
        # cv2.imwrite("variedcam2.jpg", processed_timestep["observation"]["camera"]["image"]["varied_camera"][2])
        # cv2.imwrite("variedcam3.jpg", processed_timestep["observation"]["camera"]["image"]["varied_camera"][3])



        import robomimic.utils.torch_utils as TorchUtils
        cartesian_position = np.array(observation["robot_state"]["cartesian_position"])
        eef_pos = cartesian_position[0:3].astype(np.float64)
        eef_euler = cartesian_position[3:6].astype(np.float64)
        eef_euler = torch.from_numpy(eef_euler)
        eef_quat = TorchUtils.euler_angles_to_quat(eef_euler)
        eef_quat = eef_quat.numpy().astype(np.float64)

        
        obs = {
            "robot_state/cartesian_position": cartesian_position,
            "robot_state/eef_pos": eef_pos,
            "robot_state/eef_quat": eef_quat,
            "robot_state/gripper_position": np.array([observation["robot_state"]["gripper_position"]]),
            "camera/image/hand_camera_left_image": wrist_camera,
            "camera/image/varied_camera_2_left_image": varied_cam_2_left,
            "camera/image/varied_camera_1_left_image": varied_cam_1_left,
            "lang_emb": self.lang_emb 
        }

        print(np.linalg.norm(self.lang_emb))


        
        self.fs_wrapper.add_obs(obs)
        obs_history = self.fs_wrapper.get_obs_history()
        # print("obs process time2:", time.time() - t2)

        # obs_history = obs

        # for im in obs_history["camera/image/hand_camera_image"]:
        #     im = np.array(im)
        #     im = np.moveaxis(im, 0, -1)
        #     im = (im * 255).astype(np.uint8)
        #     self.video_writer.append_data(im)
        
        t1 = time.time()
        action = self.policy(obs_history)#[0]

        # clip action
        action = np.clip(action, a_min=-1, a_max=1)
        # print("run policy:", int(1000 * (time.time() - t1)))
        # print(action)
        # for i in range(3):
        #     action[i] *= 0.2
        # action[-1] = 0

        return action

    def reset(self):
        self.fs_wrapper.reset()
        self.policy.start_episode()
    

class FrameStackWrapper:
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        self.num_frames = num_frames

        ### TODO: add action padding option + adding action to obs to include action history in obs ###

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _set_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )

    def reset(self):
        self.obs_history = None

    def get_obs_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        if self.num_frames == 1:
            return { k : np.concatenate(self.obs_history[k], axis=0)[0] for k in self.obs_history }
        else:
            return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def add_obs(self, obs):
        if self.obs_history is None:
            self._set_initial_obs_history(obs)

        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
