import json
import os
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy

from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.evaluation.policy_wrapper import PolicyWrapperRobomimic
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

# from robomimic.scripts.config_gen.helper import get_r2d2_datasets


def eval_launcher(variant, run_id, exp_id):
    # Get Directory #
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Prepare Log Directory #
    variant["exp_name"] = os.path.join(variant["exp_name"], "run{0}/id{1}/".format(run_id, exp_id))
    log_dir = os.path.join(dir_path, "../../evaluation_logs", variant["exp_name"])

    # Set Random Seeds #
    torch.manual_seed(variant["seed"])
    np.random.seed(variant["seed"])

    # Set Compute Mode #
    use_gpu = variant.get("use_gpu", False)
    torch.device("cuda:0" if use_gpu else "cpu")

    """ old ckpts """
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-02-mt-bc-im128/ds_prior_r2d2-v1-im128_ncams_2_exclsamelyts_False_demoinput_False_langcond_True/2023-06-02-16-38-54/models/model_epoch_100.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-02-mt-bc-im128/ds_prior_r2d2-v1-im128_ncams_2_exclsamelyts_False_demoinput_True_langcond_False/2023-06-02-16-38-49/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-02-mt-bc-im128/ds_prior_r2d2-v1-im128_ncams_3_exclsamelyts_False_demoinput_False_langcond_True/2023-06-02-16-38-43/models/model_epoch_300.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-02-mt-bc-im128/ds_prior_r2d2-v1-im128_ncams_3_exclsamelyts_False_demoinput_True_langcond_False/2023-06-02-16-38-38/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-03-floor1/ds_prior_t2c-cans-floor1-im128_ncams_3_exclsamelyts_False_demoinput_False_langcond_True/2023-06-03-05-23-49/models/model_epoch_200.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_ret_pt/06-03-floor1/ds_prior_floor1-im128_ncams_3_exclsamelyts_False_demoinput_False_langcond_True/2023-06-03-05-23-56/models/model_epoch_100.pth"
    # ckpt_path = "~/expdata/r2d2/im/bc_xfmr/05-25-c2t-and-t2c-cans/ds_pnp-t2c-cans_predfutureacs_False/2023-05-25-22-34-36/models/model_epoch_200.pth"

    ckpt_path = variant["ckpt_path"]

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config = json.loads(ckpt_dict["config"])

    ### infer image size ###
    print(ckpt_dict["shape_metadata"]["all_shapes"].keys())
    imsize = 128 #ckpt_dict["shape_metadata"]["all_shapes"]["camera/image/varied_camera_1_left_image"][2]
    print(imsize)

    ckpt_dict["config"] = json.dumps(config)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    # determine the action space (relative or absolute)
    action_keys = config["train"]["action_keys"]["r2d2"]
    if "action/rel_pos" in action_keys:
        action_space = "cartesian_velocity"
        for k in action_keys:
            assert not k.startswith("action/abs_")
    elif "action/abs_pos" in action_keys:
        action_space = "cartesian_position"
        for k in action_keys:
            assert not k.startswith("action/rel_")
    else:
        raise ValueError

    # determine the action space for the gripper
    if "action/gripper_velocity" in action_keys:
        gripper_action_space = "velocity"
    elif "action/gripper_position" in action_keys:
        gripper_action_space = "position"
    else:
        raise ValueError

    # Prepare Policy Wrapper #
    # data_processing_kwargs = variant.get("data_processing_kwargs", {})
    data_processing_kwargs = dict(
        timestep_filtering_kwargs=dict(
            action_space=action_space, #"cartesian_velocity",
            gripper_action_space=gripper_action_space,
            robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
            camera_extrinsics=[],
        ),
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=True,
            # bgr_to_rgb=False, # very important!
            to_tensor=True,
            augment=False,
        ),
    )
    timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

    policy_data_processing_kwargs = {} #policy_variant.get("data_processing_kwargs", {})
    policy_timestep_filtering_kwargs = policy_data_processing_kwargs.get("timestep_filtering_kwargs", {})
    policy_image_transform_kwargs = policy_data_processing_kwargs.get("image_transform_kwargs", {})

    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)

    wrapped_policy = PolicyWrapperRobomimic(
        policy=policy,
        timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        image_transform_kwargs=policy_image_transform_kwargs,
        eval_mode=True,
    )

    # camera_kwargs = variant.get("camera_kwargs", {})
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    )
    
    policy_camera_kwargs = {} #policy_variant.get("camera_kwargs", {})
    policy_camera_kwargs.update(camera_kwargs)

    # print("sleeping...")
    # import time; time.sleep(100)

    env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )
    controller = VRPolicy()

    # Launch GUI #
    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=wrapped_policy,
        save_traj_dir=log_dir,
        save_data=variant.get("save_data", True),
    )
    RobotGUI(robot=data_collector)
