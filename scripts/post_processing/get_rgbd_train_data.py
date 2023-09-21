import glob
import json
import os
import kornia
import torch
import torch.nn.functional as tfn
import open3d as o3d

import cv2
import numpy as np
from tqdm import tqdm
import fnmatch
import h5py

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader

"""Code copied from:
efm: https://github.com/TRI-ML/efm/tree/3c164ab202878a06d130476c776af352c4736468/efm/models/depth
vidar: https://github.com/TRI-ML/efm/blob/3c164ab202878a06d130476c776af352c4736468/efm/models/depth/utils.py
"""

def is_tensor(data):
    return type(data) == torch.Tensor

def is_tuple(data):
    return isinstance(data, tuple)

def is_list(data):
    return isinstance(data, list) or isinstance(data, torch.nn.ModuleList)

def is_dict(data):
    return isinstance(data, dict) or isinstance(data, torch.nn.ModuleDict)

def is_seq(data):
    return is_tuple(data) or is_list(data)

def format_image(rgb):
    return torch.tensor(rgb.transpose(2,0,1)[None]).to(torch.float32).cuda() / 255.0

def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner


@iterate1
def interpolate(tensor, size, scale_factor, mode):
    if size is None and scale_factor is None:
        return tensor
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        recompute_scale_factor=False, mode=mode,
        align_corners=None,
    )


def resize_input(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor = None,
    resize: tuple = None
):
    """Resizes input data

    Args:
        rgb (torch.Tensor): input image (B,3,H,W)
        intrinsics (torch.Tensor): camera intrinsics (B,3,3)
        resize (tuple, optional): resize shape. Defaults to None.

    Returns:
        rgb: resized image (B,3,h,w)
        intrinsics: resized intrinsics (B,3,3)
    """
    # Don't resize if not requested
    if resize is None:
        if intrinsics is None:
            return rgb
        else:
            return rgb, intrinsics
    # Resize rgb
    orig_shape = [float(v) for v in rgb.shape[-2:]]
    rgb = interpolate(rgb, mode="bilinear", scale_factor=None, size=resize)
    # Return only rgb if there are no intrinsics
    if intrinsics is None:
        return rgb
    # Resize intrinsics
    shape = [float(v) for v in rgb.shape[-2:]]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= shape[1] / orig_shape[1]
    intrinsics[:, 1] *= shape[0] / orig_shape[0]
    # return resized input
    return rgb, intrinsics

class StereoModel(torch.nn.Module):
    """Learned Stereo model.

    Takes as input two images plus intrinsics and outputs a metrically scaled depth map.

    Taken from: https://github.com/ToyotaResearchInstitute/mmt_stereo_inference
    Paper here: https://arxiv.org/pdf/2109.11644.pdf
    Authors: Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, Max Bajracharya

    Pre-trained checkpoint here: s3://tri-ml-models/efm/depth/stereo.pt

    Args:
        cfg (Config): configuration file to initialize the model
        ckpt (str, optional): checkpoint path to load a pre-trained model. Defaults to None.
        baseline (float): Camera baseline. Defaults to 0.12 (ZED baseline)
    """

    def __init__(self, ckpt: str = None, baseline: float = 0.12):
        super().__init__()
        # Initialize model
        self.baseline = baseline
        self.model = torch.jit.load(ckpt).cuda()
        self.model.eval()

    def inference(
        self,
        rgb_left: torch.Tensor,
        rgb_right: torch.Tensor,
        intrinsics: torch.Tensor,
        resize: tuple = None,
    ):
        """Performs inference on input data

        Args:
            rgb_left (torch.Tensor): input float32 image (B,3,H,W)
            rgb_right (torch.Tensor): input float32 image (B,3,H,W)
            intrinsics (torch.Tensor): camera intrinsics (B,3,3)
            resize (tuple, optional): resize shape. Defaults to None.

        Returns:
            depth: output depth map (B,1,H,W)
        """
        rgb_left, intrinsics = resize_input(
            rgb=rgb_left, intrinsics=intrinsics, resize=resize
        )
        rgb_right = resize_input(rgb=rgb_right, resize=resize)

        with torch.no_grad():
            output, _ = self.model(rgb_left, rgb_right)

        disparity_sparse = output["disparity_sparse"]
        mask = disparity_sparse != 0
        depth = torch.zeros_like(disparity_sparse)
        depth[mask] = self.baseline * intrinsics[0, 0, 0] / disparity_sparse[mask]

        return depth, output["disparity"], disparity_sparse


# Get (RGBD_1, RGBD_2, RGBD_3) for a given trajectory in addition
# to camera intrinsics information
def get_rgbd_tuples(filepath, stereo_ckpt):
    svo_files = []
    for root, _, files in os.walk(filepath):
        for filename in files:
            if fnmatch.fnmatch(filename, "*.svo"):
                svo_files.append(os.path.join(root, filename))
    if len(svo_files) == 0:
        return [], [], [], [], [], [], []

    cameras = []
    frame_counts = []
    serial_numbers = []
    cam_matrices = []
    cam_distortions = []

    for svo_file in svo_files:
        # Open SVO Reader
        serial_number = svo_file.split("/")[-1][:-4]
        camera = SVOReader(svo_file, serial_number=serial_number)
        camera.set_reading_parameters(image=True, depth=True, pointcloud=False, concatenate_images=False)
        im_key = '%s_left' % serial_number
        # Intrinsics are the same for the left and the right camera
        cam_matrices.append(camera.get_camera_intrinsics()[im_key]['cameraMatrix'])
        cam_distortions.append(camera.get_camera_intrinsics()[im_key]['distCoeffs'])
        frame_count = camera.get_frame_count()
        cameras.append(camera)
        frame_counts.append(frame_count)
        serial_numbers.append(serial_number)

    # return serial_numbers
    cameras = [x for y, x in sorted(zip(serial_numbers, cameras))]
    cam_matrices = [x for y, x in sorted(zip(serial_numbers, cam_matrices))]
    cam_distortions = [x for y, x in sorted(zip(serial_numbers, cam_distortions))]
    serial_numbers = sorted(serial_numbers)

    # Make sure all cameras have the same framecounts
    assert frame_counts.count(frame_counts[0]) == len(frame_counts)
    rgb_im_traj = []
    zed_depth_im_traj = []
    tri_depth_im_traj = []
    for j in range(frame_counts[0]-1):
        print("J: ", j)
        if j > 1:
            break
        rgb_ims = []
        zed_depth_ims = []
        tri_depth_ims = []
        for i, camera in enumerate(cameras):
            output = camera.read_camera(return_timestamp=True)
            if output is None:
                break
            else:
                data_dict, timestamp = output
            im_key = '%s_left' % serial_numbers[i]
            im_key_right = '%s_right'  % serial_numbers[i]
            rgb_im = cv2.cvtColor(data_dict['image'][im_key], cv2.COLOR_BGRA2BGR)
            # cv2.imwrite("testing" + str(i) + "_rgb.jpg", rgb_im)
            zed_depth_im =  np.expand_dims(data_dict['depth'][im_key], -1)

            # Get TRI Depth
            left_rgb, right_rgb = data_dict['image'][im_key], data_dict['image'][im_key_right]
            intrinsics = np.array([cam_matrices[i]])
            model = StereoModel(stereo_ckpt)
            model.cuda()
            # Need the image sides to be divisible by 32.
            height, width, _ = left_rgb.shape
            height = height - height % 32
            width = width - width % 32
            left_rgb = left_rgb[:height, :width, :3]
            right_rgb = right_rgb[:height, :width, :3]

            tri_depth_im, disparity, disparity_sparse = model.inference(
                rgb_left=format_image(left_rgb),
                rgb_right=format_image(right_rgb),
                intrinsics=torch.tensor(intrinsics).to(torch.float32).cuda(), 
                resize=None,
            )
            tri_depth_im = tri_depth_im.cpu().detach().numpy()[0, 0]
            # print("TRI DEPTH: ", tri_depth_im.shape)
            # print("ZED DEPTH: ", zed_depth_im.shape)
            # cv2.imwrite("testing" + str(i) + "_depth.jpg", zed_depth_im)
            # cv2.imwrite("testing" + str(i) + "_tri_depth.jpg", tri_depth_im)

            rgb_ims.append(rgb_im)
            zed_depth_ims.append(zed_depth_im)
            tri_depth_ims.append(tri_depth_im)

        rgb_ims = np.array(rgb_ims)
        zed_depth_ims = np.array(zed_depth_ims)
        tri_depth_ims = np.array(tri_depth_ims)

        rgb_im_traj.append(rgb_ims)
        zed_depth_im_traj.append(zed_depth_ims)
        tri_depth_im_traj.append(tri_depth_ims)

    rgb_im_traj = np.array(rgb_im_traj)
    zed_depth_im_traj = np.array(zed_depth_im_traj)
    tri_depth_im_traj = np.array(tri_depth_im_traj)

    # # Close Everything #
    for camera in cameras:
        camera.disable_camera()

    return rgb_im_traj, zed_depth_im_traj, tri_depth_im_traj, serial_numbers, frame_counts[0], cam_matrices, cam_distortions

# Get camera extrinsics
def get_camera_extrinsics(filepath, serial_numbers, frame_count):
    filename = os.path.join(filepath, "trajectory.h5")
    # Get extrinsics for the trajectory
    with h5py.File(filename, "r") as f:
        extrinsics_trajs = []
        for _, serial_num in enumerate(serial_numbers):
            extrinsics_key = str(serial_num) + "_left"
            extrinsics_trajs.append(f['/observation/camera_extrinsics/' + extrinsics_key][:])
    combined_extrinsics = np.swapaxes(np.array(extrinsics_trajs), 0, 1)[:frame_count-1]
    return combined_extrinsics 

# Get actions
def get_actions(filepath, frame_count):
    filename = os.path.join(filepath, "trajectory.h5")
    with h5py.File(filename, "r") as f:
        cartesian_position = f["/action/cartesian_position"][:]
        gripper_position = f["/action/gripper_position"][:]
        gripper_position = np.expand_dims(gripper_position, -1)
        actions = np.concatenate((cartesian_position, gripper_position), axis=-1)[:frame_count-1]
    return actions
        

if __name__ == "__main__":
    r2d2_data_path = "/home/ashwinbalakrishna/Desktop/data/mixed_data"
    save_path = "/home/ashwinbalakrishna/Desktop/git-repos/r2d2"
    stereo_ckpt = "/home/ashwinbalakrishna/Desktop/git-repos/r2d2/stereo_20230724.pt"

    hf = h5py.File(os.path.join(save_path, 'rgbd_train_data.h5'), 'a')

    for i, traj_name in enumerate(os.listdir(r2d2_data_path)):
        print("I: ", i, "TRAJ NAME: ", traj_name)
        if i > 0: 
            break
        traj_group = hf.require_group(traj_name)
        traj_path = os.path.join(r2d2_data_path, traj_name)
        svo_path = os.path.join(traj_path, 'recordings/SVO')
        rgb_im_traj, zed_depth_im_traj, tri_depth_im_traj, serial_numbers, frame_count, cam_matrices, cam_distortions = get_rgbd_tuples(svo_path, stereo_ckpt)
        print("FRAME COUNT: ", frame_count)
        if not len(rgb_im_traj):
            continue
        combined_extrinsics = get_camera_extrinsics(traj_path, serial_numbers, frame_count)
        actions = get_actions(traj_path, frame_count)

        # assert(combined_extrinsics.shape[0] == rgb_im_traj.shape[0])
        # assert(actions.shape[0] == actions.shape[0])
        traj_group.create_dataset("rgb_im_traj", data=rgb_im_traj)
        traj_group.create_dataset("zed_depth_im_traj", data=zed_depth_im_traj)
        traj_group.create_dataset("tri_depth_im_traj", data=tri_depth_im_traj)
        traj_group.create_dataset("actions", data=actions)
        traj_group.create_dataset("extrinsics_traj", data=combined_extrinsics)
        traj_group.create_dataset("camera_matrices", data=cam_matrices)
        traj_group.create_dataset("camera_distortions", data=cam_distortions)

