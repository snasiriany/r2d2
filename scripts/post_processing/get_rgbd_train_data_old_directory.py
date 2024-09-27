import glob
import json
import os
import torch
import torch.nn.functional as tfn
import open3d as o3d
import random
import time
import pickle

import cv2
import numpy as np
import fnmatch
import h5py

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
torch._C._jit_set_profiling_executor(False)

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
    return torch.tensor(rgb.transpose(0,3,1,2)).to(torch.float32).cuda() / 255.0

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

    def __init__(self, ckpt: str = None):
        super().__init__()
        # Initialize model
        self.model = torch.jit.load(ckpt).cuda()
        self.model.eval()

    def inference(
        self,
        rgb_left: torch.Tensor,
        rgb_right: torch.Tensor,
        intrinsics: torch.Tensor,
        resize: tuple = None,
        baseline: float = 0.12
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
        depth[mask] = baseline * intrinsics[0, 0, 0] / disparity_sparse[mask]

        return depth, output["disparity"], disparity_sparse, rgb_left, rgb_right, intrinsics

# Get (RGBD_1, RGBD_2, RGBD_3) for a given trajectory in addition
# to camera intrinsics information
def get_rgbd_tuples(filepath, stereo_ckpt, batch_size=16):
    svo_files = []
    for root, _, files in os.walk(filepath):
        for filename in files:
            if fnmatch.fnmatch(filename, "*.svo"):
                svo_files.append(os.path.join(root, filename))
    if len(svo_files) == 0:
        return [], [], [], [], [], []

    cameras = []
    frame_counts = []
    serial_numbers = []
    cam_matrices = []

    for svo_file in svo_files:
        # Open SVO Reader
        serial_number = svo_file.split("/")[-1][:-4]
        camera = SVOReader(svo_file, serial_number=serial_number)
        camera.set_reading_parameters(image=True, depth=True, pointcloud=False, concatenate_images=False)
        im_key = '%s_left' % serial_number
        # Intrinsics are the same for the left and the right camera
        cam_matrices.append(camera.get_camera_intrinsics()[im_key]['cameraMatrix'])
        frame_count = camera.get_frame_count()
        cameras.append(camera)
        frame_counts.append(frame_count)
        serial_numbers.append(serial_number)

    # return serial_numbers
    cameras = [x for y, x in sorted(zip(serial_numbers, cameras))]
    cam_matrices = [x for y, x in sorted(zip(serial_numbers, cam_matrices))]
    serial_numbers = sorted(serial_numbers)

    assert frame_counts.count(frame_counts[0]) == len(frame_counts)
    
    left_rgb_im_traj = []
    right_rgb_im_traj = []
    tri_depth_im_traj = []

    for i, camera in enumerate(cameras):
        cam_left_rgb_im_traj = []
        cam_right_rgb_im_traj = []
        intrinsics =  np.repeat(cam_matrices[i][np.newaxis, :, :], frame_counts[0] - 1, axis=0)
        for j in range(frame_counts[0] - 1):
            output = camera.read_camera(return_timestamp=True)
            if output is None:
                break
            else:
                data_dict, timestamp = output
            im_key_left = '%s_left' % serial_numbers[i]
            im_key_right = '%s_right'  % serial_numbers[i]

            rgb_im_left = cv2.cvtColor(data_dict['image'][im_key_left], cv2.COLOR_BGRA2BGR)
            rgb_im_right = cv2.cvtColor(data_dict['image'][im_key_right], cv2.COLOR_BGRA2BGR)

            cam_left_rgb_im_traj.append(rgb_im_left)
            cam_right_rgb_im_traj.append(rgb_im_right)
        cam_left_rgb_im_traj = np.array(cam_left_rgb_im_traj)
        cam_right_rgb_im_traj = np.array(cam_right_rgb_im_traj)

        model = StereoModel(stereo_ckpt)
        model.cuda()
        # Need the image sides to be divisible by 32.
        _, height, width, _ = cam_left_rgb_im_traj.shape
        height = height - height % 32
        width = width - width % 32
        cam_left_rgb_im_traj = cam_left_rgb_im_traj[:, :height, :width, :3]
        cam_right_rgb_im_traj = cam_right_rgb_im_traj[:, :height, :width, :3]

        print("CAM LEFT MAX: ", np.mean(cam_left_rgb_im_traj))
        print("CAM RIGHT MAX: ", np.mean(cam_right_rgb_im_traj))
        # assert(False)

        cam_tri_depth_im = []
        cam_left_rgb_im_traj_resized = []
        cam_right_rgb_im_traj_resized = []
        for k in range(len(cam_left_rgb_im_traj) // batch_size + 1):
            cam_left_rgb_batch = cam_left_rgb_im_traj[batch_size*k:batch_size*(k+1)]
            cam_right_rgb_batch = cam_right_rgb_im_traj[batch_size*k:batch_size*(k+1)]

            print("CAM LEFT RGB BATCH: ", np.mean(cam_left_rgb_batch))
            print("CAM RIGHT RGB BATCH: ", np.mean(cam_right_rgb_batch))

            intrinsics_batch = intrinsics[batch_size*k:batch_size*(k+1)]
            if len(intrinsics_batch) == 0:
                continue
            H, W = cam_left_rgb_batch.shape[1], cam_left_rgb_batch.shape[2]
            print("BASELINE: ", camera.get_camera_baseline())
            tri_depth_im_batch, disparity_batch, disparity_sparse_batch, cam_left_rgb_resized_batch, cam_right_rgb_resized_batch, resized_intrinsics = model.inference(
                rgb_left=format_image(cam_left_rgb_batch),
                rgb_right=format_image(cam_right_rgb_batch),
                intrinsics=torch.tensor(intrinsics_batch).to(torch.float32).cuda(), 
                resize=None,
                baseline=camera.get_camera_baseline()
            )
            tri_depth_im_batch = tri_depth_im_batch.cpu().detach().numpy()
            print("TRI DEPTH IM BATCH: ", np.mean(tri_depth_im_batch))
            assert(False)
            cam_tri_depth_im.append(tri_depth_im_batch)
            cam_left_rgb_im_traj_resized.append(cam_left_rgb_resized_batch.cpu().detach().numpy())
            cam_right_rgb_im_traj_resized.append(cam_right_rgb_resized_batch.cpu().detach().numpy())
        
        cam_left_rgb_im_traj_resized = np.concatenate(cam_left_rgb_im_traj_resized, axis=0)
        cam_right_rgb_im_traj_resized = np.concatenate(cam_right_rgb_im_traj_resized, axis=0)
        cam_tri_depth_im = np.concatenate(cam_tri_depth_im, axis=0)

        left_rgb_im_traj.append(255*np.transpose(cam_left_rgb_im_traj_resized, (0, 2, 3, 1)))
        right_rgb_im_traj.append(255*np.transpose(cam_right_rgb_im_traj_resized, (0, 2, 3, 1)))
        tri_depth_im_traj.append(cam_tri_depth_im)

        cam_matrices[i] = resized_intrinsics.cpu().detach().numpy()[0]

    left_rgb_im_traj = np.array(left_rgb_im_traj)
    right_rgb_im_traj = np.array(right_rgb_im_traj)
    tri_depth_im_traj = np.array(tri_depth_im_traj)

    # # Close Everything #
    for camera in cameras:
        camera.disable_camera()

    left_rgb_im_traj = np.swapaxes(left_rgb_im_traj, 0, 1)
    right_rgb_im_traj = np.swapaxes(right_rgb_im_traj, 0, 1)
    tri_depth_im_traj = np.squeeze(np.swapaxes(tri_depth_im_traj, 0, 1))

    return left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, frame_counts[0], cam_matrices

# Get camera extrinsics
def get_camera_extrinsics(filepath, serial_numbers, frame_count):
    # Get extrinsics for the trajectory
    with h5py.File(filepath, "r") as f:
        extrinsics_trajs = []
        for _, serial_num in enumerate(serial_numbers):
            extrinsics_key = str(serial_num) + "_left"
            extrinsics_trajs.append(f['/observation/camera_extrinsics/' + extrinsics_key][:])
    combined_extrinsics = np.swapaxes(np.array(extrinsics_trajs), 0, 1)[:frame_count-1]
    return combined_extrinsics 

# Get actions
def get_actions(filepath, frame_count):
    with h5py.File(filepath, "r") as f:
        cartesian_position = f["/action/cartesian_position"][:]
        gripper_position = f["/action/gripper_position"][:]
        gripper_position = np.expand_dims(gripper_position, -1)
        actions = np.concatenate((cartesian_position, gripper_position), axis=-1)[:frame_count-1]
    return actions

def list_directories(root_directory, depth=1):
    if depth == 0:
        return []
    
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if dirpath != root_directory and len(dirpath.split(os.sep)) - root_directory.count(os.sep) == depth:
            directories.append(dirpath)
        if len(dirpath.split(os.sep)) - root_directory.count(os.sep) > depth:
            del dirnames[:]
    return directories

def get_input_output_paths(r2d2_data_path, save_path, prefix):
    folder_depth = 5 if "r2d2_full" in r2d2_data_path else 2
    input_traj_paths = sorted(list_directories(r2d2_data_path, depth=folder_depth))
    output_traj_paths = []
    
    for traj_path in input_traj_paths:
        relative_path = traj_path.split(prefix, 1)[-1].strip()
        output_traj_paths.append(os.path.join(save_path, relative_path))
    return input_traj_paths, output_traj_paths

# Function to get evenly spaced samples with first and last included
def get_evenly_spaced_samples(lst, num_samples):
    if num_samples <= 2:
        return lst[:num_samples]  # Return the first num_samples if num_samples <= 2
    else:
        step = (len(lst) - 1) / (num_samples - 1)  # Calculate step size
        return [lst[int(i * step)] for i in range(num_samples)]  # Generate samples
        

if __name__ == "__main__":
    r2d2_data_path = "/mnt/fsx/ashwinbalakrishna/datasets/0921"
    save_path = "/mnt/fsx/ashwinbalakrishna/datasets/narrow_debugging_old_dataloader_directory"
    prefix = r2d2_data_path.split("/")[-1] + "/"
    
    stereo_ckpt = "/mnt/fsx/ashwinbalakrishna/stereo_20230724.pt"
    num_samples_per_traj = 30
    num_trajectories = 10000
    os.makedirs(save_path, exist_ok=True)

    # hf = h5py.File(os.path.join(save_path, 'rgbd_train_data_fix_scaling_mini.h5'), 'a')

    if os.path.exists(os.path.join(save_path, 'filepaths.pkl')):
        with open(os.path.join(save_path, 'filepaths.pkl'), 'rb') as file:
            path_info = pickle.load(file)
            input_traj_paths, output_traj_paths = path_info["input_paths"], path_info["output_paths"]
    else:
        input_traj_paths, output_traj_paths = get_input_output_paths(r2d2_data_path, save_path, prefix=prefix)
        with open(os.path.join(save_path, 'filepaths.pkl'), 'wb') as file:
            # Dump the object into the pickle file
            pickle.dump({"input_paths": input_traj_paths, "output_paths": output_traj_paths}, file)

    for i, (input_traj_path, output_traj_path) in enumerate(zip(input_traj_paths, output_traj_paths)):
        traj_name = input_traj_path.split(prefix)[-1]
        print("I: ", i, "TRAJ NAME: ", traj_name)
        start_time = time.time()
        if i > num_trajectories: 
            break

        print("INPUT TRAJ PATH: ",  input_traj_path)
        print("OUTPUT TRAJ PATH: ", output_traj_path)

        # if os.path.exists(os.path.join(output_traj_path, 'low_dim_info.npz')) and os.path.exists(os.path.join(output_traj_path, 'images')):
        #     print("SKIPPED: this trajectory has already been processed")
        #     continue

        h5_path = os.path.join(input_traj_path, "trajectory.h5")
        svo_path = os.path.join(input_traj_path, "recordings/SVO")

        left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, frame_count, cam_matrices = get_rgbd_tuples(svo_path, stereo_ckpt)
        print("TRI DEPTH IM TRAJ SHAPE: ", tri_depth_im_traj[0][0].shape)
        print("TRI DEPTH IM TRAJ MAX: ", np.max(tri_depth_im_traj[0][0]))
        assert(False)
        if not len(left_rgb_im_traj):
            continue

        idxs = list(range(len(left_rgb_im_traj)))
        traj_idxs = np.array(get_evenly_spaced_samples(idxs, num_samples_per_traj))
        combined_extrinsics = get_camera_extrinsics(h5_path, serial_numbers, frame_count)
        actions = get_actions(h5_path, frame_count)

        combined_extrinsics = combined_extrinsics[traj_idxs]
        actions = actions[traj_idxs]
        left_rgb_im_traj = left_rgb_im_traj[traj_idxs]
        right_rgb_im_traj = right_rgb_im_traj[traj_idxs]
        tri_depth_im_traj = tri_depth_im_traj[traj_idxs]

        assert(combined_extrinsics.shape[0] == left_rgb_im_traj.shape[0])
        assert(tri_depth_im_traj.shape[0] == left_rgb_im_traj.shape[0])
        assert(actions.shape[0] == left_rgb_im_traj.shape[0])
        assert(right_rgb_im_traj.shape[0] == left_rgb_im_traj.shape[0])
    
        os.makedirs(output_traj_path, exist_ok=True)
        np.savez(os.path.join(output_traj_path, 'low_dim_info.npz'), actions=actions, extrinsics_traj=combined_extrinsics, camera_matrices=cam_matrices, traj_idxs=traj_idxs)

        left_rgb_im_traj = left_rgb_im_traj.astype(np.uint8)
        right_rgb_im_traj = right_rgb_im_traj.astype(np.uint8)
        # tri_depth_im_traj = np.clip(tri_depth_im_traj, 0, MAX_DEPTH)
        # tri_depth_im_traj = (tri_depth_im_traj / MAX_DEPTH) * 255
        # tri_depth_im_traj = tri_depth_im_traj.astype(np.uint8)
    
        images_dir = os.path.join(output_traj_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        print("TRI DEPTH: ", tri_depth_im_traj.shape)

        for camera_idx in range(left_rgb_im_traj.shape[1]):
            rgb_left_dir = os.path.join(images_dir, "left_rgb", f"camera_{camera_idx}")
            rgb_right_dir = os.path.join(images_dir, "right_rgb", f"camera_{camera_idx}")
            depth_dir = os.path.join(images_dir, "tri_depth", f"camera_{camera_idx}")
            os.makedirs(rgb_right_dir, exist_ok=True)
            os.makedirs(rgb_left_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            for t in range(left_rgb_im_traj.shape[0]):
                cv2.imwrite(os.path.join(rgb_left_dir, f"left_rgb_{t:03}.jpg"), left_rgb_im_traj[t, camera_idx])
                cv2.imwrite(os.path.join(rgb_right_dir, f"right_rgb_{t:03}.jpg"), right_rgb_im_traj[t, camera_idx])
                np.save(os.path.join(depth_dir, f"tri_depth_{t:03}.npy"), tri_depth_im_traj[t, camera_idx])

        print("TIME ELAPSED: ", time.time() - start_time)