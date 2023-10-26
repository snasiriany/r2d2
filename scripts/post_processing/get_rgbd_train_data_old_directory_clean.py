import glob
import json
import os
import torch
import random
import time
import argparse
import pickle

import cv2
from PIL import Image
import numpy as np
import fnmatch
import h5py
import tensorrt as trt

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
from r2d2.trajectory_utils.misc import load_trajectory

VARIANT = "base"
NUM_DISPARITIES = 384
HEIGHT = 704
WIDTH = 1280
MAX_DEPTH = 10
BATCH_SIZE = 16
USE_TRT_MODEL = False
TRT_LOG_LEVEL = trt.Logger.ERROR

if not USE_TRT_MODEL:
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
    format_image = (torch.tensor(rgb.transpose(0,3,1,2)).to(torch.float32).cuda() / 255.0)
    if USE_TRT_MODEL:
        format_image = format_image.contiguous()
    return format_image

def get_stereo_depth(trt_engine, trt_context, left_rgb, right_rgb, intrinsics, baseline):
    left_image_tensor = format_image(left_rgb)
    right_image_tensor = format_image(right_rgb)

    trt_tensors = {
        "left_input": left_image_tensor,
        "right_input": right_image_tensor,
        "disparity": torch.zeros((BATCH_SIZE, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_sparse": torch.zeros((BATCH_SIZE, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_small": torch.zeros((BATCH_SIZE, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
        "confidence": torch.zeros((BATCH_SIZE, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
    }
    trt_buffers = [trt_tensors[trt_engine.get_tensor_name(idx)].data_ptr() for idx in range(trt_engine.num_bindings)]
    trt_context.execute_v2(trt_buffers)
    disparity = trt_tensors["disparity"].cpu().detach().numpy()
    disparity_sparse = trt_tensors["disparity_sparse"].cpu().detach().numpy()
    mask = disparity_sparse != 0
    depth = np.zeros_like(disparity_sparse)
    depth[mask] = baseline * intrinsics[0, 0] / disparity_sparse[mask]
    return depth

def load_trt_engine(engine_name):
    trt_logger = trt.Logger(TRT_LOG_LEVEL)
    trt_runtime = trt.Runtime(trt_logger)
    with open(engine_name, "rb") as in_file:
        trt_serialized_engine_in = in_file.read()
    trt_engine = trt_runtime.deserialize_cuda_engine(trt_serialized_engine_in)
    trt_context = None
    while trt_context is None:
        trt_context = trt_engine.create_execution_context()
    return trt_engine, trt_context

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


def get_images(svo_path, model_dict, num_samples_per_traj):
    svo_files = [os.path.join(svo_path, file) for file in os.listdir(svo_path) if file.endswith(".svo")]
    if len(svo_files) == 0:
        print("SKIPPED: not enough SVO files!")
        return [], [], [], [], [], [], []

    cameras = []
    frame_counts = []
    serial_numbers = []
    cam_matrices = []
    cam_baselines = []

    for svo_file in svo_files:
        # Open SVO Reader
        serial_number = svo_file.split("/")[-1][:-4]
        camera = SVOReader(svo_file, serial_number=serial_number)
        camera.set_reading_parameters(image=True, depth=True, pointcloud=False, concatenate_images=False)
        im_key = '%s_left' % serial_number
        # Intrinsics are the same for the left and the right camera
        cam_matrices.append(camera.get_camera_intrinsics()[im_key]['cameraMatrix'])
        cam_baselines.append(camera.get_camera_baseline())
        frame_count = camera.get_frame_count()
        cameras.append(camera)
        frame_counts.append(frame_count)
        serial_numbers.append(serial_number)

    cameras = [x for y, x in sorted(zip(serial_numbers, cameras))]
    cam_matrices = [x for y, x in sorted(zip(serial_numbers, cam_matrices))]
    cam_baselines = [x for y, x in sorted(zip(serial_numbers, cam_baselines))]
    serial_numbers = sorted(serial_numbers)

    if frame_counts.count(frame_counts[0]) != len(frame_counts):
        return [], [], [], [], [], [], []
    
    left_rgb_im_traj = []
    right_rgb_im_traj = []
    tri_depth_im_traj = []

    # Get evenly spaced ids in a trajectory
    idxs = list(range(frame_counts[0] - 1))
    traj_idxs = get_evenly_spaced_samples(idxs, num_samples_per_traj)

    for i, camera in enumerate(cameras):
        intrinsics = cam_matrices[i]
        intrinsics_traj =  np.repeat(cam_matrices[i][np.newaxis, :, :], frame_counts[0] - 1, axis=0)
        baseline = cam_baselines[i]

        cam_left_rgb_im_traj = []
        cam_right_rgb_im_traj = []

        for j in range(frame_counts[0] - 1):
            if j not in traj_idxs:
                continue
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

        # Need the image sides to be divisible by 32.
        _, height, width, _ = cam_left_rgb_im_traj.shape
        height = height - height % 32
        width = width - width % 32
        cam_left_rgb_im_traj = cam_left_rgb_im_traj[:, :height, :width, :3]
        cam_right_rgb_im_traj = cam_right_rgb_im_traj[:, :height, :width, :3]

        cam_tri_depth_im = []
        for k in range(len(cam_left_rgb_im_traj) // BATCH_SIZE + 1):
            cam_left_rgb_batch = cam_left_rgb_im_traj[BATCH_SIZE*k:BATCH_SIZE*(k+1)]
            cam_right_rgb_batch = cam_right_rgb_im_traj[BATCH_SIZE*k:BATCH_SIZE*(k+1)]
            intrinsics_batch = intrinsics_traj[BATCH_SIZE*k:BATCH_SIZE*(k+1)]

            if len(cam_left_rgb_batch) == 0:
                continue
            if not (cam_left_rgb_batch.shape[1] == HEIGHT and cam_left_rgb_batch.shape[2] == WIDTH):
                print("SKIPPED: Shape mismatch")
                return [], [], [], [], [], [], []

            cam_left_rgb_batch_padded = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, 3))
            cam_left_rgb_batch_padded[:len(cam_left_rgb_batch), :, :, :] = cam_left_rgb_batch
            cam_right_rgb_batch_padded = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, 3))
            cam_right_rgb_batch_padded[:len(cam_right_rgb_batch), :, :, :] = cam_right_rgb_batch

            if USE_TRT_MODEL:
                tri_depth_im_batch = get_stereo_depth(model_dict["trt_engine"], model_dict["trt_context"], 
                    cam_left_rgb_batch_padded, cam_right_rgb_batch_padded, intrinsics, baseline)[:len(cam_left_rgb_batch)]
            else:
                tri_depth_im_batch, _, _, _, _, _ = model_dict["model"].inference(
                    rgb_left=format_image(cam_left_rgb_batch),
                    rgb_right=format_image(cam_right_rgb_batch),
                    intrinsics=torch.tensor(intrinsics_batch).to(torch.float32).cuda(), 
                    resize=None,
                    baseline=baseline
                )
                tri_depth_im_batch = tri_depth_im_batch.cpu().detach().numpy()

            cam_tri_depth_im.append(tri_depth_im_batch)
        
        cam_tri_depth_im = np.concatenate(cam_tri_depth_im, axis=0)

        left_rgb_im_traj.append(cam_left_rgb_im_traj)
        right_rgb_im_traj.append(cam_right_rgb_im_traj)
        tri_depth_im_traj.append(cam_tri_depth_im)

    left_rgb_im_traj = np.array(left_rgb_im_traj)
    right_rgb_im_traj = np.array(right_rgb_im_traj)
    tri_depth_im_traj = np.array(tri_depth_im_traj)

    # # Close Everything #
    for camera in cameras:
        camera.disable_camera()

    left_rgb_im_traj = np.swapaxes(left_rgb_im_traj, 0, 1)
    right_rgb_im_traj = np.swapaxes(right_rgb_im_traj, 0, 1)
    tri_depth_im_traj = np.squeeze(np.swapaxes(tri_depth_im_traj, 0, 1))

    return left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, frame_counts[0], cam_matrices, traj_idxs

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
        

def main(process_id, num_processes):
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    if USE_TRT_MODEL:
        engine_path = os.path.join("/home/ubuntu/trt_stereo", 
            "stereo_{}_h{}_w{}_b_{}_d{}__{}.engine".format(VARIANT, HEIGHT, WIDTH, NUM_DISPARITIES, BATCH_SIZE, device_name))
        trt_engine, trt_context = load_trt_engine(engine_path)
        model_dict = {"trt_engine": trt_engine, "trt_context": trt_context}
    else:
        stereo_ckpt = "/mnt/fsx/ashwinbalakrishna/stereo_20230724.pt"
        model = StereoModel(stereo_ckpt)
        model.cuda()
        model_dict = {"model": model}

    # r2d2_data_path = "/mnt/fsx/surajnair/datasets/raw_r2d2_full"
    # save_path = "/mnt/fsx/ashwinbalakrishna/datasets/define_train_data/r2d2_all"
    r2d2_data_path = "/mnt/fsx/ashwinbalakrishna/datasets/0921"
    save_path = "/mnt/fsx/ashwinbalakrishna/datasets/narrow_debugging_old_dataloader_directory_clean_TRT"
    prefix = r2d2_data_path.split("/")[-1] + "/"
    resize_shape = (WIDTH, HEIGHT)

    num_samples_per_traj = 32
    num_trajectories = 100000
    num_failures = 0
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(os.path.join(save_path, 'filepaths.pkl')):
        with open(os.path.join(save_path, 'filepaths.pkl'), 'rb') as file:
            path_info = pickle.load(file)
            input_traj_paths, output_traj_paths = path_info["input_paths"], path_info["output_paths"]
    else:
        input_traj_paths, output_traj_paths = get_input_output_paths(r2d2_data_path, save_path, prefix=prefix)
        with open(os.path.join(save_path, 'filepaths.pkl'), 'wb') as file:
            # Dump the object into the pickle file
            pickle.dump({"input_paths": input_traj_paths, "output_paths": output_traj_paths}, file)
    
    if os.path.exists(os.path.join(save_path, 'failures.pkl')):
        with open(os.path.join(save_path, 'failures.pkl'), 'rb') as file:
            # Load the object from the pickle file
            failures = pickle.load(file)
    else:
        failures = {'Missing SVO/H5 File': set(), 'Trajectory Loading Error': set(), 'Trajectory too Long': set(),
                        'Trajectory is Empty': set(), 'Shape Mismatch': set()}

    for i, (input_traj_path, output_traj_path) in enumerate(zip(input_traj_paths, output_traj_paths)):
        if i % num_processes == process_id:
            traj_name = input_traj_path.split(prefix)[-1]
            failure = False
            for key in failures:
                if traj_name in failures[key]:
                    print("SKIPPED: ALREADY MARKED AS A FAILURE")
                    failure = True
            if failure:
                num_failures += 1
                continue
            print("INPUT TRAJ PATH: ",  input_traj_path)
            print("OUTPUT TRAJ PATH: ", output_traj_path)
            print("I: ", i, "TRAJ NAME: ", traj_name)
            if os.path.exists(os.path.join(output_traj_path, 'low_dim_info.npz')) and os.path.exists(os.path.join(output_traj_path, 'images')):
                print("SKIPPED: this trajectory has already been processed")
                continue
            start_time = time.time()
            if i > num_trajectories: 
                break

            h5_path = os.path.join(input_traj_path, "trajectory.h5")
            svo_path = os.path.join(input_traj_path, "recordings/SVO")
            if not (os.path.exists(svo_path) and os.path.exists(h5_path)):
                print("SKIPPED: missing SVO or H5 file")
                failures['Missing SVO/H5 File'].add(traj_name)
                continue

            left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, frame_count, cam_matrices, traj_idxs = get_images(svo_path, model_dict, num_samples_per_traj)
            if not len(left_rgb_im_traj):
                print("SKIPPED: no images in trajectory")
                failures['Shape Mismatch'].add(traj_name)
                continue
            combined_extrinsics = get_camera_extrinsics(h5_path, serial_numbers, frame_count)
            actions = get_actions(h5_path, frame_count)
            combined_extrinsics = combined_extrinsics[traj_idxs]
            actions = actions[traj_idxs]

            if ((combined_extrinsics.shape[0] != left_rgb_im_traj.shape[0]) or \
                (tri_depth_im_traj.shape[0] != left_rgb_im_traj.shape[0]) or \
                (actions.shape[0] != left_rgb_im_traj.shape[0]) or \
                (right_rgb_im_traj.shape[0] != left_rgb_im_traj.shape[0]) or \
                (left_rgb_im_traj.shape[1] != right_rgb_im_traj.shape[1]) or \
                (left_rgb_im_traj.shape[1] != tri_depth_im_traj.shape[1]) or \
                (left_rgb_im_traj.shape[1] != 3)):
                print("SKIPPED: the shapes don't make sense!")
                failures['Shape Mismatch'].add(traj_name)
                continue

            print("TIME ELAPSED: ", time.time() - start_time)
            os.makedirs(output_traj_path, exist_ok=True)
            np.savez(os.path.join(output_traj_path, 'low_dim_info.npz'), actions=actions, extrinsics_traj=combined_extrinsics, camera_matrices=cam_matrices, traj_idxs=np.array(traj_idxs))

            left_rgb_im_traj = left_rgb_im_traj.astype(np.uint8)
            right_rgb_im_traj = right_rgb_im_traj.astype(np.uint8)
            # tri_depth_im_traj = np.clip(tri_depth_im_traj, 0, MAX_DEPTH)
            # tri_depth_im_traj = (tri_depth_im_traj / MAX_DEPTH) * 255
            # tri_depth_im_traj = tri_depth_im_traj.astype(np.uint8)
        
            images_dir = os.path.join(output_traj_path, "images")
            os.makedirs(images_dir, exist_ok=True)

            for camera_idx in range(left_rgb_im_traj.shape[1]):
                rgb_left_dir = os.path.join(images_dir, "left_rgb", f"camera_{camera_idx}")
                rgb_right_dir = os.path.join(images_dir, "right_rgb", f"camera_{camera_idx}")
                depth_dir = os.path.join(images_dir, "tri_depth", f"camera_{camera_idx}")
                os.makedirs(rgb_right_dir, exist_ok=True)
                os.makedirs(rgb_left_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                for t in range(left_rgb_im_traj.shape[0]):
                    cv2.imwrite(os.path.join(rgb_left_dir, f"left_rgb_{t:03}.jpg"), cv2.resize(left_rgb_im_traj[t, camera_idx], resize_shape))
                    cv2.imwrite(os.path.join(rgb_right_dir, f"right_rgb_{t:03}.jpg"), cv2.resize(right_rgb_im_traj[t, camera_idx], resize_shape))
                    np.save(os.path.join(depth_dir, f"tri_depth_{t:03}.npy"), cv2.resize(tri_depth_im_traj[t, camera_idx], resize_shape))

            print("TIME ELAPSED FINAL: ", time.time() - start_time)

    print("NUM FAILURES: ", num_failures)
    # Open the pickle file in binary write mode
    with open(os.path.join(save_path, 'failures.pkl'), 'wb') as file:
        # Dump the object into the pickle file
        pickle.dump(failures, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--num_processes', type=int, default=1)
    args = parser.parse_args()
    main(args.process_id, args.num_processes)