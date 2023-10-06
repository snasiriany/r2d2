import glob
import json
import os
import torch
import random
import time

import cv2
from PIL import Image
import numpy as np
import h5py
import tensorrt as trt

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader
from r2d2.trajectory_utils.misc import load_trajectory

VARIANT = "base"
TRT_LOG_LEVEL = trt.Logger.ERROR
NUM_DISPARITIES = 384
HEIGHT = 704
WIDTH = 1280
MAX_DEPTH = 10
TRT_BATCH_SIZE = 32

def format_image(rgb):
    return (torch.tensor(rgb.transpose(0,3,1,2)).to(torch.float32).cuda() / 255.0).contiguous()

def make_disparity_vis(disparity, max_disparity=NUM_DISPARITIES):
    vis_disparity = disparity / max_disparity
    vis_disparity[vis_disparity < 0.0] = 0.0
    vis_disparity[vis_disparity > 1.0] = 1.0
    mapped = np.round(vis_disparity * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(mapped, cv2.COLORMAP_JET)
    mapped[vis_disparity < 1e-3, :] = 0
    mapped[vis_disparity > 1.0 - 1e-3, :] = 0
    return mapped

def get_stereo_depth(trt_engine, trt_context, left_rgb, right_rgb, intrinsics, baseline):
    left_image_tensor = format_image(left_rgb)
    right_image_tensor = format_image(right_rgb)

    trt_tensors = {
        "left_input": left_image_tensor,
        "right_input": right_image_tensor,
        "disparity": torch.zeros((TRT_BATCH_SIZE, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_sparse": torch.zeros((TRT_BATCH_SIZE, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_small": torch.zeros((TRT_BATCH_SIZE, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
        "confidence": torch.zeros((TRT_BATCH_SIZE, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
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
    trt_context = trt_engine.create_execution_context()
    return trt_engine, trt_context

def get_images(filename, traj, traj_idxs, engine_path):
    trt_engine, trt_context = load_trt_engine(engine_path)
    svo_path = os.path.join(filename, "recordings/SVO")
    svo_files = [os.path.join(svo_path, file) for file in os.listdir(svo_path) if file.endswith(".svo")]
    if len(svo_files) == 0:
        return [], [], [], [], [], []

    frame_counts = []
    serial_numbers = []
    cam_matrices = []
    cam_baselines = []
    cameras = []

    for svo_file in svo_files:
        # Open SVO Reader
        serial_number = svo_file.split("/")[-1][:-4]
        camera = SVOReader(svo_file, serial_number=serial_number)
        camera.set_reading_parameters(image=True, concatenate_images=False)
        im_key = '%s_left' % serial_number
        # Intrinsics are the same for the left and the right camera
        cam_matrices.append(camera.get_camera_intrinsics()[im_key]['cameraMatrix'])
        cam_baselines.append(camera.get_camera_baseline())
        frame_count = camera.get_frame_count()
        frame_counts.append(frame_count)
        serial_numbers.append(serial_number)
        cameras.append(camera)

    # return serial_numbers
    cameras = [x for y, x in sorted(zip(serial_numbers, cameras))]
    cam_matrices = [x for y, x in sorted(zip(serial_numbers, cam_matrices))]
    serial_numbers = sorted(serial_numbers)

    assert frame_counts.count(frame_counts[0]) == len(frame_counts)

    left_rgb_im_traj = []
    right_rgb_im_traj = []
    tri_depth_im_traj = []

    image_obs_list = [frame["observation"]["image"] for i, frame in enumerate(traj) if i in traj_idxs]

    for i, cam_id in enumerate(serial_numbers):
        intrinsics = cam_matrices[i]
        baseline = cam_baselines[i]
        left_key, right_key = f"{cam_id}_left", f"{cam_id}_right"

        cam_left_rgb_im_traj = []
        cam_right_rgb_im_traj = []

        for j, image_obs in enumerate(image_obs_list):
            rgb_im_left, rgb_im_right = image_obs[left_key], image_obs[right_key] 
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
        for k in range(len(cam_left_rgb_im_traj) // TRT_BATCH_SIZE + 1):
            cam_left_rgb_batch = cam_left_rgb_im_traj[TRT_BATCH_SIZE*k:TRT_BATCH_SIZE*(k+1)]
            cam_right_rgb_batch = cam_right_rgb_im_traj[TRT_BATCH_SIZE*k:TRT_BATCH_SIZE*(k+1)]
            if len(cam_left_rgb_batch) == 0:
                continue
            tri_depth_im_batch = get_stereo_depth(trt_engine, trt_context, 
                cam_left_rgb_batch, cam_right_rgb_batch, intrinsics, baseline)[:len(cam_left_rgb_batch)]

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

    return left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, cam_matrices

# Get camera extrinsics
def get_camera_extrinsics(serial_numbers, traj, traj_idxs):
    extrinsics_keys = [str(serial_num) + "_left" for serial_num in serial_numbers]
    combined_extrinsics = np.array([[frame["observation"]["camera_extrinsics"][extrinsics_key] \
                                        for i, frame in enumerate(traj) \
                                        if i in traj_idxs] for extrinsics_key in extrinsics_keys])
    return np.swapaxes(np.array(combined_extrinsics), 0, 1)

# Get actions
def get_actions(traj, traj_idxs):
    cartesian_position = np.array([frame["action"]["cartesian_position"] for i, frame in enumerate(traj) if i in traj_idxs])
    gripper_position = np.expand_dims(np.array([frame["action"]["gripper_position"] for i, frame in enumerate(traj) if i in traj_idxs]), -1)
    actions = np.concatenate((cartesian_position, gripper_position), axis=-1)
    return actions


if __name__ == "__main__":
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    engine_path = os.path.join("/home/ashwinbalakrishna/Desktop/trt_stereo", 
        "stereo_{}_h{}_w{}_d{}__{}.engine".format(VARIANT, HEIGHT, WIDTH, NUM_DISPARITIES, device_name))
    r2d2_data_path = "/home/ashwinbalakrishna/Desktop/data/mixed_data"
    save_path = "/home/ashwinbalakrishna/Desktop/r2d2/png_data"

    num_samples_per_traj = 1000
    num_trajectories = 2
    os.makedirs(save_path, exist_ok=True)
    hf = h5py.File(os.path.join(save_path, 'trajectories_data.h5'), 'a')

    for i, traj_name in enumerate(os.listdir(r2d2_data_path)):
        print("I: ", i, "TRAJ NAME: ", traj_name)
        if traj_name in hf:
            print("SKIPPED!")
            continue
        start_time = time.time()
        if i > num_trajectories: 
            break
        traj_path = os.path.join(r2d2_data_path, traj_name)

        h5_path = os.path.join(traj_path, "trajectory.h5")
        svo_path = os.path.join(traj_path, "recordings/SVO")
        traj = load_trajectory(h5_path, recording_folderpath=svo_path)

        # Get random idxs per trajectory while enforcing that 
        # we always have the first element of trajectory
        idxs = list(range(1, len(traj)))
        random.shuffle(idxs)
        traj_idxs = [0] + idxs[:num_samples_per_traj-1]

        left_rgb_im_traj, right_rgb_im_traj, tri_depth_im_traj, serial_numbers, cam_matrices = get_images(traj_path, traj, traj_idxs, engine_path)
        if not len(left_rgb_im_traj):
            continue
        combined_extrinsics = get_camera_extrinsics(serial_numbers, traj, traj_idxs)
        actions = get_actions(traj, traj_idxs)

        assert(combined_extrinsics.shape[0] == left_rgb_im_traj.shape[0])
        assert(tri_depth_im_traj.shape[0] == left_rgb_im_traj.shape[0])
        assert(actions.shape[0] == left_rgb_im_traj.shape[0])
        assert(right_rgb_im_traj.shape[0] == left_rgb_im_traj.shape[0])
        assert left_rgb_im_traj.shape[1] == right_rgb_im_traj.shape[1] == tri_depth_im_traj.shape[1] == 3

        print("TIME ELAPSED: ", time.time() - start_time)

        traj_group = hf.require_group(traj_name)
        traj_group.create_dataset("actions", data=actions)
        traj_group.create_dataset("extrinsics_traj", data=combined_extrinsics)
        traj_group.create_dataset("camera_matrices", data=cam_matrices)

        left_rgb_im_traj = left_rgb_im_traj.astype(np.uint8)
        right_rgb_im_traj = right_rgb_im_traj.astype(np.uint8)
        # tri_depth_im_traj = np.clip(tri_depth_im_traj, 0, MAX_DEPTH)
        # tri_depth_im_traj = (tri_depth_im_traj / MAX_DEPTH) * 255
        # tri_depth_im_traj = tri_depth_im_traj.astype(np.uint8)

        images_dir = os.path.join(save_path, "png_images", traj_name)

        for camera_idx in range(left_rgb_im_traj.shape[1]):
            rgb_left_dir = os.path.join(images_dir, "left_rgb", f"camera_{camera_idx}")
            rgb_right_dir = os.path.join(images_dir, "right_rgb", f"camera_{camera_idx}")
            depth_dir = os.path.join(images_dir, "tri_depth", f"camera_{camera_idx}")
            os.makedirs(rgb_right_dir, exist_ok=True)
            os.makedirs(rgb_left_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            for t in range(left_rgb_im_traj.shape[0]):
                cv2.imwrite(os.path.join(rgb_left_dir, f"left_rgb_{t:03}.png"), left_rgb_im_traj[t, camera_idx])
                cv2.imwrite(os.path.join(rgb_right_dir, f"right_rgb_{t:03}.png"), right_rgb_im_traj[t, camera_idx])
                cv2.imwrite(os.path.join(depth_dir, f"tri_depth_{t:03}.png"), tri_depth_im_traj[t, camera_idx])

        print("TIME ELAPSED FINAL: ", time.time() - start_time)