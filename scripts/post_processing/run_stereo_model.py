import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as tfn
import time
import os
import glob
import tensorrt as trt

from r2d2.trajectory_utils.misc import load_trajectory
from r2d2.camera_utils.recording_readers.svo_reader import SVOReader

VARIANT = "base"
TRT_LOG_LEVEL = trt.Logger.ERROR
NUM_DISPARITIES = 384
HEIGHT = 704
WIDTH = 1280
TRT_BATCH_SIZE = 1

def format_image(rgb):
    return (torch.tensor(rgb.transpose(2,0,1)[None]).to(torch.float32).cuda() / 255.0).contiguous()

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
        "disparity": torch.zeros((1, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_sparse": torch.zeros((1, 1, HEIGHT, WIDTH)).cuda().to(torch.float32),
        "disparity_small": torch.zeros((1, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
        "confidence": torch.zeros((1, 1, HEIGHT//4, WIDTH//4)).cuda().to(torch.float32),
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

def plot_depth_images(tri_depths, left_rgbs, right_rgbs, output_file='depth_image_grid.png'):
    images = []
    for tri_depth, left_rgb, right_rgb in zip(tri_depths, left_rgbs, right_rgbs):
        tri_depth_vis = make_disparity_vis(tri_depth, 9.0)
        images.append(left_rgb)
        images.append(right_rgb)
        images.append(tri_depth_vis)

    # Create a 3x4 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 5))
    titles = ["Left RGB", "Right RGB", "TRI Depth"] * 3

    # Iterate through the images and display them on the subplots
    for i, ax in enumerate(axes.ravel()):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Assuming your images are grayscale, change the cmap as needed
            if i < 3:
                ax.set_title(titles[i])
            ax.axis('off')  # Turn off axis labels

    # Adjust spacing and display the plot
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # Save the figure as an image (e.g., PNG)
    plt.savefig(output_file, bbox_inches='tight')

def main():
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    engine_name = os.path.join("/home/ashwinbalakrishna/Desktop/trt_stereo", 
        "stereo_{}_h{}_w{}_b_{}_d{}__{}.engine".format(VARIANT, HEIGHT, WIDTH, NUM_DISPARITIES, TRT_BATCH_SIZE, device_name))

    trt_engine, trt_context = load_trt_engine(engine_name)

    filepath = "/home/ashwinbalakrishna/Desktop/data/r2d2data/lab-uploads/AUTOLab/success/2023-07-07/Fri_Jul__7_14:57:48_2023"
    traj_filepath = os.path.join(filepath, "trajectory.h5")
    svo_path = os.path.join(filepath, "recordings/SVO")
    traj = load_trajectory(traj_filepath, recording_folderpath=svo_path)
    svo_files = [os.path.join(svo_path, file) for file in os.listdir(svo_path) if file.endswith(".svo")]

    frame_counts = []
    serial_numbers = []
    cam_matrices = []
    cam_baselines = []

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

    # return serial_numbers
    cam_matrices = [x for y, x in sorted(zip(serial_numbers, cam_matrices))]
    cam_baselines = [x for y, x in sorted(zip(serial_numbers, cam_baselines))]
    serial_numbers = sorted(serial_numbers)

    assert frame_counts.count(frame_counts[0]) == len(frame_counts)

    timestep = np.random.randint(frame_counts[0])
    frame = traj[timestep]
    obs = frame["observation"]
    image_obs = obs["image"]

    tri_depths = []
    left_rgbs = []
    right_rgbs = []

    for i, cam_id in enumerate(serial_numbers):
        left_key, right_key = f"{cam_id}_left", f"{cam_id}_right"
        left_rgb, right_rgb = image_obs[left_key], image_obs[right_key] 
        intrinsics = cam_matrices[i]
        baseline = cam_baselines[i]
        # Need the image sides to be divisible by 32.
        height, width, _ = left_rgb.shape
        height = height - height % 32
        width = width - width % 32
        left_rgb = left_rgb[:height, :width, :3]
        right_rgb = right_rgb[:height, :width, :3]
        left_rgbs.append(left_rgb)
        right_rgbs.append(right_rgb)
        depth = get_stereo_depth(trt_engine, trt_context, left_rgb, right_rgb, intrinsics, baseline)
        tri_depths.append(depth[0, 0, :, :])

    plot_depth_images(tri_depths, left_rgbs, right_rgbs)

if __name__ == "__main__":
    with torch.no_grad():
        main()