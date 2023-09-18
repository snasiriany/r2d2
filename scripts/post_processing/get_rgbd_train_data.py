import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
import fnmatch
import h5py

from r2d2.camera_utils.recording_readers.svo_reader import SVOReader

# Get (RGBD_1, RGBD_2, RGBD_3) for a given trajectory in addition
# to camera intrinsics information
def get_rgbd_tuples(filepath):
    svo_files = []
    for root, _, files in os.walk(filepath):
        for filename in files:
            if fnmatch.fnmatch(filename, "*.svo"):
                svo_files.append(os.path.join(root, filename))
    if len(svo_files) == 0:
        return [], [], [], [], []

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
    rgbd_im_traj = []
    for j in range(frame_counts[0]-1):
        rgbd_ims = []
        for i, camera in enumerate(cameras):
            output = camera.read_camera(return_timestamp=True)
            if output is None:
                break
            else:
                data_dict, timestamp = output
            im_key = '%s_left' % serial_numbers[i]
            rgb_im = cv2.cvtColor(data_dict['image'][im_key], cv2.COLOR_BGRA2BGR)
            # cv2.imwrite("testing" + str(i) + "_rgb.jpg", rgb_im)
            depth_im =  np.expand_dims(data_dict['depth'][im_key], -1)
            # cv2.imwrite("testing" + str(i) + "_depth.jpg", depth_im)
            rgbd_im = np.concatenate((rgb_im, depth_im), axis=-1)
            rgbd_ims.append(rgbd_im)
        rgbd_ims = np.array(rgbd_ims)
        rgbd_im_traj.append(rgbd_ims)
    rgbd_im_traj = np.array(rgbd_im_traj)

    # # Close Everything #
    for camera in cameras:
        camera.disable_camera()

    return rgbd_im_traj, serial_numbers, frame_counts[0], cam_matrices, cam_distortions

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
        

if __name__ == "__main__":
    r2d2_data_path = "/home/ashwinbalakrishna/Desktop/data/r2d2data"
    save_path = "/home/ashwinbalakrishna/Desktop/git-repos/r2d2"

    hf = h5py.File(os.path.join(save_path, 'rgbd_train_data.h5'), 'a')

    for i, traj_name in enumerate(os.listdir(r2d2_data_path)):
        print("I: ", i, "TRAJ NAME: ", traj_name)
        if i > 100: 
            break
        traj_group = hf.require_group(traj_name)
        traj_path = os.path.join(r2d2_data_path, traj_name)
        svo_path = os.path.join(traj_path, 'recordings/SVO')
        rgbd_im_traj, serial_numbers, frame_count, cam_matrices, cam_distortions = get_rgbd_tuples(svo_path)
        if not len(rgbd_im_traj):
            continue
        combined_extrinsics = get_camera_extrinsics(traj_path, serial_numbers, frame_count)
        assert(combined_extrinsics.shape[0] == rgbd_im_traj.shape[0])

        traj_group.create_dataset("rgbd_im_traj", data=rgbd_im_traj)
        traj_group.create_dataset("extrinsics_traj", data=combined_extrinsics)
        traj_group.create_dataset("camera_matrices", data=cam_matrices)
        traj_group.create_dataset("camera_distortions", data=cam_distortions)

