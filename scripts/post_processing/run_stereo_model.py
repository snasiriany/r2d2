import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as tfn

from r2d2.trajectory_utils.misc import load_trajectory

torch._C._jit_set_profiling_executor(False)

filepath = "/home/ashwinbalakrishna/Desktop/data/mixed_data/Fri_Apr_21_10:35:02_2023/trajectory.h5"
recording_folderpath = "/home/ashwinbalakrishna/Desktop/data/mixed_data/Fri_Apr_21_10:35:02_2023/recordings/SVO"
traj = load_trajectory(filepath, recording_folderpath=recording_folderpath)

timestep = 0
frame = traj[timestep]
obs = frame["observation"]
image_obs = obs["image"]
cam_id = "19824535"
left_key, right_key = f"{cam_id}_left", f"{cam_id}_right"
left_rgb, right_rgb = image_obs[left_key], image_obs[right_key]
left_pose, right_pose = obs["camera_extrinsics"][left_key], obs["camera_extrinsics"][right_key]

# TODO(Ashwin): update intrinsics with correct one's you read from SVO
# I'm hardcoding intrinsics right now but these need to come from the actual cameras.
# That can be accomplished by loading cameras in r2d2 (the relevant class is `ZedCamera`).
fx = 1075.0 / 2.0
fy = 1220.0 / 2.0
cx = 1280.0 / 4.0
cy = 1024.0 / 4.0
intrinsics = np.array([
    [[fx, 0.0,     cx],
    [0.0,     fy, cy],
    [0.0,     0.0,     1.0]]
])

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

        return depth, output["disparity"], disparity_sparse

"""To get the stereo model checkpoint do the following:
- make sure you have access to s3
- go to https://tri-sso.awsapps.com/start#/ and sign in
- select Toyota/TRI -> S3WritersAccessForTriNA and copy the credentials for reading from S3 (or configure it some other way)
- then copy the model with `aws s3 cp s3://tri-mmt-data/marktj/scratch/20230724/stereo_20230724.pt .`
"""
stereo_ckpt = "/home/ashwinbalakrishna/Desktop/git-repos/r2d2/stereo_20230724.pt"

model = StereoModel(stereo_ckpt)
model.cuda()

def format_image(rgb):
    return torch.tensor(rgb.transpose(0,3,1,2)).to(torch.float32).cuda() / 255.0

# Need the image sides to be divisible by 32.
height, width, _ = left_rgb.shape
height = height - height % 32
width = width - width % 32
left_rgb = left_rgb[:height, :width, :3]
right_rgb = right_rgb[:height, :width, :3]
left_rgb = np.expand_dims(left_rgb, axis=0)
right_rgb = np.expand_dims(right_rgb, axis=0)
print("LEFT SHAPE: ", format_image(left_rgb).shape)
print("RIGHT SHAPE: ", format_image(right_rgb).shape)
print("INTRINSICS SHAPE: ", intrinsics.shape)
depth, disparity, disparity_sparse = model.inference(
    rgb_left=format_image(left_rgb),
    rgb_right=format_image(right_rgb),
    intrinsics=torch.tensor(intrinsics).to(torch.float32).cuda(), 
    resize=None,
)

def make_cv_disparity_image(disparity, max_disparity):
    vis_disparity = disparity / max_disparity
    vis_disparity[vis_disparity < 0.0] = 0.0
    vis_disparity[vis_disparity > 1.0] = 1.0
    vis_disparity = vis_disparity.cpu()
    np_img = (vis_disparity.numpy() * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(np_img, cv2.COLORMAP_JET)
    mapped[vis_disparity < 1e-3, :] = 0
    mapped[vis_disparity > 1.0 - 1e-3, :] = 0
    return mapped

def write_cv_disparity_image(fname_suffix):
    num_disparities = 384 // 2
    disparity_vis = make_cv_disparity_image(disparity[0, 0, :, :], num_disparities)
    disparity_sparse_vis = make_cv_disparity_image(disparity_sparse[0, 0, :, :], num_disparities)
    cv2.imwrite("disparity_vis%s.jpg"%fname_suffix, disparity_vis)
    cv2.imwrite("disparity_sparse_vis%s.jpg"%fname_suffix, disparity_sparse_vis)

write_cv_disparity_image("0")

