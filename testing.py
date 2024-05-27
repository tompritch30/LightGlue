from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import numpy as np
import cv2

## helper files
import poses
import plotting

from pathlib import Path
import matplotlib.pyplot as plt

images = Path("../../Data/TarTanAir-P002-Forest/image_left")
torch.set_grad_enabled(False)

## Load extractor and matcher module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

imageIdx0 = 10
imageIdx1 = 11

image0_filename = f"{imageIdx0:06d}_left.png"
image1_filename = f"{imageIdx1:06d}_left.png"

image0 = load_image(images / image0_filename)
image1 = load_image(images / image1_filename)

########## SuperPoint #############
# Fields for SuperPoint:
# "descriptor_dim": 256,"nms_radius": 4, "max_num_keypoints": None, "detection_threshold": 0.0005, "remove_borders": 4,
extractor = SuperPoint(max_num_keypoints=64, descriptor_dim=256).eval().to(device)  # load the extractor

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))

# torch.save(feats0, 'feats0.pth')
# torch.save(feats1, 'feats1.pth')

# Feats 0 is a dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
# keypoints is tensor([[ [287.3125, 304.8125], ... ]]) i.e. is a coordinate for each point
# score: tensor([[0.5934, 0.5182, 0.5137, 0.4981]]) i.e. confidence score 0-1 for all
# descriptor : tensor([[[ 0.0531,w/ 256 elements], ]]) is a 2D list : [[ [256 elements], [256 elements] etc ... ]]
# image_size, Value: tensor([[640., 480.]]) - in px

# print(type(feats0))
# print(feats0.keys())
# for key, value in feats0.items():
#     if isinstance(value, (list, tuple)):
#         print(f"Key: {key}, Sample Value: {value[:5]}")  # Print first 5 elements if it's a list or tuple
#     elif isinstance(value, np.ndarray):  # Assuming you are using NumPy arrays
#         print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
#     else:
#         print(f"Key: {key}, Value: {value}")
# print(feats0, "\n\n", feats0)

############ LightGlue #########
# feats0 = torch.load('feats0.pth')
# feats1 = torch.load('feats1.pth')

# # Fields for LightGlue
# # {"name": "lightglue", "input_dim": 256, "descriptor_dim": 256, "add_scale_ori": False, "n_layers": 9,
# #     "num_heads": 4, "flash": True, "mp": False, "depth_confidence": 0.95, "width_confidence": 0.99,
# #     "filter_threshold": 0.1, "weights": None  }
matcher = LightGlue(features="superpoint").eval().to(device)

# Do feature mapping
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# m_kpts0 = matchedkeypoints image 0
# print("Shape:", m_kpts0.shape)
# print("Data type:", m_kpts0.dtype)
# print("First elements:", m_kpts0)
# print("Mean:", m_kpts0.mean().item())
# print("Standard deviation:", m_kpts0.std().item())
# print("Min value:", m_kpts0.min().item())
# print("Max value:", m_kpts0.max().item())

# torch.save(m_kpts0, 'm_kpts0.pth')
# torch.save(m_kpts1, 'm_kpts1.pth')
# Is a list of coordinates from superglue for matches

# m_kpts0 = torch.load('m_kpts0.pth')
# m_kpts1 = torch.load('m_kpts1.pth')

######### Poses ###########
### Ground truth poses
pose_file = "../../Data/TarTanAir-P002-Forest/pose_left.txt"
pose_matrix1, pose_matrix2 = poses.get_pose_matrices(pose_file, imageIdx0, imageIdx1)

print("Pose Matrix 1:\n", pose_matrix1)
print("Pose Matrix 2:\n", pose_matrix2)

### SuperGlue poses
## Convert to numpy arrays
m_kpts0 = m_kpts0.numpy().reshape(-1, 2)
m_kpts1 = m_kpts1.numpy().reshape(-1, 2)

## Triangulate points
points_3d = poses.triangulate_points(m_kpts0, m_kpts1, pose_matrix1, pose_matrix2, poses.K)
print("Triangulated 3D Points:\n", points_3d)

## Reproject points to the image planes
reprojected_kpts0 = poses.reproject_points(points_3d, pose_matrix1, poses.K)
reprojected_kpts1 = poses.reproject_points(points_3d, pose_matrix2, poses.K)

print("Reprojected Keypoints Image 0:\n", reprojected_kpts0)
print("Reprojected Keypoints Image 1:\n", reprojected_kpts1)

# Compute reprojection errors
error0 = poses.compute_reprojection_error(m_kpts0, reprojected_kpts0)
error1 = poses.compute_reprojection_error(m_kpts1, reprojected_kpts1)

print("Reprojection Error Image 0:\n", error0)
print("Reprojection Error Image 1:\n", error1)

## Epipolar error
# Compute the Essential matrix (as an example)
E, _ = cv2.findEssentialMat(m_kpts0, m_kpts1, poses.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Compute epipolar errors
epipolar_error = poses.compute_epipolar_error(E, m_kpts0, m_kpts1)
print("Epipolar Error:\n", epipolar_error)

# Plotting normal graph
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

### Plotting epipolar graph ###
# # Map errors to colors
print("NOW trying to plot the epipolar graph \n\n")
colors = plotting.map_errors_to_colors(epipolar_error)
print(colors)

# # Plot images
axes = plotting.plot_images([image0, image1])
print("Axes:", axes)

# Calculate the offset for the second image
image_width = image0.shape[1] if image0.ndim == 3 else image0.shape[0]

# # Plot matches with color based on epipolar error
for i, (pt1, pt2, color) in enumerate(zip(m_kpts0, m_kpts1, colors)):
    x0, y0 = pt1
    x1, y1 = pt2
    # x1 += image_width  # Offset x-coordinate of the second image
    axes[0].plot([x0, x1 + image_width], [y0, y1], color=color, lw=1.0)  # thicker lines

viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

# Show plot
plt.show()

# From readme
# # # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher
#
# # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
# image0 = load_image(images / "000011_left.png").cuda()
# image1 = load_image(images / "000012_left.png").cuda()
#
# # extract local features
# feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
# feats1 = extractor.extract(image1)

# # match the features
# matches01 = matcher({'image0': feats0, 'image1': feats1})
# feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
# matches = matches01['matches']  # indices with shape (K,2)
# points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
# points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
