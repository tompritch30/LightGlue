from matplotlib.patches import ConnectionPatch

from external.LightGlue.code import superglueUtils
from external.LightGlue.lightglue import LightGlue, SuperPoint
from external.LightGlue.lightglue.utils import load_image, rbd
from external.LightGlue.lightglue import viz2d
import torch
import numpy as np
import cv2

## helper files
import poses

from pathlib import Path
import matplotlib.pyplot as plt

# plt.use('TkAgg')

basePath = "C:/Users/thoma/OneDrive/2023 Masters/Project/ProjectCode/Data/TarTanAir-P002-Forest"
images = Path(f"{basePath}/image_left")
torch.set_grad_enabled(False)

## Load extractor and matcher module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

imageIdx0 = 20
imageIdx1 = 21

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

# torch.save(m_kpts0, 'm_kpts0.pth')
# torch.save(m_kpts1, 'm_kpts1.pth')
# Is a list of coordinates from superglue for matches

# m_kpts0 = torch.load('m_kpts0.pth')
# m_kpts1 = torch.load('m_kpts1.pth')

######### Poses ###########
### Ground truth poses
pose_file = f"{basePath}/pose_left.txt"
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

## Using SuperGlue geometry functions
# Normalize keypoints
norm_kpts0 = (m_kpts0 - poses.K[[0, 1], [2, 2]][None]) / poses.K[[0, 1], [0, 1]][None]
norm_kpts1 = (m_kpts1 - poses.K[[0, 1], [2, 2]][None]) / poses.K[[0, 1], [0, 1]][None]

# Compute transformation matrix T_0to1 if not already available
T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1

# Calculate epipolar error using the SuperGlue method
epipolar_error = superglueUtils.compute_epipolar_error(norm_kpts0, norm_kpts1, T_0to1, poses.K, poses.K)
print("SuperGlue Epipolar Error with functions:\n", epipolar_error)

# Create error dictionaries for each image
error_dict_image0 = {}
error_dict_image1 = {}
for idx, error in enumerate(epipolar_error):
    keypoint0 = tuple(m_kpts0[idx])
    keypoint1 = tuple(m_kpts1[idx])
    error_dict_image0[keypoint0] = error
    error_dict_image1[keypoint1] = error

print("Error Dictionary Image 0:", error_dict_image0)
print("Error Dictionary Image 1:", error_dict_image1)

pts0 = np.array(m_kpts0)
pts1 = np.array(m_kpts1)

print(pts0, pts1, epipolar_error, sep="\n\n")

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(pts0, pts1, color="lime", lw=0.2)
viz2d.add_text(0, f'testing ', fs=20)

# plt.show()
plt.savefig('my_plot.png')

# Plot images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
viz2d.plot_images([image0, image1])

# Plot matches with color based on errors
viz2d.plot_colored_matches(pts0, pts1, epipolar_error, color_map=viz2d.cm_RdGn, lw=1.5, ps=4, a=1.0)

# Add text annotation
viz2d.add_text(0, 'Testing points from dict', fs=20)

# Display the plot
# plt.show()
plt.savefig('colouringtrees.png')

