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

basePath = "C:/Users/thoma/OneDrive/2023 Masters/Project/ProjectCode/Data/TarTanAir-P002-Forest"
images = Path(f"{basePath}/image_left")
torch.set_grad_enabled(False)

## Load extractor and matcher module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

imageIdx0 = 10
imageIdx1 = 11

image0_filename = f"{imageIdx0:06d}_left.png"
image1_filename = f"{imageIdx1:06d}_left.png"

image0 = load_image(images / image0_filename)
image1 = load_image(images / image1_filename)

# ########## SuperPoint #############
# # Fields for SuperPoint:
# # "descriptor_dim": 256,"nms_radius": 4, "max_num_keypoints": None, "detection_threshold": 0.0005, "remove_borders": 4,
# extractor = SuperPoint(max_num_keypoints=64, descriptor_dim=256).eval().to(device)  # load the extractor
#
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
#
# # torch.save(feats0, 'feats0.pth')
# # torch.save(feats1, 'feats1.pth')
#
# # Feats 0 is a dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
# # keypoints is tensor([[ [287.3125, 304.8125], ... ]]) i.e. is a coordinate for each point
# # score: tensor([[0.5934, 0.5182, 0.5137, 0.4981]]) i.e. confidence score 0-1 for all
# # descriptor : tensor([[[ 0.0531,w/ 256 elements], ]]) is a 2D list : [[ [256 elements], [256 elements] etc ... ]]
# # image_size, Value: tensor([[640., 480.]]) - in px
#
# # print(type(feats0))
# # print(feats0.keys())
# # for key, value in feats0.items():
# #     if isinstance(value, (list, tuple)):
# #         print(f"Key: {key}, Sample Value: {value[:5]}")  # Print first 5 elements if it's a list or tuple
# #     elif isinstance(value, np.ndarray):  # Assuming you are using NumPy arrays
# #         print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
# #     else:
# #         print(f"Key: {key}, Value: {value}")
# # print(feats0, "\n\n", feats0)
#
#
# ############ LightGlue #########
# # feats0 = torch.load('feats0.pth')
# # feats1 = torch.load('feats1.pth')
#
# # # Fields for LightGlue
# # # {"name": "lightglue", "input_dim": 256, "descriptor_dim": 256, "add_scale_ori": False, "n_layers": 9,
# # #     "num_heads": 4, "flash": True, "mp": False, "depth_confidence": 0.95, "width_confidence": 0.99,
# # #     "filter_threshold": 0.1, "weights": None  }
# matcher = LightGlue(features="superpoint").eval().to(device)
#
# # Do feature mapping
# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension
#
# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#
# # torch.save(m_kpts0, 'm_kpts0.pth')
# # torch.save(m_kpts1, 'm_kpts1.pth')
# # Is a list of coordinates from superglue for matches
#
# # m_kpts0 = torch.load('m_kpts0.pth')
# # m_kpts1 = torch.load('m_kpts1.pth')
#
# ######### Poses ###########
# ### Ground truth poses
# pose_file = f"{basePath}/pose_left.txt"
# pose_matrix1, pose_matrix2 = poses.get_pose_matrices(pose_file, imageIdx0, imageIdx1)
#
# print("Pose Matrix 1:\n", pose_matrix1)
# print("Pose Matrix 2:\n", pose_matrix2)
#
# ### SuperGlue poses
# ## Convert to numpy arrays
# m_kpts0 = m_kpts0.numpy().reshape(-1, 2)
# m_kpts1 = m_kpts1.numpy().reshape(-1, 2)
#
# ## Triangulate points
# points_3d = poses.triangulate_points(m_kpts0, m_kpts1, pose_matrix1, pose_matrix2, poses.K)
# print("Triangulated 3D Points:\n", points_3d)
#
# ## Reproject points to the image planes
# reprojected_kpts0 = poses.reproject_points(points_3d, pose_matrix1, poses.K)
# reprojected_kpts1 = poses.reproject_points(points_3d, pose_matrix2, poses.K)
#
# print("Reprojected Keypoints Image 0:\n", reprojected_kpts0)
# print("Reprojected Keypoints Image 1:\n", reprojected_kpts1)
#
# # Compute reprojection errors
# error0 = poses.compute_reprojection_error(m_kpts0, reprojected_kpts0)
# error1 = poses.compute_reprojection_error(m_kpts1, reprojected_kpts1)
#
# print("Reprojection Error Image 0:\n", error0)
# print("Reprojection Error Image 1:\n", error1)
#
# ## Epipolar error
# # https://github.com/johannes-graeter/momo
# # Compute the Essential matrix (as an example)
# # E, _ = cv2.findEssentialMat(m_kpts0, m_kpts1, poses.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
# # # Compute epipolar errors
# # epipolar_error = poses.compute_epipolar_error(E, m_kpts0, m_kpts1)
# # print("Normal Epipolar Error:\n", epipolar_error)
# # # Construct transformation matrix T_0to1
# # T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1
# #
# # print(kpts0, kpts1, T_0to1, poses.K)
# #
# # # Use the SuperGlue method to compute epipolar error
# # epipolar_error = poses.compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, poses.K, poses.K)
# # print("My SuperGlue Epipolar Error:\n", epipolar_error)
#
# ## Using SuperGlue geometry functions
# # Normalize keypoints
# norm_kpts0 = (m_kpts0 - poses.K[[0, 1], [2, 2]][None]) / poses.K[[0, 1], [0, 1]][None]
# norm_kpts1 = (m_kpts1 - poses.K[[0, 1], [2, 2]][None]) / poses.K[[0, 1], [0, 1]][None]
#
# # Compute transformation matrix T_0to1 if not already available
# T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1
#
# # Calculate epipolar error using the SuperGlue method
# epipolar_error = superglueUtils.compute_epipolar_error(norm_kpts0, norm_kpts1, T_0to1, poses.K, poses.K)
# print("SuperGlue Epipolar Error with fnctions:\n", epipolar_error)
#
# # Create error dictionaries for each image
# error_dict_image0 = {}
# error_dict_image1 = {}
# for idx, error in enumerate(epipolar_error):
#     keypoint0 = tuple(m_kpts0[idx])
#     keypoint1 = tuple(m_kpts1[idx])
#     error_dict_image0[keypoint0] = error
#     error_dict_image1[keypoint1] = error
#
# print("Error Dictionary Image 0:", error_dict_image0)
# print("Error Dictionary Image 1:", error_dict_image1)
#
# # Plotting normal graph
# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=1)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
#
# pts0 = np.array([list(pt) for pt in error_dict_image0.keys()])
# pts1 = np.array([list(pt) for pt in error_dict_image1.keys()])
#
# errors = list(error_dict_image0.values())

# print("\n\n\nhere are the errors\n:", pts0, pts1, errors, sep="\n")
pts0 = np.array([[137.9375, 64.1875], [191.6875, 59.1875], [345.4375, 228.5625], [148.5625, 280.4375],
        [159.1875, 47.9375], [576.6875, 115.4375], [520.4375, 396.6875], [175.4375, 37.9375],
        [456.0625, 297.9375], [264.1875, 106.0625], [538.5625, 439.8125], [301.0625, 40.4375],
        [251.0625, 62.9375], [577.9375, 87.3125], [347.9375, 11.6875], [247.9375, 79.8125],
        [439.8125, 7.9375], [457.9375, 112.9375]])

pts1 = np.array([[99.8125, 69.8125], [162.9375, 59.1875], [311.6875, 251.6875], [119.8125, 280.4375],
        [139.1875, 141.0625], [476.0625, 118.5625], [457.3125, 377.3125], [145.4375, 42.9375],
        [354.8125, 288.5625], [361.0625, 134.1875], [416.6875, 364.8125], [229.1875, 41.0625],
        [220.4375, 59.1875], [476.0625, 76.0625], [296.0625, 22.3125], [231.0625, 63.5625],
        [493.5625, 16.0625], [449.8125, 184.1875]])

errors = [0.002907528241603695, 0.0028924842188165186, 0.002880773161212839, 0.0028800448095319545,
          0.0028697429621220316, 0.002983036980895951, 0.002913153503132172, 0.00289568311886416,
          0.0029783167523154777, 0.0026981768636036556, 0.0030059265704460757, 0.0029546556814259994,
          0.0028933542788950996, 0.0029885241513693834, 0.002923248583335889, 0.0028733440246676307,
          0.0027632819479406325, 0.002839756308815569]

print(len(pts0), len(pts1), len(errors))

# Set up plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot images
viz2d.plot_images([image0, image1]) #, axes=axes)

# Plot matches with color based on errors
viz2d.plot_colored_matches(pts0, pts1, errors, axes=axes)

# Add text annotation
# viz2d.add_text(axes[0], 'Testing points from dict', fs=20)

# Display the plot
plt.show()

# viz2d.plot_colored_matches(pts0, pts1, errors)
# plt.show()
#
#
# axis = viz2d.plot_images([image0, image1])
# viz2d.plot_colored_matches(pts0, pts1, errors)
# # viz2d.plot_matches(pts0, pts1, color="red", lw=1)
# viz2d.add_text(0, f'Testing points from dict', fs=20)

# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
# plt.show()
