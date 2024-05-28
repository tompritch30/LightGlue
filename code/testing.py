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


all_epipolar_errors = []

for i in range(1):
    imageIdx0 = i
    imageIdx1 = i+1

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

    pose1, pose2 = poses.get_pose_values(pose_file, imageIdx0, imageIdx1)

    # print("Pose Matrix 1:\n", pose_matrix1)
    # print("Pose Matrix 2:\n", pose_matrix2)

    # m_kpts0 = np.array(m_kpts0)
    # m_kpts1 = np.array(m_kpts1)

    # No reshape
    # SuperGlue Epipolar Error with functions:
    #  [9.40385145e-02 8.41636584e-04 5.44638362e-05 1.60499261e-03
    #  4.39575870e-03 3.36463937e-01 2.26170936e-01 3.32395459e-01
    #  2.80005481e-02 6.10881653e-03 5.90647047e-03 6.77905921e-02
    #  1.84237719e-02] --numpy is the same

    ### SuperGlue poses
    ## Convert to numpy arrays
    ### NEED TO CHECK IF REQUIRED
    m_kpts0 = m_kpts0.numpy().reshape(-1, 2)
    m_kpts1 = m_kpts1.numpy().reshape(-1, 2)

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

    ## Using SuperGlue geometry functions
    # Compute transformation matrix T_0to1 if not already available
    # T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1

    # Calculate epipolar error using the SuperGlue method
    # image0_shape = (480, 640)  # Height, Width
    # image1_shape = (480, 640)
    # scales = [1.0, 1.0]
    # rot0 = 0
    # rot1 = 0

    T_0to1, K0, K1 = poses.compute_transform_with_adjustments(pose1, pose2) # , image0_shape, image1_shape, scales, rot0, rot1)

    # print("Here is the input to epipolar", m_kpts0, m_kpts1, T_0to1, K0, poses.K, sep="\n\n")
    epipolar_error = superglueUtils.compute_epipolar_error(m_kpts0, m_kpts1, T_0to1, poses.K, poses.K)
    print("SuperGlue Epipolar Error with functions:\n", epipolar_error)

    all_epipolar_errors.append(epipolar_error)
    #
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

# # Flatten the list of epipolar errors
# all_epipolar_errors = np.concatenate(all_epipolar_errors)
#
# # Save the epipolar errors to a file
# output_file = "epipolar_errors_six_offset.txt"
# np.savetxt(output_file, all_epipolar_errors, delimiter=",")
#
# # Calculate and print statistics
# average_error = np.mean(all_epipolar_errors)
# min_error = np.min(all_epipolar_errors)
# max_error = np.max(all_epipolar_errors)
#
# print(f"Average Epipolar Error: {average_error}")
# print(f"Minimum Epipolar Error: {min_error}")
# print(f"Maximum Epipolar Error: {max_error}")