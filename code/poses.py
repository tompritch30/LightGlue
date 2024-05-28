# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import cv2
#
# # Camera intrinsics (example values)
# fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
# fov = 90  # degrees
# width = 640
# height = 480
#
# """
# Pose file
# The camera pose file is a text file containing the translation and orientation of the camera in a fixed coordinate frame. Note that our automatic evaluation tool expects both the ground truth trajectory and the estimated trajectory to be in this format.
#     Each line in the text file contains a single pose.
#     The number of lines/poses is the same as the number of image frames in that trajectory.
#     The format of each line is 'tx ty tz qx qy qz qw'.
#     tx ty tz (3 floats) give the position of the optical center of the color camera with respect to the world origin in the world frame.
#     qx qy qz qw (4 floats) give the orientation of the optical center of the color camera in the form of a unit quaternion with respect to the world frame.
#     The camera motion is defined in the NED frame. That is to say, the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right, the z-axis is pointing to the camera's downward.
# """
#
# # Tartan Depth data
# # https://github.com/castacks/tartanair_tools/blob/master/data_type.md
# # The unit of the depth value is meter. The baseline between the left and right cameras is 0.25m.
#
# # Code to work out image flow: https://github.com/huyaoyu/ImageFlow
#
# # TartanAir Camera intrinsic matrix
# K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
#
#
# def load_pose_file(filename):
#     """Load the pose file and return it as a numpy array."""
#     return np.loadtxt(filename)
#
#
# def pose_to_matrix(pose):
#     """Convert a pose (tx, ty, tz, qx, qy, qz, qw) to a 4x4 transformation matrix."""
#     tx, ty, tz, qx, qy, qz, qw = pose
#     rotation = R.from_quat([qx, qy, qz, qw])
#     rotation_matrix = rotation.as_matrix()
#
#     transformation_matrix = np.eye(4)
#     transformation_matrix[:3, :3] = rotation_matrix
#     transformation_matrix[:3, 3] = [tx, ty, tz]
#
#     return transformation_matrix
#
#
# def get_pose_matrices(pose_file, idx1, idx2):
#     """Retrieve and convert poses for the specified indices to transformation matrices."""
#     poses = load_pose_file(pose_file)
#     pose1 = poses[idx1]
#     pose2 = poses[idx2]
#
#     pose_matrix1 = pose_to_matrix(pose1)
#     pose_matrix2 = pose_to_matrix(pose2)
#
#     return pose_matrix1, pose_matrix2
#
#
# def triangulate_points(kpts0, kpts1, pose_matrix1, pose_matrix2, K):
#     # Convert the pose matrices to projection matrices
#     P1 = K @ pose_matrix1[:3, :]
#     P2 = K @ pose_matrix2[:3, :]
#
#     # Convert keypoints to homogeneous coordinates
#     kpts0_hom = cv2.convertPointsToHomogeneous(kpts0).reshape(-1, 3)
#     kpts1_hom = cv2.convertPointsToHomogeneous(kpts1).reshape(-1, 3)
#
#     # Triangulate points
#     points_4d_hom = cv2.triangulatePoints(P1, P2, kpts0.T, kpts1.T)
#
#     # Convert from homogeneous to 3D
#     points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)
#
#     return points_3d
#
#
# def reproject_points(points_3d, pose_matrix, K):
#     # Project 3D points to 2D
#     points_2d_hom = K @ (pose_matrix[:3, :] @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T)
#     points_2d = cv2.convertPointsFromHomogeneous(points_2d_hom.T).reshape(-1, 2)
#
#     return points_2d
#
#
# def compute_reprojection_error(kpts, reprojected_kpts):
#     error = np.linalg.norm(kpts - reprojected_kpts, axis=1)
#     return error
#
#
# def compute_epipolar_error(E, kpts0, kpts1):
#     kpts0_hom = cv2.convertPointsToHomogeneous(kpts0).reshape(-1, 3)
#     kpts1_hom = cv2.convertPointsToHomogeneous(kpts1).reshape(-1, 3)
#     epipolar_lines = E @ kpts0_hom.T
#     error = np.sum(kpts1_hom * epipolar_lines.T, axis=1)
#     return np.abs(error)
#
#
# # Define a pair of keypoints in two images
# kpts0 = np.array([[150, 120], [200, 180], [250, 240]], dtype=np.float32)
# kpts1 = np.array([[152, 118], [198, 182], [249, 239]], dtype=np.float32)
#
# # Define example poses
# pose1 = [0, 0, 0, 0, 0, 0, 1]  # Identity quaternion
# pose2 = [1, 0, 0, 0, 0, 1, 0]  # 180-degree rotation around z-axis
# # pose3 = [0, 0, 1, 0, 0, 0, 1]  # Translation along z-axis
#
# pose_matrix1 = pose_to_matrix(pose1)
# pose_matrix2 = pose_to_matrix(pose2)
# # pose_matrix3 = pose_to_matrix(pose3)
#
# # Compute the Essential matrix
# E, _ = cv2.findEssentialMat(kpts0, kpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

# Camera intrinsics (example values)
fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
fov = 90  # degrees
width = 640
height = 480

# TartanAir Camera intrinsic matrix
K = np.array([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])

def load_pose_file(filename):
    """Load the pose file and return it as a numpy array."""
    return np.loadtxt(filename)

def pose_to_matrix(pose):
    """Convert a pose (tx, ty, tz, qx, qy, qz, qw) to a 4x4 transformation matrix."""
    tx, ty, tz, qx, qy, qz, qw = pose
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix

def get_pose_matrices(pose_file, idx1, idx2):
    """Retrieve and convert poses for the specified indices to transformation matrices."""
    poses = load_pose_file(pose_file)
    pose1 = poses[idx1]
    pose2 = poses[idx2]

    pose_matrix1 = pose_to_matrix(pose1)
    pose_matrix2 = pose_to_matrix(pose2)

    return pose_matrix1, pose_matrix2

def triangulate_points(kpts0, kpts1, pose_matrix1, pose_matrix2, K):
    # Convert the pose matrices to projection matrices
    P1 = K @ pose_matrix1[:3, :]
    P2 = K @ pose_matrix2[:3, :]

    # Convert keypoints to homogeneous coordinates
    kpts0_hom = cv2.convertPointsToHomogeneous(kpts0).reshape(-1, 3)
    kpts1_hom = cv2.convertPointsToHomogeneous(kpts1).reshape(-1, 3)

    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P1, P2, kpts0.T, kpts1.T)

    # Convert from homogeneous to 3D
    points_3d = cv2.convertPointsFromHomogeneous(points_4d_hom.T).reshape(-1, 3)

    return points_3d

def reproject_points(points_3d, pose_matrix, K):
    # Project 3D points to 2D
    points_2d_hom = K @ (pose_matrix[:3, :] @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T)
    points_2d = cv2.convertPointsFromHomogeneous(points_2d_hom.T).reshape(-1, 2)

    return points_2d

def compute_reprojection_error(kpts, reprojected_kpts):
    error = np.linalg.norm(kpts - reprojected_kpts, axis=1)
    return error


# def to_homogeneous(points):
#     return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
#
# def compute_epipolar_error(E, kpts0, kpts1):
#     kpts0_hom = cv2.convertPointsToHomogeneous(kpts0).reshape(-1, 3)
#     kpts1_hom = cv2.convertPointsToHomogeneous(kpts1).reshape(-1, 3)
#
#     # Debugging prints
#     print("E shape:", E.shape)
#     print("kpts0_hom shape:", kpts0_hom.shape)
#     print("kpts1_hom shape:", kpts1_hom.shape)
#
#     # Ensure E is 3x3
#     if E.ndim == 2 and E.shape[0] == 3 and E.shape[1] == 3:
#         epipolar_lines = E @ kpts0_hom.T
#     else:
#         raise ValueError("Essential matrix E must be 3x3")
#
#     # Debugging prints
#     print("epipolar_lines shape:", epipolar_lines.shape)
#
#     epipolar_lines = E @ kpts0_hom.T
#
#     errors = []
#     for i in range(kpts1_hom.shape[0]):
#         line = epipolar_lines[:, i]
#         point = kpts1_hom[i, :]
#
#         a, b, c = line
#         x, y, _ = point
#
#         distance = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
#         errors.append(distance)
#
#     return np.array(errors)
#
#     # error = np.sum(kpts1_hom * epipolar_lines.T, axis=1)
#     # return np.abs(error)
#
#
# def compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
#     # Normalize keypoints by the intrinsics
#     kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
#     kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
#
#     # Convert keypoints to homogeneous coordinates
#     kpts0 = to_homogeneous(kpts0)
#     kpts1 = to_homogeneous(kpts1)
#
#     # Extract translation and form the skew-symmetric matrix
#     t0, t1, t2 = T_0to1[:3, 3]
#     t_skew = np.array([
#         [0, -t2, t1],
#         [t2, 0, -t0],
#         [-t1, t0, 0]
#     ])
#
#     # Compute the Essential matrix
#     E = t_skew @ T_0to1[:3, :3]
#
#     # Compute the epipolar lines in the second image
#     Ep0 = kpts0 @ E.T  # N x 3
#
#     # Compute the epipolar constraint
#     p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
#
#     # Compute the epipolar lines in the first image
#     Etp1 = kpts1 @ E  # N x 3
#
#     # Compute the squared epipolar error
#     d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))
#
#     return d

def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

def normalize_keypoints(kpts, K):
    return (kpts - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]

def compute_manual_essential_matrix(T_0to1):
    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]
    return E

def compute_epipolar_error(E, kpts0, kpts1):
    kpts0_hom = to_homogeneous(kpts0)
    kpts1_hom = to_homogeneous(kpts1)

    epipolar_lines = E @ kpts0_hom.T

    errors = []
    for i in range(kpts1_hom.shape[0]):
        line = epipolar_lines[:, i]
        point = kpts1_hom[i, :]

        a, b, c = line
        x, y, _ = point

        distance = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
        errors.append(distance)

    return np.array(errors)

def compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = normalize_keypoints(kpts0, K0)
    kpts1 = normalize_keypoints(kpts1, K1)

    kpts0_hom = to_homogeneous(kpts0)
    kpts1_hom = to_homogeneous(kpts1)

    E = compute_manual_essential_matrix(T_0to1)

    Ep0 = kpts0_hom @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1_hom * Ep0, -1)  # N
    Etp1 = kpts1_hom @ E  # N x 3

    d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))

    return d

# # Define a pair of keypoints in two images
# kpts0 = np.array([
#     [150, 120], [200, 180], [250, 240], [300, 300], [350, 350],
#     [400, 100], [450, 200], [500, 250], [550, 300], [600, 350]
# ], dtype=np.float32)
#
# kpts1 = np.array([
#     [152, 118], [198, 182], [249, 239], [298, 302], [348, 352],
#     [402, 98], [448, 202], [502, 248], [552, 298], [602, 348]
# ], dtype=np.float32)
#
# # Define example poses
# pose1 = [0, 0, 0, 0, 0, 0, 1]  # Identity quaternion
# pose2 = [1, 0, 0, 0, 0, 1, 0]  # 180-degree rotation around z-axis
#
# pose_matrix1 = pose_to_matrix(pose1)
# pose_matrix2 = pose_to_matrix(pose2)
#
# # Construct transformation matrix T_0to1
# T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1
#
# # Normalize keypoints
# kpts0_normalized = normalize_keypoints(kpts0, K)
# kpts1_normalized = normalize_keypoints(kpts1, K)
#
# # Compute Essential matrix manually
# E_manual = compute_manual_essential_matrix(T_0to1)
#
# # Compute epipolar errors using original method
# epipolar_error = compute_epipolar_error(E_manual, kpts0_normalized, kpts1_normalized)
# print("Epipolar Error:\n", epipolar_error)
#
# # Use the SuperGlue method to compute epipolar error
# superglue_epipolar_error = compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, K, K)
# print("SuperGlue Epipolar Error:\n", superglue_epipolar_error)

#
# # Compute the Essential matrix
# E, _ = cv2.findEssentialMat(kpts0, kpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#
# # Compute epipolar error
# epipolar_error = compute_epipolar_error(E, kpts0, kpts1)
# print("Epipolar Error:\n", epipolar_error)
#
# # Construct transformation matrix T_0to1
# T_0to1 = np.linalg.inv(pose_matrix2) @ pose_matrix1
#
# # Use the SuperGlue method to compute epipolar error
# epipolar_error = compute_superglue_epipolar_error(kpts0, kpts1, T_0to1, K, K)
# print("SuperGlue Epipolar Error:\n", epipolar_error)


# [1.95825578e-11 1.88975946e-10 1.00000000e+00 2.22712515e-10
#  2.66140887e-10 9.43117584e-10 1.41943701e-09 1.17239551e-09
#  1.33820777e-09 1.50583901e-09]

# Epipolar Error:
#  [2.76939188e-11 2.67252346e-10 1.41421356e+00 3.14963059e-10
#  3.76380052e-10 1.33376968e-09 2.00738707e-09 1.65801764e-09
#  1.89251157e-09 2.12957795e-09]



# # Triangulate points using the poses
# points_3d = triangulate_points(kpts0, kpts1, pose_matrix1, pose_matrix2, K)
# print("Triangulated 3D Points:\n", points_3d)
#
# # Reproject the 3D points back to the image planes
# reprojected_kpts0 = reproject_points(points_3d, pose_matrix1, K)
# reprojected_kpts1 = reproject_points(points_3d, pose_matrix2, K)
#
# print("Reprojected Keypoints Image 0:\n", reprojected_kpts0)
# print("Reprojected Keypoints Image 1:\n", reprojected_kpts1)
#
# # Compute reprojection errors
# reprojection_error0 = compute_reprojection_error(kpts0, reprojected_kpts0)
# reprojection_error1 = compute_reprojection_error(kpts1, reprojected_kpts1)
#
# print("Reprojection Error Image 0:\n", reprojection_error0)
# print("Reprojection Error Image 1:\n", reprojection_error1)
