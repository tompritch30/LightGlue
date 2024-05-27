import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

# Camera intrinsics (example values)
fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0

# Camera intrinsic matrix
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

def compute_epipolar_error(E, kpts0, kpts1):
    kpts0_hom = cv2.convertPointsToHomogeneous(kpts0).reshape(-1, 3)
    kpts1_hom = cv2.convertPointsToHomogeneous(kpts1).reshape(-1, 3)
    epipolar_lines = E @ kpts0_hom.T
    error = np.sum(kpts1_hom * epipolar_lines.T, axis=1)
    return np.abs(error)
