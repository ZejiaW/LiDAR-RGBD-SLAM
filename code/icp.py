import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd

def kabsch(m, z, weights=None):
    """
    m and z are N x dim matrices (dim-D points)
    weights is an N-dimensional vector of weights
    returns the rotation matrix R and translation vector p
    """
    dims = m.shape[1]
    # compute centroids
    m_centroid = np.mean(m, axis=0)
    z_centroid = np.mean(z, axis=0)
    if weights is None:
        weights = np.ones(m.shape[0])
    # compute covariance matrix
    Q = np.zeros((dims, dims))
    for i in range(m.shape[0]):
        Q += weights[i] * np.outer(m[i] - m_centroid, z[i] - z_centroid)
    # compute SVD of Q
    U, D, V = svd(Q)
    V = V.T
    # compute the determinant of UV^T
    det = np.linalg.det(U @ V.T)
    # compute the rotation matrix
    if dims == 2:
        R = U @ np.array([[1, 0], [0, det]]) @ V.T
    else:
        R = U @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]]) @ V.T
    # compute the translation vector
    p = m_centroid - R @ z_centroid

    T = np.eye(dims + 1)
    T[:dims, :dims] = R
    T[:dims, dims] = p

    return T, R, p

def find_nearest_neighbor(src_points, dest_points):
    neighbor_finder = NearestNeighbors(n_neighbors=1)
    neighbor_finder.fit(dest_points)
    distances, indices = neighbor_finder.kneighbors(src_points, return_distance=True)
    return distances.ravel(), indices.ravel()

def icp(src_pc, target_pc, init_pose=None, max_iter=100, tolerance=1e-5):
    """
    ICP alforithm for point cloud registration with unknown correspondences

    Args:
        src_pc: (N, dim)
        target_pc: (N, dim)
        init_pose: (dim+1, dim+1)
        max_iter: int
        tolerance: float

    Returns:
        T: (dim+1, dim+1)
        dist: list of distance between src_pc and target_pc
    """

    dims = src_pc.shape[1]

    src_pc_homo = np.hstack([src_pc, np.ones((src_pc.shape[0], 1))]) # (N, dim+1)
    target_pc_homo = np.hstack([target_pc, np.ones((target_pc.shape[0], 1))]) # (N, dim+1)
    if init_pose is not None:
        src_pc_homo = np.dot(init_pose, src_pc_homo.T).T # (N, dim+1)

    dist = []
    prev_error = 0

    for i in range(max_iter):
        distances, indices = find_nearest_neighbor(src_pc_homo[:, :dims], target_pc_homo[:, :dims])
        T, R, p = kabsch(target_pc_homo[indices, :dims], src_pc_homo[:, :dims])
        src_pc_homo = np.dot(T, src_pc_homo.T).T # (N, dim+1)
        mean_dist = np.mean(distances)
        dist.append(mean_dist)
        if np.abs(prev_error - mean_dist) < tolerance:
            break
        prev_error = mean_dist

    T, R, p = kabsch(src_pc_homo[:, :dims], src_pc)
    return T, dist
        
            