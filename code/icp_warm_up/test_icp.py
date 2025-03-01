
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from icp import icp

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose, you need to estimate the pose with ICP
    # pose = np.eye(4)
    # assume that the initial pose guess only has rotation with respect to one axis

    best_dist_lst = float('inf')
    best_init = None
    for _ in range(100):
      init_pose = np.eye(4)
      axis = np.random.uniform(0, 1, 3)
      axis = axis / np.linalg.norm(axis)
      theta = np.random.uniform(0, 2*np.pi)
      init_pose[:3, :3] = rotation_matrix(axis, theta)
      pose, dist = icp(source_pc, target_pc, init_pose=init_pose, max_iter=50, tolerance=1e-5)
      if dist[-1] < best_dist_lst:
        best_dist_lst = dist[-1]
        best_init = init_pose

    pose, dist = icp(source_pc, target_pc, init_pose=best_init, max_iter=200, tolerance=1e-5)

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc, pose, f'../../figures/{obj_name}_{i}.png')

