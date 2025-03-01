import os
import scipy.io as sio
import numpy as np
import open3d as o3d


def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  model_fname = os.path.join('./data', model_name, 'model.mat')
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose, save_path):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  # Create a visualizer
  vis = o3d.visualization.Visualizer()
  vis.create_window(visible=False)  # Create a window but do not show it
  vis.add_geometry(source_pcd)
  vis.add_geometry(target_pcd)

  # Capture the screen and save it as a PNG file
  vis.poll_events()
  vis.update_renderer()
  vis.capture_screen_image(save_path)
  vis.destroy_window()

  # o3d.visualization.draw_geometries([source_pcd, target_pcd]).save_to_file(save_path)

  # show the world coordinate frame in the same figure
  # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=target_pc[0])
  # o3d.visualization.draw_geometries([source_pcd, target_pcd, frame]) # there will be only frame


