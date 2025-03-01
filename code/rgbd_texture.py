import cv2
import transforms3d as t3d
import open3d as o3d
import numpy as np
from tqdm import tqdm

from load_data import data_synch


def camera2robot_transform():
    K = np.array([[585.05, 0, 242.941],
                  [0, 585.05, 315.84],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    camera_position = np.array([0.18, 0.005, 0.36]) 
    camera_orientation = np.array([0, 0.36, 0.021]) 
    R = t3d.euler.euler2mat(camera_orientation[0], camera_orientation[1], camera_orientation[2], 'sxyz')
    T_C2R = np.zeros((4, 4))
    T_C2R[:3, :3] = R
    T_C2R[:3, 3] = camera_position
    T_C2R[3, 3] = 1
    return T_C2R, K_inv


def img2pc(disparity_file, rgb_file):
    disparity_img = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED)
    bgr_img = cv2.imread(rgb_file)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    dd = (-0.00304 * disparity_img) + 3.31
    depth = 1.03 / dd

    h, w = disparity_img.shape
    mesh = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')  
    i_idxs = mesh[0].flatten()
    j_idxs = mesh[1].flatten()

    rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051, dtype=np.int32)  # force int for indexing
    rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051, dtype=np.int32)

    # some may be out of bounds, just clip them
    rgb_i = np.clip(rgb_i, 0, h - 1)
    rgb_j = np.clip(rgb_j, 0, w - 1)

    colors = rgb_img[rgb_i, rgb_j]

    T_C2R, K_inv = camera2robot_transform()

    homo_pixel_coords = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)]) # (3, h * w)
    pixel_in_camera_coords = K_inv @ homo_pixel_coords * depth.flatten() # (3, h * w)
    oRr = np.array([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
    pixel_in_camera_coords = oRr.T @ pixel_in_camera_coords
    homo_pixel_in_camera_coords = np.vstack((pixel_in_camera_coords, np.ones_like(pixel_in_camera_coords[0]))) # (4, h * w)
    homo_pc_robot_coord = T_C2R @ homo_pixel_in_camera_coords # (4, h * w)
    pc_robot = homo_pc_robot_coord[:3, :].T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_robot)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # get the ground(z < -0.6 and distance < 5)
    mask = (pc_robot[:, 2] < 0.3) & (np.linalg.norm(pc_robot[:, :2], axis=1) < 20)
    pc_robot = pc_robot[mask]
    colors = colors[mask]

    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(pc_robot)
    pcd_ground.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd, pcd_ground

def match_rgbd2lidar(lidar_stamps, disp_stamps, rgb_stamps, dataset_index):
    disparity_folder = f'../data/dataRGBD/Disparity{dataset_index}/'
    rgb_folder = f'../data/dataRGBD/RGB{dataset_index}/'

    aligned_ground_pointcloud = [None] * len(lidar_stamps)
    aligned_rgb_pointcloud = [None] * len(lidar_stamps)
    
    matched_disp_ts, matched_disp_idx, mask = data_synch(rgb_stamps, disp_stamps, 1 + np.arange(disp_stamps.shape[0])[np.newaxis, ...], threshold=0.1)
    matched_lidar_ts, matched_lidar_idx = data_synch(rgb_stamps[mask], lidar_stamps, np.arange(lidar_stamps.shape[0])[np.newaxis, ...])
    matched_rgb_idx = np.where(mask)[0] + 1

    matched_disp_idx = matched_disp_idx.reshape(-1)
    matched_lidar_idx = matched_lidar_idx.reshape(-1)

    print(matched_rgb_idx.shape, matched_lidar_idx.shape, matched_disp_idx.shape)
    for i in tqdm(range(1, len(rgb_stamps[mask]), 5)):
        rgb_idx = matched_rgb_idx[i]
        lidar_idx = matched_lidar_idx[i]
        disp_idx = matched_disp_idx[i]

        disparity_filename = f'{disparity_folder}disparity{dataset_index}_{int(disp_idx)}.png'
        rgb_filename = f'{rgb_folder}rgb{dataset_index}_{int(rgb_idx)}.png'

        pcd, pcd_ground = img2pc(disparity_filename, rgb_filename)
        aligned_ground_pointcloud[lidar_idx] = pcd_ground
        aligned_rgb_pointcloud[lidar_idx] = pcd

    return aligned_ground_pointcloud, aligned_rgb_pointcloud

