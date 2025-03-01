import imageio
import os
import numpy as np
from tqdm import tqdm
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt
from PIL import Image



def lidar2pointcloud(
        lidar_ranges,
        lidar_angle_min, 
        lidar_angle_increment, 
        lidar_range_min = 0.1, 
        lidar_range_max = 30.0,
    ):
    """
    Convert the lidar data to a pointcloud.

    Args:
        lidar_ranges: (num_ranges, N)
        lidar_angle_min: float
        lidar_angle_increment: float
        lidar_range_min: float
        lidar_range_max: float

    Returns:
        points: (N, num_ranges, 2)
    """
    # shape of angles: (num_ranges,)
    angles = np.arange(lidar_angle_min, lidar_angle_min + lidar_ranges.shape[0] * lidar_angle_increment, lidar_angle_increment)
    # shape of points: (N, num_ranges, 2)
    points = np.zeros((lidar_ranges.shape[1], lidar_ranges.shape[0], 2))
    points[:, :, 0] = lidar_ranges.T * np.cos(angles)
    points[:, :, 1] = lidar_ranges.T * np.sin(angles)
    mask = (lidar_ranges.T >= lidar_range_min) & (lidar_ranges.T <= lidar_range_max) # (N, num_ranges)
    points = np.where(mask[..., np.newaxis], points, np.nan)
    return points


def gen_occ_map(odom_poses, lidar_points, width=800, resolution=0.1, aligned_ground_pointcloud=None, interval=20):
    """
    Generate the occupancy map from the lidar points and the odom poses.

    Args:
        odom_poses: (N, 3, 3)
        lidar_points: (N, num_ranges, 2)
        width: int
        resolution: float

    Returns:
        prob_map: (width, width)
        colored_map: (width, width, 3)
    """
    lambda_occ = np.log(0.7 / 0.3)
    grid_map = np.zeros((width, width), dtype=np.float32)
    colored_map = np.zeros((width, width, 3), dtype=np.uint8)
    # for odom_pose, lidar_point in zip(odom_poses, lidar_points):
    for i in tqdm(range(1, lidar_points.shape[0], interval)):
        odom_pose = odom_poses[i]
        lidar_point = lidar_points[i]
        # lidar_point: (num_ranges, 2), odom_pose: (3, 3)
        # delete all NaN points in lidar_points
        lidar_point = lidar_point[~np.isnan(lidar_point[:, 0]) & ~np.isnan(lidar_point[:, 1])]
        lidar_points_world = np.dot(odom_pose[:2, :2], lidar_point.T).T + odom_pose[:2, 2]
        points_map = (lidar_points_world / resolution + width / 2).astype(np.int32)

        # current position of the robot
        sx, sy = int(odom_pose[0, 2] / resolution + width / 2), int(odom_pose[1, 2] / resolution + width / 2)
        for point in points_map:
            ex, ey = point
            ray = bresenham2D(sx, sy, ex, ey)
            for x, y in zip(ray[0], ray[1]):
                x, y = int(x), int(y)
                if (0 <= x < width) and (0 <= y < width):
                    grid_map[y, x] -= lambda_occ
            if (0 <= ex < width) and (0 <= ey < width):
                grid_map[ey, ex] += 2 * lambda_occ

        prob_map = np.exp(grid_map) / (1 + np.exp(grid_map))

        if aligned_ground_pointcloud is not None:
            pcd = aligned_ground_pointcloud[i]
            if pcd is not None:
                # print("[Info] projecting ground pointcloud to the occupancy map at index %d." % i)
                points = np.asarray(pcd.points)[:, :2]
                colors = np.asarray(pcd.colors) * 255
                pose = odom_poses[i]
                points_world = np.dot(pose[:2, :2], points.T).T + pose[:2, 2]
                points_map = (points_world / resolution + width / 2).astype(np.int32)
                mask = prob_map[points_map[:, 1], points_map[:, 0]] > - np.inf
                colored_map[points_map[mask, 1], points_map[mask, 0]] = colors[mask]
        
    prob_map = np.exp(grid_map) / (1 + np.exp(grid_map))

    if aligned_ground_pointcloud is not None:
        return prob_map, colored_map
    else:
        return prob_map

def dynamic_occ_map(odom_poses, lidar_points, width=800, resolution=0.1, aligned_ground_pointcloud=None, interval=20, save_path=None):
    """
    Generate the dynamic occupancy map from the lidar points and the odom poses and save the image frames.
    """

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    lambda_occ = np.log(0.7 / 0.3)
    grid_map = np.zeros((width, width), dtype=np.float32)
    colored_map = np.zeros((width, width, 3), dtype=np.uint8)
    # for odom_pose, lidar_point in zip(odom_poses, lidar_points):
    for i in tqdm(range(600, lidar_points.shape[0], interval)):
        odom_pose = odom_poses[i]
        lidar_point = lidar_points[i]
        # lidar_point: (num_ranges, 2), odom_pose: (3, 3)
        # delete all NaN points in lidar_points
        lidar_point = lidar_point[~np.isnan(lidar_point[:, 0]) & ~np.isnan(lidar_point[:, 1])]
        lidar_points_world = np.dot(odom_pose[:2, :2], lidar_point.T).T + odom_pose[:2, 2]
        points_map = (lidar_points_world / resolution + width / 2).astype(np.int32)

        # current position of the robot
        sx, sy = int(odom_pose[0, 2] / resolution + width / 2), int(odom_pose[1, 2] / resolution + width / 2)
        for point in points_map:
            ex, ey = point
            ray = bresenham2D(sx, sy, ex, ey)
            for x, y in zip(ray[0], ray[1]):
                x, y = int(x), int(y)
                if (0 <= x < width) and (0 <= y < width):
                    grid_map[y, x] -= lambda_occ
            if (0 <= ex < width) and (0 <= ey < width):
                grid_map[ey, ex] += 2 * lambda_occ

        prob_map = np.exp(grid_map) / (1 + np.exp(grid_map))

        if aligned_ground_pointcloud is not None:
            pcd = aligned_ground_pointcloud[i]
            if pcd is not None:
                points = np.asarray(pcd.points)[:, :2]
                colors = np.asarray(pcd.colors) * 255
                pose = odom_poses[i]
                points_world = np.dot(pose[:2, :2], points.T).T + pose[:2, 2]
                points_map = (points_world / resolution + width / 2).astype(np.int32)
                mask = prob_map[points_map[:, 1], points_map[:, 0]] > - np.inf
                colored_map[points_map[mask, 1], points_map[mask, 0]] = colors[mask]
        
        # plt.subplot(1, 2, 1)
        # plt.imshow(prob_map, cmap='gray')
        # plt.plot(sx, sy, 'ro')
        # plt.title('Occupancy Grid Map')
        # plt.subplot(1, 2, 2)
        # plt.imshow(colored_map)
        # plt.title('Color Map')
        # plt.pause(0.01)
        
        prob_map_rgb = (prob_map * 255).astype(np.uint8)
        prob_map_rgb = np.stack((prob_map_rgb, prob_map_rgb, prob_map_rgb), axis=-1)
        for j in range(1, i):
            pose = odom_poses[j]
            sx, sy = int(pose[0, 2] / resolution + width / 2), int(pose[1, 2] / resolution + width / 2)
            prob_map_rgb[sy-2:sy+2, sx-2:sx+2] = [255, 0, 0]
        
        map_combined = np.concatenate((prob_map_rgb, colored_map), axis=1)
        imageio.imwrite(os.path.join(save_path, f'frame_{i}.png'), map_combined)


def gen_gif(img_path, save_path, fps=100, frame_skip=10):
    if not os.path.exists('/'.join(save_path.split('/')[:-1])):
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    images = sorted([img for img in os.listdir(img_path) if img.endswith(".png")])
    images = images[::frame_skip]
    gif_images = [Image.open(os.path.join(img_path, image)) for image in images]
    gif_images[0].save(f'{save_path}.gif', 
                       save_all=True, 
                       append_images=gif_images[1:], 
                       optimize=False, 
                       duration=int(1000 / fps), 
                       loop=0)