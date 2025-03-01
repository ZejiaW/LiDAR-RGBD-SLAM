import gtsam
import numpy as np
from tqdm import tqdm
from icp import icp
from motion import pose2SE2

def add_factor(graph, i, j, pose, noise=0.1):
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noise, noise, noise]))
    relative_pose = gtsam.Pose2(pose[0, 2], pose[1, 2], np.arctan2(pose[1, 0], pose[0, 0]))
    factor = gtsam.BetweenFactorPose2(i, j, relative_pose, noise)
    graph.add(factor)

def find_nearest_pose(odom_poses, target_pose):
    distances = np.linalg.norm(odom_poses - target_pose, axis=(1, 2))
    return np.argmin(distances)

def process_pc(source_idx, target_idx, lidar_points):
    source_pc = lidar_points[source_idx]
    source_pc = source_pc[~np.isnan(source_pc[:, 0]) & ~np.isnan(source_pc[:, 1])]
    target_pc = lidar_points[target_idx]
    target_pc = target_pc[~np.isnan(target_pc[:, 0]) & ~np.isnan(target_pc[:, 1])]
    min_size = min(source_pc.shape[0], target_pc.shape[0])
    source_pc, target_pc = source_pc[:min_size], target_pc[:min_size]
    return source_pc, target_pc


def fixed_interval_graph_slam(encoder_imu_odom_poses, lidar_points, interval=10):
    """
    Fixed interval graph slam.

    Args:
        encoder_imu_odom_poses (numpy.ndarray): Array of SE(2) poses. (N, 3, 3)
        lidar_points (numpy.ndarray): Array of LiDAR points. (N, num_ranges, 2)
        interval (int): Interval of the fixed interval graph slam.

    Returns:
        numpy.ndarray: Array of SE(2) poses. (N, 3, 3)
    """
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise))

    T = np.eye(3)
    odom_poses = [T]
    for i in tqdm(range(1, len(lidar_points))):
        source_pc, target_pc = process_pc(i-1, i, lidar_points)
        init_pose = np.linalg.inv(encoder_imu_odom_poses[i-1]).dot(encoder_imu_odom_poses[i])
        pose, _ = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)
        T = np.dot(T, pose)
        odom_poses.append(T)

        add_factor(graph, i, i+1, pose)

        if i % interval == 0 and i > interval:
            source_pc, target_pc = process_pc(i-interval, i, lidar_points)
            init_pose = np.linalg.inv(encoder_imu_odom_poses[i-interval]).dot(encoder_imu_odom_poses[i])
            pose, dist = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)

            if dist[-1] < 0.1:
                add_factor(graph, i-interval, i, pose)
    
    initial_estimate = gtsam.Values()
    for i in range(1, len(odom_poses)+1):
        initial_pose = gtsam.Pose2(odom_poses[i-1][0, 2], odom_poses[i-1][1, 2], \
                                   np.arctan2(odom_poses[i-1][1, 0], odom_poses[i-1][0, 0]))
        initial_estimate.insert(i, initial_pose)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    res = []
    for i in range(1, len(odom_poses)+1):
        res.append(pose2SE2([result.atPose2(i).x(), result.atPose2(i).y(), result.atPose2(i).theta()]))

    return np.array(res)



def proximity_graph_slam(encoder_imu_odom_poses, 
                         lidar_points, 
                         fixed_closure_interval=30,
                         proximity_closure_interval=100,
                         proximity_closure_step=800,
                         optimization_interval=1000,
                         proximity_threshold=0.2):
    """
    Proximity graph slam.
    """
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
    graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise))

    T = np.eye(3)
    odom_poses = [T]

    for i in tqdm(range(1, len(lidar_points))):
        source_pc, target_pc = process_pc(i-1, i, lidar_points)
        init_pose = np.linalg.inv(encoder_imu_odom_poses[i-1]).dot(encoder_imu_odom_poses[i])
        pose, distances = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)
        if distances[-1] < proximity_threshold / 2:
            T = np.dot(T, pose)
            add_factor(graph, i, i+1, pose)
        else:
            T = np.dot(T, init_pose)
        odom_poses.append(T)
        add_factor(graph, i, i+1, init_pose, noise=0.05)

        if i % fixed_closure_interval == 0 and i > fixed_closure_interval:
            source_pc, target_pc = process_pc(i-fixed_closure_interval, i, lidar_points)
            init_pose = np.linalg.inv(encoder_imu_odom_poses[i-fixed_closure_interval]).dot(encoder_imu_odom_poses[i])
            pose, distances = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)
            if distances[-1] < proximity_threshold / 2:
                add_factor(graph, i-fixed_closure_interval, i, pose)

        if i % proximity_closure_interval == 0 and i > proximity_closure_step:
            nearest_pose_idx = find_nearest_pose(odom_poses[:i-proximity_closure_step], odom_poses[i])
            source_pc, target_pc = process_pc(nearest_pose_idx, i, lidar_points)
            init_pose = np.linalg.inv(encoder_imu_odom_poses[nearest_pose_idx]).dot(encoder_imu_odom_poses[i])
            pose, distances = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)
            # print("[Info] Nearest pose index: {}, distance: {}".format(nearest_pose_idx, distances[-1]))
            if distances[-1] < proximity_threshold:
                print("[Info] Loop closure factor between {} and {} added.".format(nearest_pose_idx+1, i))
                add_factor(graph, nearest_pose_idx+1, i, pose)

        if i % optimization_interval == lidar_points.shape[0] % optimization_interval - 1:
            print("[Info] Optimization at iteration {}".format(i))
            initial_estimate = gtsam.Values()
            for i in range(1, len(odom_poses)+1):
                initial_pose = gtsam.Pose2(odom_poses[i-1][0, 2], odom_poses[i-1][1, 2], \
                                           np.arctan2(odom_poses[i-1][1, 0], odom_poses[i-1][0, 0]))
                initial_estimate.insert(i, initial_pose)
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
            result = optimizer.optimize()
    
    initial_estimate = gtsam.Values()
    for i in range(1, len(odom_poses)+1):
        initial_pose = gtsam.Pose2(odom_poses[i-1][0, 2], odom_poses[i-1][1, 2], \
                                   np.arctan2(odom_poses[i-1][1, 0], odom_poses[i-1][0, 0]))
        initial_estimate.insert(i, initial_pose)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    res = []
    for i in range(1, len(odom_poses)+1):
        res.append(pose2SE2([result.atPose2(i).x(), result.atPose2(i).y(), result.atPose2(i).theta()]))
    return np.array(res)
    