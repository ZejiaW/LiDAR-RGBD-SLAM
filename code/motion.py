import numpy as np
from icp import icp

def pose2SE2(pose):
    """
    Convert a pose to a SE2 matrix.
    """
    return np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                    [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                    [0, 0, 1]])

def delta_yaw_imu(imu_stamps, imu_angular_velocity, t0, t1):
    """
    Compute the change in yaw between two timestamps using the IMU angular velocity data.

    Args:
        imu_stamps (numpy.ndarray): Array of timestamps. (N, )
        imu_angular_velocity (numpy.ndarray): Array of angular velocities. (N, )
        t0 (float): Start timestamp.
        t1 (float): End timestamp.

    Returns:
        float: Change in yaw between t0 and t1.
    """
    assert t0 < t1, "t0 must be less than t1"
    time_window = imu_angular_velocity[np.where((imu_stamps >= t0) & (imu_stamps <= t1))]
    if len(time_window) == 0:
        return 0.0
    else:
        return np.mean(time_window) * (t1 - t0)
    

def encoder_imu_odometry(synched_encoder_counts_2lidar, imu_angular_velocity, lidar_stamps, imu_stamps):
    """
    Compute the odometry pose using the encoder counts and IMU angular velocity.

    Args:
        synched_encoder_counts_2lidar (numpy.ndarray): Array of encoder counts. (4, N)
        imu_angular_velocity (numpy.ndarray): Array of angular velocities. (M, )
        lidar_stamps (numpy.ndarray): Array of timestamps. (N, )
        imu_stamps (numpy.ndarray): Array of timestamps. (M, )

    Returns:
        numpy.ndarray: Array of SE(2) poses. (N, 3, 3)
    """
    odom_pose = [[0.0, 0.0, 0.0]]
    for i in range(1, len(lidar_stamps)):
        pose = odom_pose[-1]

        d_r = (synched_encoder_counts_2lidar[0][i] + synched_encoder_counts_2lidar[2][i]) / 2 * 0.0022
        d_l = (synched_encoder_counts_2lidar[1][i] + synched_encoder_counts_2lidar[3][i]) / 2 * 0.0022

        pose[0] += (d_l + d_r) / 2 * np.cos(pose[2])
        pose[1] += (d_l + d_r) / 2 * np.sin(pose[2])
        pose[2] += delta_yaw_imu(imu_stamps, imu_angular_velocity[2], lidar_stamps[i-1], lidar_stamps[i])

        if pose[2] > np.pi:
            pose[2] -= 2 * np.pi
        elif pose[2] < -np.pi:
            pose[2] += 2 * np.pi

        odom_pose.append(pose.copy())
    
    odom_poses = [pose2SE2(pose) for pose in odom_pose]
    
    return np.array(odom_poses)

def lidar_scan_matching_odometry(lidar_points, encoder_imu_odom_poses=None):
    """
    Compute the odometry pose using the LiDAR scan matching.

    Args:
        lidar_points (numpy.ndarray): Array of LiDAR points. (N, num_ranges, 2)
        encoder_imu_odom_poses (numpy.ndarray): Array of SE(2) poses. (N, 3, 3)

    Returns:
        numpy.ndarray: Array of SE(2) poses. (N, 3, 3)
    """
    T = np.eye(3)
    odom_poses = [T]
    for i in range(lidar_points.shape[0]-1):
        source_pc = lidar_points[i]
        source_pc = source_pc[~np.isnan(source_pc[:, 0]) & ~np.isnan(source_pc[:, 1])]
        target_pc = lidar_points[i+1]
        target_pc = target_pc[~np.isnan(target_pc[:, 0]) & ~np.isnan(target_pc[:, 1])]
        min_size = min(source_pc.shape[0], target_pc.shape[0])
        source_pc = source_pc[:min_size]
        target_pc = target_pc[:min_size]
        init_pose = np.linalg.inv(encoder_imu_odom_poses[i]).dot(encoder_imu_odom_poses[i+1]) if encoder_imu_odom_poses is not None else None
        pose, _ = icp(target_pc, source_pc, init_pose=init_pose, max_iter=20, tolerance=1e-5)
        T = np.dot(T, pose)
        odom_poses.append(T)
    return np.array(odom_poses)
