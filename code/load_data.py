import numpy as np

def load_data(dataset_index):
    with np.load("../data/Encoders%d.npz"%dataset_index) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    print("[Info] Encoder data loaded: ")
    print("* Encoder counts shape: ", encoder_counts.shape)
    print("* Encoder stamps shape: ", encoder_stamps.shape)
    print("**************************************************")

    with np.load("../data/Hokuyo%d.npz"%dataset_index) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

    print("[Info] Hokuyo data loaded: ")
    print("* Lidar angle min: ", lidar_angle_min)
    print("* Lidar angle max: ", lidar_angle_max)
    print("* Lidar angle increment: ", lidar_angle_increment)
    print("* Lidar range min: ", lidar_range_min)
    print("* Lidar range max: ", lidar_range_max)
    print("* Lidar ranges shape: ", lidar_ranges.shape)
    print("* Lidar stamps shape: ", lidar_stamps.shape)
    print("**************************************************")

    with np.load("../data/Imu%d.npz"%dataset_index) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements

    print("[Info] Imu data loaded: ")
    print("* Imu angular velocity shape: ", imu_angular_velocity.shape)
    print("* Imu linear acceleration shape: ", imu_linear_acceleration.shape)
    print("* Imu stamps shape: ", imu_stamps.shape)
    print("**************************************************")

    with np.load("../data/Kinect%d.npz"%dataset_index ) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

    print("[Info] Kinect data loaded: ")
    print("* Disparity stamps shape: ", disp_stamps.shape)
    print("* Rgb stamps shape: ", rgb_stamps.shape)
    print("**************************************************")

    return encoder_counts, encoder_stamps, lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps, imu_angular_velocity, imu_linear_acceleration, imu_stamps, disp_stamps, rgb_stamps

def data_synch(anchor_ts, target_ts, target_data, threshold=None):
    """
    Synchronize two datasets based on anchor timestamps.

    Args:
      anchor_ts: anchor timestamps (n, )
      target_ts: target timestamps (m, )
      target_data: target data (..., m)

    Returns:
      target_data: target data (..., n)
    """
    # print(anchor_ts.shape, target_ts.shape, target_data.shape)
    n = anchor_ts.shape[0]
    target_ts = target_ts[np.newaxis, :]
    anchor_ts = anchor_ts[:, np.newaxis]
    # if the minimum difference is bigger than threshold, delete that line
    if threshold is not None:
        diff = np.abs(anchor_ts - target_ts)
        mask = np.min(diff, axis=1) < threshold
        anchor_ts = anchor_ts[mask]
    idx = np.argmin(np.abs(anchor_ts - target_ts), axis=1)
    assert len(idx) == anchor_ts.shape[0], "idx and anchor_data must have the same length"
    if threshold is not None:
        return target_ts[0][idx], target_data[..., idx], mask
    else:
        return target_ts[0][idx], target_data[..., idx]
  

if __name__ == '__main__':
    dataset = 20
    
    with np.load("../data/Encoders%d.npz"%dataset) as data:
        encoder_counts = data["counts"] # 4 x n encoder counts
        encoder_stamps = data["time_stamps"] # encoder time stamps

    with np.load("../data/Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
      
    with np.load("../data/Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    
    with np.load("../data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

