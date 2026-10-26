// ROS-free basic types shared by all modules.
#pragma once
#include <deque>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lio
{
// x,y,z,intensity; `curvature` stores the per-point time offset from scan start in milliseconds.
using PointType = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

struct IMU
{
    double timestamp = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

// Wheel / chassis odometry sample. Velocity is in the body (base_link) frame, m/s.
struct WheelOdom
{
    double timestamp = 0.0;
    Eigen::Vector3d vel = Eigen::Vector3d::Zero(); // vx forward, vy lateral, vz up
};

// One LiDAR scan plus the IMU (and optional wheel) samples that cover it.
struct MeasureGroup
{
    double lidar_time_begin = 0.0;
    double lidar_time_end = 0.0;
    PointCloud::Ptr lidar;
    std::deque<IMU> imus;
    std::deque<WheelOdom> wheels;
};

// Rigid pose (rotation + translation) with a timestamp.
struct PoseStamped
{
    double time = 0.0;
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
};

PointCloud::Ptr transformCloud(const PointCloud &in, const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
} // namespace lio
