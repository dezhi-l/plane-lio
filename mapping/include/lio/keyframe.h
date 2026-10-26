// Keyframe container shared by loop closure and multi-session map merge.
#pragma once
#include <string>
#include <vector>
#include "lio/types.h"

namespace lio
{
struct Keyframe
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    Eigen::Matrix3d local_rot = Eigen::Matrix3d::Identity();  // LIO (odometry) pose
    Eigen::Vector3d local_pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d global_rot = Eigen::Matrix3d::Identity(); // pose-graph optimised pose
    Eigen::Vector3d global_pos = Eigen::Vector3d::Zero();
    PointCloud::Ptr cloud;                                    // body frame, downsampled
};
using Keyframes = std::vector<Keyframe, Eigen::aligned_allocator<Keyframe>>;

// Simple on-disk session format: <dir>/keyframes.tum (global poses) + <dir>/keyframes.bin (clouds).
void saveSession(const std::string &dir, const Keyframes &kfs);
Keyframes loadSession(const std::string &dir);
void writeTUM(const std::string &file, const std::vector<PoseStamped> &poses);
PointCloud::Ptr voxelDownsample(const PointCloud::Ptr &in, double leaf);
// Concatenate keyframe clouds in their global poses.
PointCloud::Ptr buildMap(const Keyframes &kfs, double leaf, size_t first = 0, size_t last = size_t(-1));
} // namespace lio
