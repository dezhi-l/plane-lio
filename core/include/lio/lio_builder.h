// LIO front end: IMU propagation + IESKF. Residual is point-to-plane (default) or GICP.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "lio/imu_processor.h"
#include "lio/voxel_map.h"
#include "lio/dynamic_filter.h"
#include "lio/motion_obs.h"

namespace lio
{
enum class Residual { POINT_TO_PLANE, GICP };

struct LIOParams
{
    Residual residual = Residual::POINT_TO_PLANE;
    double scan_resolution = 0.5;
    double map_resolution = 0.5;
    size_t map_capacity = 5000000;
    size_t grid_capacity = 20;
    VoxelMap::MODE near_mode = VoxelMap::MODE::NEARBY_1;
    size_t max_iterations = 10;
    // point-to-plane
    double p2plane_gain = 1000.0;     // information of a 1 m residual (1/sigma^2)
    double p2plane_robust_delta = 0.1; // Cauchy kernel scale [m]
    double plane_thresh = 0.1;        // max point-to-fitted-plane distance of the kNN [m]
    int knn = 5;
    // gicp
    double gicp_gain = 100.0;
    // degeneracy-aware update (X-ICP / solution-remapping style): LiDAR information along
    // eigen-directions weaker than degen_ratio * strongest (per 3x3 trans / rot block) is dropped,
    // so the IMU prediction constrains those directions. 0 disables.
    double degen_ratio = 0.0;
    // Wheel / NHC / ZUPT / bias-limit observations fused in the iterated update (see motion_obs.h).
    MotionParams motion;
    // Online dynamic-point filter (range-image appear-in-front check) before map insert.
    DynamicFilterParams dynamic;
    // keyframe insertion into the local map
    double key_rot_thresh = 0.18;  // rad
    double key_pos_thresh = 0.5;   // m
    ImuNoise imu_noise;
    Eigen::Matrix3d imu_ext_rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d imu_ext_pos = Eigen::Vector3d::Zero();
    bool extrinsic_est_en = false;
    bool align_gravity = true;
};

class LIOBuilder
{
public:
    explicit LIOBuilder(const LIOParams &params);

    bool process(const MeasureGroup &meas);
    const kf::State &state() const { return kf_->x(); }
    PointCloud::Ptr cloudLidar() const { return cloud_lidar_; }
    PointCloud::Ptr cloudBody() const;
    PointCloud::Ptr cloudWorld() const;

private:
    void measurement(const kf::State &s, kf::MeasurementSystem &m);
    void pointToPlane(const kf::State &s, kf::MeasurementSystem &m);
    void projectDegenerate(kf::MeasurementSystem &m);
    void gicp(const kf::State &s, kf::MeasurementSystem &m);
    void updateMotionInput(const MeasureGroup &meas);
    void addJacobianExtrinsic(const kf::State &s, const Eigen::Vector3d &p_lidar, Eigen::Matrix<double, -1, 12> &J) const;

    struct Plane
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point;  // lidar frame
        Eigen::Vector3d normal; // world frame
        double d = 0;           // n.x + d = 0
        bool valid = false;
    };
    struct GICPCorr
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d mean_a, mean_b;
        Eigen::Matrix3d cov_a, cov_b;
    };

    LIOParams params_;
    bool map_initialized_ = false;
    std::shared_ptr<kf::IESKF> kf_;
    std::unique_ptr<IMUProcessor> imu_;
    std::unique_ptr<VoxelMap> map_;
    std::unique_ptr<FastVoxelMap> scan_filter_;
    PointCloud::Ptr cloud_lidar_;
    std::vector<PointCov> scan_points_;
    std::vector<Plane, Eigen::aligned_allocator<Plane>> planes_;
public:
    size_t degenerate_updates = 0;
    size_t dynamic_dropped = 0;
    size_t zupt_scans = 0; // scans on which the wheel reported standstill and ZUPT was applied

private:
    std::unique_ptr<DynamicFilter> dynamic_;
    MotionInput motion_in_;
    int standstill_count_ = 0;
    std::vector<GICPCorr, Eigen::aligned_allocator<GICPCorr>> gicp_corr_;
    Eigen::Matrix3d key_rot_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d key_pos_ = Eigen::Vector3d::Zero();
    size_t frame_count_ = 0;
};

Residual parseResidual(const std::string &s);
} // namespace lio
