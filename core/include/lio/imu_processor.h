// IMU static initialisation, forward propagation and scan de-skewing.
#pragma once
#include <memory>
#include <vector>
#include "lio/ieskf.h"
#include "lio/types.h"

namespace lio
{
struct ImuNoise
{
    double acc_cov = 0.01, gyro_cov = 0.01, acc_bias_cov = 1e-4, gyro_bias_cov = 1e-4;
    bool proper_bias_rw = false; // continuous random walk: bias variance grows as cov*dt (default: cov*dt^2)
};

class IMUProcessor
{
public:
    IMUProcessor(std::shared_ptr<kf::IESKF> kf, const ImuNoise &noise,
                 const Eigen::Matrix3d &rot_ext, const Eigen::Vector3d &pos_ext, bool align_gravity);

    // False while initialising. On success `out` is the scan de-skewed into the end-of-scan LiDAR frame.
    bool process(const MeasureGroup &meas, PointCloud::Ptr &out);
    // Factor converting raw accelerometer units (g or m/s^2) to m/s^2; 1 until initialised.
    double accScale() const { return mean_acc_.norm() > 1e-6 ? kf::GRAVITY / mean_acc_.norm() : 1.0; }

private:
    struct ImuPose
    {
        double offset;
        Eigen::Vector3d acc, gyro, vel, pos;
        Eigen::Matrix3d rot;
    };
    void init(const MeasureGroup &meas);
    void undistort(const MeasureGroup &meas, PointCloud::Ptr &out);

    std::shared_ptr<kf::IESKF> kf_;
    kf::Matrix12d Q_ = kf::Matrix12d::Identity();
    Eigen::Matrix3d rot_ext_;
    Eigen::Vector3d pos_ext_;
    bool align_gravity_;

    bool initialized_ = false;
    int init_count_ = 0;
    const int max_init_count_ = 20;
    Eigen::Vector3d mean_acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gyro_ = Eigen::Vector3d::Zero();
    IMU last_imu_;
    Eigen::Vector3d last_acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero();
    double last_lidar_time_end_ = 0.0;
    std::vector<ImuPose> imu_poses_;
};
} // namespace lio
