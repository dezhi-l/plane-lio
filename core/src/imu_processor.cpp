#include "lio/imu_processor.h"
#include <algorithm>
#include <sophus/so3.hpp>
#include <Eigen/Geometry>

namespace lio
{
IMUProcessor::IMUProcessor(std::shared_ptr<kf::IESKF> kf, const ImuNoise &n,
                           const Eigen::Matrix3d &rot_ext, const Eigen::Vector3d &pos_ext, bool align_gravity)
    : kf_(std::move(kf)), rot_ext_(rot_ext), pos_ext_(pos_ext), align_gravity_(align_gravity)
{
    Q_.block<3, 3>(0, 0).diagonal().setConstant(n.gyro_cov);
    Q_.block<3, 3>(3, 3).diagonal().setConstant(n.acc_cov);
    Q_.block<3, 3>(6, 6).diagonal().setConstant(n.gyro_bias_cov);
    Q_.block<3, 3>(9, 9).diagonal().setConstant(n.acc_bias_cov);
}

void IMUProcessor::init(const MeasureGroup &meas)
{
    // Static initialisation: average acc/gyro to get gravity direction and gyro bias.
    for (const auto &imu : meas.imus)
    {
        init_count_++;
        mean_acc_ += (imu.acc - mean_acc_) / init_count_;
        mean_gyro_ += (imu.gyro - mean_gyro_) / init_count_;
    }
    if (meas.imus.empty() || init_count_ < max_init_count_)
        return;
    initialized_ = true;

    kf::State &s = kf_->x();
    s.rot_ext = rot_ext_;
    s.pos_ext = pos_ext_;
    s.bg = mean_gyro_;
    if (align_gravity_)
    {   // rotate the world frame so that gravity is -z
        s.rot = Eigen::Quaterniond::FromTwoVectors((-mean_acc_).normalized(), Eigen::Vector3d(0, 0, -1)).toRotationMatrix();
        s.initG(Eigen::Vector3d(0, 0, -1));
    }
    else
        s.initG(-mean_acc_);

    kf::Matrix23d &P = kf_->P();
    P.setIdentity();
    P.block<3, 3>(6, 6) *= 1e-5;   // rot_ext
    P.block<3, 3>(9, 9) *= 1e-5;   // pos_ext
    P.block<3, 3>(15, 15) *= 1e-4; // bg
    P.block<3, 3>(18, 18) *= 1e-3; // ba
    P.block<2, 2>(21, 21) *= 1e-5; // gravity
    last_imu_ = meas.imus.back();
    last_lidar_time_end_ = meas.lidar_time_end;
}

void IMUProcessor::undistort(const MeasureGroup &meas, PointCloud::Ptr &out)
{
    std::deque<IMU> imus(meas.imus.begin(), meas.imus.end());
    imus.push_front(last_imu_);
    const double acc_scale = kf::GRAVITY / mean_acc_.norm(); // handles acc in g or m/s^2

    out = meas.lidar;
    std::sort(out->points.begin(), out->points.end(),
              [](const PointType &a, const PointType &b) { return a.curvature < b.curvature; });

    kf::State s = kf_->x();
    imu_poses_.clear();
    imu_poses_.push_back({0.0, last_acc_, last_gyro_, s.vel, s.pos, s.rot});

    kf::Input inp;
    for (size_t i = 0; i + 1 < imus.size(); i++)
    {
        const IMU &head = imus[i], &tail = imus[i + 1];
        if (tail.timestamp < last_lidar_time_end_)
            continue;
        inp.gyro = 0.5 * (head.gyro + tail.gyro);
        inp.acc = 0.5 * (head.acc + tail.acc) * acc_scale; // upstream used head.acc twice (typo)
        const double dt = tail.timestamp - std::max(head.timestamp, last_lidar_time_end_);
        kf_->predict(inp, dt, Q_);
        s = kf_->x();
        last_gyro_ = inp.gyro - s.bg;
        last_acc_ = s.rot * (inp.acc - s.ba) + s.g;
        imu_poses_.push_back({tail.timestamp - meas.lidar_time_begin, last_acc_, last_gyro_, s.vel, s.pos, s.rot});
    }
    // propagate to scan end
    kf_->predict(inp, meas.lidar_time_end - imus.back().timestamp, Q_);
    last_imu_ = imus.back();
    last_lidar_time_end_ = meas.lidar_time_end;

    s = kf_->x();
    if (out->empty())
        return;
    // Back-propagate every point to the scan-end LiDAR frame.
    auto it_pcl = out->points.end() - 1;
    for (auto it = imu_poses_.end() - 1; it != imu_poses_.begin(); --it)
    {
        const ImuPose &head = *(it - 1), &tail = *it;
        for (; it_pcl->curvature / 1000.0 > head.offset; --it_pcl)
        {
            const double dt = it_pcl->curvature / 1000.0 - head.offset;
            const Eigen::Vector3d p(it_pcl->x, it_pcl->y, it_pcl->z);
            const Eigen::Matrix3d R_i = head.rot * Sophus::SO3d::exp(tail.gyro * dt).matrix();
            const Eigen::Vector3d t_i = head.pos + head.vel * dt + 0.5 * tail.acc * dt * dt;
            const Eigen::Vector3d pc = s.rot_ext.transpose() *
                (s.rot.transpose() * (R_i * (s.rot_ext * p + s.pos_ext) + t_i - s.pos) - s.pos_ext);
            it_pcl->x = pc.x();
            it_pcl->y = pc.y();
            it_pcl->z = pc.z();
            if (it_pcl == out->points.begin())
                return;
        }
    }
}

bool IMUProcessor::process(const MeasureGroup &meas, PointCloud::Ptr &out)
{
    if (!initialized_)
    {
        init(meas);
        return false;
    }
    undistort(meas, out);
    return true;
}

} // namespace lio
