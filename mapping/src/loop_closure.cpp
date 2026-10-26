#include "lio/loop_closure.h"
#include <iostream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sophus/so3.hpp>
#include <Eigen/Geometry>

namespace lio
{
namespace
{
gtsam::Pose3 toPose3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) { return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t)); }
} // namespace

LoopClosure::LoopClosure(const LoopParams &p) : params_(p)
{
    gtsam::ISAM2Params ip;
    ip.relinearizeThreshold = 0.01;
    ip.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(ip);
    graph_ = std::make_unique<gtsam::NonlinearFactorGraph>();
    init_ = std::make_unique<gtsam::Values>();
}
LoopClosure::~LoopClosure() = default;

void LoopClosure::toGlobal(const Eigen::Matrix3d &r, const Eigen::Vector3d &t, Eigen::Matrix3d &gr, Eigen::Vector3d &gt) const
{
    gr = offset_rot_ * r;
    gt = offset_rot_ * t + offset_pos_;
}

bool LoopClosure::addPose(double time, const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const PointCloud::Ptr &cloud_body)
{
    if (!kfs_.empty())
    {
        const Keyframe &last = kfs_.back();
        const Eigen::Vector3d dp = last.local_rot.transpose() * (pos - last.local_pos);
        const Eigen::Matrix3d dR = last.local_rot.transpose() * rot;
        const double dr = Sophus::SO3d(Eigen::Quaterniond(dR).normalized()).log().norm();
        if (dp.norm() < params_.key_dist_thresh && dr < params_.key_rot_thresh)
            return false;
    }
    Keyframe k;
    k.time = time;
    k.local_rot = rot;
    k.local_pos = pos;
    toGlobal(rot, pos, k.global_rot, k.global_pos);
    k.cloud = voxelDownsample(cloud_body, params_.keyframe_leaf);
    kfs_.push_back(k);

    const int cur = int(kfs_.size()) - 1;
    init_->insert(cur, toPose3(k.global_rot, k.global_pos));
    if (cur == 0)
        graph_->add(gtsam::PriorFactor<gtsam::Pose3>(0, toPose3(k.global_rot, k.global_pos),
                                                     gtsam::noiseModel::Isotropic::Variance(6, 1e-12)));
    else
    {
        const Keyframe &a = kfs_[cur - 1];
        const gtsam::Pose3 rel = toPose3(a.local_rot, a.local_pos).between(toPose3(rot, pos));
        auto noise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
        graph_->add(gtsam::BetweenFactor<gtsam::Pose3>(cur - 1, cur, rel, noise));
    }

    bool loop = false;
    int pre;
    Eigen::Matrix4d T;
    double score;
    if (params_.enable && detectLoop(cur, pre, T, score))
    {
        // T maps the current keyframe cloud (in its global pose) onto the submap around `pre`.
        const Eigen::Matrix3d Rc = T.block<3, 3>(0, 0) * kfs_[cur].global_rot;
        const Eigen::Vector3d tc = T.block<3, 3>(0, 0) * kfs_[cur].global_pos + T.block<3, 1>(0, 3);
        const gtsam::Pose3 rel = toPose3(kfs_[pre].global_rot, kfs_[pre].global_pos).between(toPose3(Rc, tc));
        graph_->add(gtsam::BetweenFactor<gtsam::Pose3>(pre, cur, rel, gtsam::noiseModel::Isotropic::Variance(6, score)));
        loops_.emplace_back(pre, cur);
        std::cout << "[loop] " << pre << " <-> " << cur << " fitness " << score << std::endl;
        loop = true;
    }
    optimize(loop);
    return true;
}

bool LoopClosure::detectLoop(int cur, int &pre, Eigen::Matrix4d &T_pre_cur, double &score)
{
    if (cur < params_.index_thresh)
        return false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr positions(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &k : kfs_)
        positions->emplace_back(k.global_pos.x(), k.global_pos.y(), k.global_pos.z());
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(positions);
    std::vector<int> ids;
    std::vector<float> d2;
    tree.radiusSearch(positions->back(), params_.search_radius, ids, d2); // sorted by distance
    pre = -1;
    for (int id : ids)
        if (kfs_[cur].time - kfs_[id].time > params_.time_thresh && cur - id >= params_.index_thresh)
        {
            pre = id;
            break;
        }
    if (pre < 0)
        return false;

    PointCloud::Ptr src = buildMap(kfs_, params_.submap_leaf, cur, cur);
    PointCloud::Ptr tgt = buildMap(kfs_, params_.submap_leaf, std::max(0, pre - params_.submap_half_window),
                                   std::min(cur - 1, pre + params_.submap_half_window));
    if (!alignICP(src, tgt, Eigen::Matrix4f::Identity(), params_.icp_max_corr, false, T_pre_cur, score) ||
        score > params_.icp_fitness_thresh)
        return false;
    return true;
}

void LoopClosure::optimize(bool loop_found)
{
    isam_->update(*graph_, *init_);
    isam_->update();
    if (loop_found)
        for (int i = 0; i < 5; i++)
            isam_->update();
    graph_->resize(0);
    init_->clear();
    const gtsam::Values est = isam_->calculateEstimate();
    for (size_t i = 0; i < kfs_.size(); i++)
    {
        const auto p = est.at<gtsam::Pose3>(i);
        kfs_[i].global_rot = p.rotation().matrix();
        kfs_[i].global_pos = p.translation();
    }
    const Keyframe &b = kfs_.back();
    offset_rot_ = b.global_rot * b.local_rot.transpose();
    offset_pos_ = b.global_pos - offset_rot_ * b.local_pos;
}
} // namespace lio
