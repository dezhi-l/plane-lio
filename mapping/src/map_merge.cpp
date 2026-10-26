#include "lio/map_merge.h"
#include <algorithm>
#include <iostream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Geometry>

namespace lio
{
namespace
{
gtsam::Pose3 toPose3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

auto odomNoise()
{
    return gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
}

PointCloud::Ptr seedCloud(const Keyframes &b, int n, double leaf)
{
    PointCloud::Ptr c(new PointCloud);
    const Eigen::Matrix3d R0 = b[0].global_rot;
    const Eigen::Vector3d t0 = b[0].global_pos;
    for (int i = 0; i < std::min<int>(n, b.size()); i++)
        *c += *transformCloud(*b[i].cloud, R0.transpose() * b[i].global_rot, R0.transpose() * (b[i].global_pos - t0));
    return voxelDownsample(c, leaf);
}

struct Cand
{
    double score;
    Eigen::Matrix4f T;
};

std::vector<int> subsample(const PointCloud &cloud, int cap)
{
    std::vector<int> sample;
    const int step = std::max<int>(1, static_cast<int>(cloud.size()) / cap);
    for (int i = 0; i < static_cast<int>(cloud.size()); i += step)
        sample.push_back(i);
    return sample;
}

int inliers(const PointCloud &seed, const std::vector<int> &sample, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
            pcl::KdTreeFLANN<PointType> &tree, double dist)
{
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    int n = 0;
    for (int i : sample)
    {
        const auto &q = seed.points[i];
        const Eigen::Vector3d w = R * Eigen::Vector3d(q.x, q.y, q.z) + t;
        PointType pw;
        pw.x = w.x(), pw.y = w.y(), pw.z = w.z();
        if (tree.nearestKSearch(pw, 1, idx, d2) > 0 && d2[0] < dist * dist)
            ++n;
    }
    return n;
}

std::vector<Cand> yawHypotheses(const Keyframes &a, const PointCloud::Ptr &seed, pcl::KdTreeFLANN<PointType> &tree,
                                const Eigen::Matrix3d &tilt, const MergeParams &p)
{
    const std::vector<int> sample = subsample(*seed, 2000);
    std::vector<Cand> cands;
    for (size_t k = 0; k < a.size(); k += p.candidate_stride)
        for (int y = 0; y < p.yaw_bins; y++)
        {
            const double yaw = 2 * M_PI * y / p.yaw_bins;
            const Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * tilt;
            const Eigen::Vector3d t = a[k].global_pos;
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = R.cast<float>();
            T.block<3, 1>(0, 3) = t.cast<float>();
            cands.push_back({double(inliers(*seed, sample, R, t, tree, p.inlier_dist)) / sample.size(), T});
        }
    std::sort(cands.begin(), cands.end(), [](const Cand &x, const Cand &y) { return x.score > y.score; });
    return cands;
}

Eigen::Matrix4d refineBest(const std::vector<Cand> &cands, const PointCloud::Ptr &seed, const PointCloud::Ptr &map,
                           double max_corr, double &best_fit)
{
    best_fit = 1e9;
    Eigen::Matrix4d best = Eigen::Matrix4d::Identity();
    for (size_t i = 0; i < std::min<size_t>(8, cands.size()); i++)
    {
        Eigen::Matrix4d T;
        double fit;
        if (alignICP(seed, map, cands[i].T, max_corr, true, T, fit) && fit < best_fit)
        {
            best_fit = fit;
            best = T;
        }
    }
    return best;
}

void addOdometry(gtsam::NonlinearFactorGraph &g, gtsam::Values &v, const Keyframes &all, size_t na)
{
    const auto odo = odomNoise();
    for (size_t i = 0; i < all.size(); i++)
    {
        v.insert(i, toPose3(all[i].global_rot, all[i].global_pos));
        if (i == 0)
            g.add(gtsam::PriorFactor<gtsam::Pose3>(0, toPose3(all[0].global_rot, all[0].global_pos),
                                                   gtsam::noiseModel::Isotropic::Variance(6, 1e-12)));
        if (i != 0 && i != na)
            g.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1, i,
                                                     toPose3(all[i - 1].global_rot, all[i - 1].global_pos)
                                                         .between(toPose3(all[i].global_rot, all[i].global_pos)),
                                                     odo));
    }
}

int addInterLoops(gtsam::NonlinearFactorGraph &g, const Keyframes &all, size_t na, const MergeParams &p)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pos(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < na; i++)
        pos->emplace_back(all[i].global_pos.x(), all[i].global_pos.y(), all[i].global_pos.z());
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(pos);
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    int nloops = 0;
    for (size_t j = na; j < all.size(); j += p.loop_stride)
    {
        const pcl::PointXYZ q(all[j].global_pos.x(), all[j].global_pos.y(), all[j].global_pos.z());
        if (tree.nearestKSearch(q, 1, idx, d2) < 1 || d2[0] > p.loop_radius * p.loop_radius)
            continue;
        const int i = idx[0];
        const PointCloud::Ptr src = buildMap(all, p.map_leaf, j, j);
        const PointCloud::Ptr tgt = buildMap(all, p.map_leaf, std::max<int>(0, i - p.submap_half_window),
                                             std::min<int>(static_cast<int>(na) - 1, i + p.submap_half_window));
        Eigen::Matrix4d T;
        double fit;
        if (!alignICP(src, tgt, Eigen::Matrix4f::Identity(), p.icp_max_corr, true, T, fit) || fit > p.icp_fitness_thresh)
            continue;
        const Eigen::Matrix3d Rc = T.block<3, 3>(0, 0) * all[j].global_rot;
        const Eigen::Vector3d tc = T.block<3, 3>(0, 0) * all[j].global_pos + T.block<3, 1>(0, 3);
        g.add(gtsam::BetweenFactor<gtsam::Pose3>(
            i, j, toPose3(all[i].global_rot, all[i].global_pos).between(toPose3(Rc, tc)),
            gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
                                              gtsam::noiseModel::Isotropic::Variance(6, std::max(fit, 1e-3)))));
        ++nloops;
    }
    return nloops;
}

Keyframes transformedB(const Keyframes &b, const Eigen::Matrix4d &T)
{
    Keyframes out;
    out.reserve(b.size());
    for (Keyframe k : b)
    {
        k.global_rot = T.block<3, 3>(0, 0) * k.global_rot;
        k.global_pos = T.block<3, 3>(0, 0) * k.global_pos + T.block<3, 1>(0, 3);
        out.push_back(k);
    }
    return out;
}
} // namespace

MergeResult mergeSessions(const Keyframes &a, const Keyframes &b, const MergeParams &p)
{
    MergeResult res;
    res.size_a = a.size();
    if (a.empty() || b.empty())
        return res;
    const PointCloud::Ptr map_a = buildMap(a, p.map_leaf);
    const PointCloud::Ptr seed = seedCloud(b, p.seed_keyframes, p.map_leaf);
    pcl::KdTreeFLANN<PointType> tree;
    tree.setInputCloud(map_a);

    const Eigen::Matrix3d Rb0 = b[0].global_rot;
    const double yaw0 = std::atan2(Rb0(1, 0), Rb0(0, 0));
    const Eigen::Matrix3d tilt = Eigen::AngleAxisd(-yaw0, Eigen::Vector3d::UnitZ()).toRotationMatrix() * Rb0;
    const std::vector<Cand> cands = yawHypotheses(a, seed, tree, tilt, p);
    double best_fit = 0;
    const Eigen::Matrix4d best_T = refineBest(cands, seed, map_a, p.icp_max_corr, best_fit);
    std::cout << "[merge] best coarse inlier ratio " << cands[0].score << ", ICP fitness " << best_fit << std::endl;
    res.init_fitness = best_fit;
    if (best_fit > p.icp_fitness_thresh)
    {
        std::cout << "[merge] initial alignment failed" << std::endl;
        return res;
    }

    Eigen::Matrix4d T_b_b0 = Eigen::Matrix4d::Identity();
    T_b_b0.block<3, 3>(0, 0) = Rb0;
    T_b_b0.block<3, 1>(0, 3) = b[0].global_pos;
    res.T_a_b = best_T * T_b_b0.inverse();

    Keyframes all(a.begin(), a.end());
    const Keyframes moved = transformedB(b, res.T_a_b);
    all.insert(all.end(), moved.begin(), moved.end());
    const size_t na = a.size();
    gtsam::NonlinearFactorGraph g;
    gtsam::Values values;
    addOdometry(g, values, all, na);
    res.inter_loops = addInterLoops(g, all, na, p);
    if (res.inter_loops == 0)
        g.add(gtsam::PriorFactor<gtsam::Pose3>(na, toPose3(all[na].global_rot, all[na].global_pos),
                                               gtsam::noiseModel::Isotropic::Variance(6, 1e-6)));
    const gtsam::Values est = gtsam::LevenbergMarquardtOptimizer(g, values).optimize();
    for (size_t i = 0; i < all.size(); i++)
    {
        const auto P = est.at<gtsam::Pose3>(i);
        all[i].global_rot = P.rotation().matrix();
        all[i].global_pos = P.translation();
    }
    std::cout << "[merge] inter-session loops: " << res.inter_loops << std::endl;
    res.merged = std::move(all);
    res.success = true;
    return res;
}
} // namespace lio
