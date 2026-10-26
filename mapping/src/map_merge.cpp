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
#include <pcl/registration/icp.h>

namespace lio
{
namespace
{
gtsam::Pose3 toPose3(const Eigen::Matrix3d &R, const Eigen::Vector3d &t) { return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t)); }

bool icpAlign(const PointCloud::Ptr &src, const PointCloud::Ptr &tgt, const Eigen::Matrix4f &guess, double max_corr,
              Eigen::Matrix4d &T, double &fitness)
{
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(max_corr);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    PointCloud out;
    icp.align(out, guess);
    fitness = icp.getFitnessScore(max_corr);
    T = icp.getFinalTransformation().cast<double>();
    return icp.hasConverged();
}

// Submap of B's first keyframes expressed relative to B's first keyframe pose.
PointCloud::Ptr seedCloud(const Keyframes &b, int n, double leaf)
{
    PointCloud::Ptr c(new PointCloud);
    const Eigen::Matrix3d R0 = b[0].global_rot;
    const Eigen::Vector3d t0 = b[0].global_pos;
    for (int i = 0; i < std::min<int>(n, b.size()); i++)
        *c += *transformCloud(*b[i].cloud, R0.transpose() * b[i].global_rot, R0.transpose() * (b[i].global_pos - t0));
    return voxelDownsample(c, leaf);
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

    // --- 1. coarse search over (A keyframe position, yaw), keeping B0's roll/pitch (gravity aligned)
    const Eigen::Matrix3d Rb0 = b[0].global_rot;
    const double yaw_b0 = std::atan2(Rb0(1, 0), Rb0(0, 0));
    const Eigen::Matrix3d Rb0_tilt = Eigen::AngleAxisd(-yaw_b0, Eigen::Vector3d::UnitZ()).toRotationMatrix() * Rb0;
    struct Cand { double score; Eigen::Matrix4f T; };
    std::vector<Cand> cands;
    std::vector<int> idx(1);
    std::vector<float> d2(1);
    // subsample the seed for scoring
    std::vector<int> sample;
    for (int i = 0; i < int(seed->size()); i += std::max<int>(1, seed->size() / 2000))
        sample.push_back(i);
    for (size_t k = 0; k < a.size(); k += p.candidate_stride)
        for (int y = 0; y < p.yaw_bins; y++)
        {
            const double yaw = 2 * M_PI * y / p.yaw_bins;
            Eigen::Matrix3d R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * Rb0_tilt;
            Eigen::Vector3d t = a[k].global_pos;
            int inl = 0;
            for (int i : sample)
            {
                const auto &q = seed->points[i];
                Eigen::Vector3d w = R * Eigen::Vector3d(q.x, q.y, q.z) + t;
                PointType pw;
                pw.x = w.x(), pw.y = w.y(), pw.z = w.z();
                if (tree.nearestKSearch(pw, 1, idx, d2) > 0 && d2[0] < p.inlier_dist * p.inlier_dist)
                    inl++;
            }
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = R.cast<float>();
            T.block<3, 1>(0, 3) = t.cast<float>();
            cands.push_back({double(inl) / sample.size(), T});
        }
    std::sort(cands.begin(), cands.end(), [](const Cand &x, const Cand &y) { return x.score > y.score; });

    // --- 2. refine the best hypotheses with ICP
    double best_fit = 1e9;
    Eigen::Matrix4d best_T = Eigen::Matrix4d::Identity();
    for (size_t i = 0; i < std::min<size_t>(8, cands.size()); i++)
    {
        Eigen::Matrix4d T;
        double fit;
        if (icpAlign(seed, map_a, cands[i].T, p.icp_max_corr, T, fit) && fit < best_fit)
        {
            best_fit = fit;
            best_T = T;
        }
    }
    std::cout << "[merge] best coarse inlier ratio " << cands[0].score << ", ICP fitness " << best_fit << std::endl;
    res.init_fitness = best_fit;
    if (best_fit > p.icp_fitness_thresh)
    {
        std::cout << "[merge] initial alignment failed" << std::endl;
        return res;
    }
    // T_seed->A ; seed frame = B0 frame  => T_a_b = T * inv(T_b_b0)
    Eigen::Matrix4d T_b_b0 = Eigen::Matrix4d::Identity();
    T_b_b0.block<3, 3>(0, 0) = Rb0;
    T_b_b0.block<3, 1>(0, 3) = b[0].global_pos;
    res.T_a_b = best_T * T_b_b0.inverse();

    // --- 3. joint pose graph
    Keyframes all(a.begin(), a.end());
    for (Keyframe k : b)
    {
        k.global_rot = res.T_a_b.block<3, 3>(0, 0) * k.global_rot;
        k.global_pos = res.T_a_b.block<3, 3>(0, 0) * k.global_pos + res.T_a_b.block<3, 1>(0, 3);
        all.push_back(k);
    }
    const size_t na = a.size();
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    auto odo = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-6).finished());
    for (size_t i = 0; i < all.size(); i++)
    {
        v.insert(i, toPose3(all[i].global_rot, all[i].global_pos));
        const bool first_of_session = (i == 0 || i == na);
        if (i == 0)
            g.add(gtsam::PriorFactor<gtsam::Pose3>(0, toPose3(all[0].global_rot, all[0].global_pos),
                                                   gtsam::noiseModel::Isotropic::Variance(6, 1e-12)));
        if (!first_of_session) // sessions keep their own (already loop-closed) internal structure
            g.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1, i, toPose3(all[i - 1].global_rot, all[i - 1].global_pos)
                                                                  .between(toPose3(all[i].global_rot, all[i].global_pos)), odo));
    }
    // initial alignment as a weak link B0 <-> nearest A keyframe is implied by the loops below.
    pcl::PointCloud<pcl::PointXYZ>::Ptr pos_a(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < na; i++)
        pos_a->emplace_back(all[i].global_pos.x(), all[i].global_pos.y(), all[i].global_pos.z());
    pcl::KdTreeFLANN<pcl::PointXYZ> ptree;
    ptree.setInputCloud(pos_a);
    for (size_t j = na; j < all.size(); j += p.loop_stride)
    {
        pcl::PointXYZ q(all[j].global_pos.x(), all[j].global_pos.y(), all[j].global_pos.z());
        if (ptree.nearestKSearch(q, 1, idx, d2) < 1 || d2[0] > p.loop_radius * p.loop_radius)
            continue;
        const int i = idx[0];
        PointCloud::Ptr src = buildMap(all, p.map_leaf, j, j);
        PointCloud::Ptr tgt = buildMap(all, p.map_leaf, std::max<int>(0, i - p.submap_half_window),
                                       std::min<int>(na - 1, i + p.submap_half_window));
        Eigen::Matrix4d T;
        double fit;
        if (!icpAlign(src, tgt, Eigen::Matrix4f::Identity(), p.icp_max_corr, T, fit) || fit > p.icp_fitness_thresh)
            continue;
        const Eigen::Matrix3d Rc = T.block<3, 3>(0, 0) * all[j].global_rot;
        const Eigen::Vector3d tc = T.block<3, 3>(0, 0) * all[j].global_pos + T.block<3, 1>(0, 3);
        g.add(gtsam::BetweenFactor<gtsam::Pose3>(i, j, toPose3(all[i].global_rot, all[i].global_pos).between(toPose3(Rc, tc)),
                                                 gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1.0),
                                                                                   gtsam::noiseModel::Isotropic::Variance(6, std::max(fit, 1e-3)))));
        res.inter_loops++;
    }
    if (res.inter_loops == 0)
    {   // fall back to the rigid initial alignment only: anchor B0
        g.add(gtsam::PriorFactor<gtsam::Pose3>(na, toPose3(all[na].global_rot, all[na].global_pos),
                                               gtsam::noiseModel::Isotropic::Variance(6, 1e-6)));
    }
    const gtsam::Values est = gtsam::LevenbergMarquardtOptimizer(g, v).optimize();
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
