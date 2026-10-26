// Multi-session map merging.
// 1) Global initial alignment of session B into session A: gravity is shared, so only
//    (x, y, z, yaw) is searched; candidates = A keyframe positions x yaw bins, scored by
//    nearest-neighbour inlier ratio, best few refined with ICP.
// 2) Inter-session loop closures (radius search + ICP), then a joint batch pose graph
//    (Levenberg-Marquardt) over both sessions with A anchored.
#pragma once
#include "lio/keyframe.h"

namespace lio
{
struct MergeParams
{
    double map_leaf = 0.3;
    int seed_keyframes = 10;       // B keyframes used for the initial alignment
    int yaw_bins = 24;
    int candidate_stride = 3;      // use every n-th A keyframe as a position hypothesis
    double inlier_dist = 0.4;
    double icp_max_corr = 1.0;
    double icp_fitness_thresh = 0.3;
    double loop_radius = 5.0;
    int loop_stride = 3;           // try inter-session loops every n-th B keyframe
    int submap_half_window = 10;
};

struct MergeResult
{
    bool success = false;
    Eigen::Matrix4d T_a_b = Eigen::Matrix4d::Identity(); // initial alignment
    double init_fitness = 0;
    int inter_loops = 0;
    Keyframes merged; // A followed by B, global poses in A's frame
    size_t size_a = 0;
};

MergeResult mergeSessions(const Keyframes &a, const Keyframes &b, const MergeParams &p);
} // namespace lio
