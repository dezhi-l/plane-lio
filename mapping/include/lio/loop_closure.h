// Back end: keyframe selection, radius-search loop detection verified by ICP,
// and incremental pose-graph optimisation with GTSAM iSAM2.
#pragma once
#include <memory>
#include <utility>
#include "lio/keyframe.h"

namespace gtsam { class ISAM2; class NonlinearFactorGraph; class Values; }

namespace lio
{
struct LoopParams
{
    bool enable = true;
    double key_rot_thresh = 0.2;   // rad, keyframe selection
    double key_dist_thresh = 1.0;  // m
    double keyframe_leaf = 0.2;    // downsampling of stored keyframe clouds
    double time_thresh = 30.0;     // s, min time gap of a loop candidate
    int index_thresh = 25;         // min keyframe-index gap
    double search_radius = 10.0;   // m
    int submap_half_window = 25;   // keyframes on each side of the candidate
    double submap_leaf = 0.3;
    double icp_max_corr = 2.0;     // m
    double icp_fitness_thresh = 0.3;
};

class LoopClosure
{
public:
    explicit LoopClosure(const LoopParams &p);
    ~LoopClosure();
    // Feed every LIO pose; returns true if it became a keyframe.
    bool addPose(double time, const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const PointCloud::Ptr &cloud_body);
    const Keyframes &keyframes() const { return kfs_; }
    const std::vector<std::pair<int, int>> &loops() const { return loops_; }
    // Map a current odometry pose into the loop-corrected global frame.
    void toGlobal(const Eigen::Matrix3d &r, const Eigen::Vector3d &t, Eigen::Matrix3d &gr, Eigen::Vector3d &gt) const;

private:
    bool detectLoop(int cur, int &pre, Eigen::Matrix4d &T_pre_cur, double &score);
    void optimize(bool loop_found);

    LoopParams params_;
    Keyframes kfs_;
    std::vector<std::pair<int, int>> loops_;
    std::unique_ptr<gtsam::ISAM2> isam_;
    std::unique_ptr<gtsam::NonlinearFactorGraph> graph_;
    std::unique_ptr<gtsam::Values> init_;
    Eigen::Matrix3d offset_rot_ = Eigen::Matrix3d::Identity(); // global <- local
    Eigen::Vector3d offset_pos_ = Eigen::Vector3d::Zero();
};
} // namespace lio
