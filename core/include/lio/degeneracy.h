// Per-scan LiDAR degeneracy detection (eigen analysis of the registration information matrix)
// and the safeguards applied to the degenerate directions.
//
// The LiDAR information H = sum w J^T J over the pose (pos 0..2, rot 3..5) is analysed per 3x3 block:
// eigenvectors with eigenvalue < max(ratio * lambda_max, abs_min) are *degenerate* (solution-remapping /
// X-ICP style): the LiDAR step is projected onto the well-constrained subspace and the degenerate
// directions are left to the IMU prediction (plus wheel / NHC rows when available, see LIOBuilder).
#pragma once
#include <vector>
#include <Eigen/Core>
#include "lio/ieskf.h"

namespace lio
{
struct DegenParams
{
    double ratio = 0.0;            // relative threshold, 0 disables detection
    double abs_min = 0.0;          // absolute threshold on the eigenvalue (same units as H)
    bool freeze_bias = true;       // keep bg / ba at their predicted values on degenerate scans
    double vel_damp_gain = 0.0;    // 1/sigma^2 of "velocity along degenerate direction = predicted" (0 off)
    double wheel_boost = 1.0;      // wheel / NHC information multiplier on degenerate scans
};

struct DegenInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d ev_t = Eigen::Vector3d::Zero(), ev_r = Eigen::Vector3d::Zero(); // ascending
    Eigen::Matrix3d dir_t = Eigen::Matrix3d::Identity(), dir_r = Eigen::Matrix3d::Identity(); // columns, same order
    int n_t = 0, n_r = 0; // number of degenerate (weakest) directions per block
    bool any() const { return n_t + n_r > 0; }
};

// Eigen analysis of the 12x12 LiDAR information (only the pos / rot diagonal blocks are used).
DegenInfo analyzeDegeneracy(const kf::Matrix12d &H, const DegenParams &p);
// Projector onto the well-constrained directions (identity on the extrinsic states).
kf::Matrix12d wellConstrainedProjector(const DegenInfo &d);
// Velocity-damping rows for the degenerate translation directions: u^T (v - v_pred) = 0.
kf::Observation velocityDampingObs(const DegenInfo &d, const kf::State &s, const Eigen::Vector3d &v_pred, double gain);

// One logged scan (final iteration of the update).
struct DegenFrame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double time = 0.0;
    size_t scan = 0, raw = 0, points = 0, planes = 0; // raw scan points, filtered points, valid plane correspondences
    DegenInfo info;
    bool wheel = false;
};
} // namespace lio
