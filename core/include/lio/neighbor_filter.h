// Plane-aware scan filter for sparse / feature-poor scans (after FastVoxelMap::filter of the reference
// in /workspace/slam/ref/neighbor_filter_ref.md; independently implemented).
//
//  1. Hash the scan into grid cells (default 0.5 m) accumulating sum, sum of outer products and count.
//  2. Cells are visited nearest-first. A cell with >= min_num points gives one output: centroid +
//     plane-shaped covariance (SVD / eigen values replaced by (1, 1, 1e-3)).
//  3. Cells with too few points are merged with their neighbours (+-range cells); when >= neighbor_min
//     occupied cells take part the merged statistics give the output.
//  4. Within `near_range` of the sensor (|x| or |y|) an output must pass a PCA check (spherical and
//     line-like cells are rejected) and is expanded into four in-plane sub-samples (cov * 0.5).
//  5. Output stops at max_outs.
// Used by LIOBuilder only when the regular voxel filter leaves fewer than `trigger` points.
#pragma once
#include <vector>
#include <Eigen/Core>
#include "lio/types.h"
#include "lio/voxel_map.h"

namespace lio
{
struct NeighborFilterParams
{
    bool enable = false;
    size_t trigger = 1000;    // run when the regular filter yields fewer points than this
    double grid = 0.5;        // [m]
    int min_num = 5;          // points a cell needs on its own
    int range = 1;            // neighbour search range in cells
    int neighbor_min = 3;     // occupied cells needed for a merged output
    double near_range = 5.0;  // [m] PCA check + sub-sampling zone
    double sub_offset = 0.25; // [m] spacing of the near-field sub-samples
    bool pca_check = true;
    double plane_ratio = 0.2; // reject (spherical) if lambda0 > plane_ratio * lambda1
    double line_ratio = 0.05; // reject (line-like) if lambda1 < line_ratio * lambda2
    size_t max_outs = 3000;
};

class NeighborFilter
{
public:
    explicit NeighborFilter(const NeighborFilterParams &p) : p_(p) {}
    void filter(const PointCloud::Ptr &cloud, std::vector<PointCov> &outs) const;

private:
    struct Stats
    {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d outer = Eigen::Matrix3d::Zero();
        size_t n = 0;
        Eigen::Vector3d centroid() const { return sum / double(n); }
        void add(const Stats &o)
        {
            sum += o.sum;
            outer += o.outer;
            n += o.n;
        }
    };
    NeighborFilterParams p_;
    void emit(const Stats &s, std::vector<PointCov> &outs) const;
};
} // namespace lio
