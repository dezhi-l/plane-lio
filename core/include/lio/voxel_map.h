#pragma once
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
#include "lio/types.h"

namespace lio
{
struct PointCov
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d point;
    Eigen::Matrix3d cov;
    PointCov(const Eigen::Vector3d &p, const Eigen::Matrix3d &c) : point(p), cov(c) {}
};

// Incremental voxel map: spatial hash, per-cell points (for kNN) and a fixed-eigenvalue covariance (for GICP).
class VoxelMap
{
public:
    enum MODE { NEARBY_1, NEARBY_7, NEARBY_19, NEARBY_26 };

    VoxelMap(double resolution, size_t capacity, size_t grid_capacity, MODE mode);

    size_t addCloud(const PointCloud::Ptr &cloud);
    bool searchKNN(const Eigen::Vector3d &point, size_t k, double range, std::vector<Eigen::Vector3d> &results);
    bool getCentroidAndCovariance(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov);
    // Bins at 1 m (resolution_inv_ is intentionally left at 1). The hash uses 1/resolution.
    bool isSameGrid(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const;
    const std::vector<Eigen::Vector3d> &searchRange() const { return delta_; }
    // Plane of the cell nearest (by centroid) to `point` among the search cells: centroid, unit normal and
    // planarity = lambda_min / lambda_mid of the cell covariance (small = planar). False if no valid cell.
    bool searchPlane(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Vector3d &normal, double &planarity);

private:
    struct Grid
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        explicit Grid(size_t cap) : max_num(cap) { points.reserve(2 * cap); }
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        Eigen::Matrix3d conv = Eigen::Matrix3d::Zero(), conv_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3d points_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
        double planarity = 1.0;
        size_t points_num = 0, min_num = 6, max_num = 20;
        bool is_valid = false, is_updated = false;
        std::vector<Eigen::Vector3d> points;
        void addPoint(const Eigen::Vector3d &p, bool insert);
        void updateConv();
    };
    struct Slot
    {
        std::list<size_t>::iterator it;
        std::shared_ptr<Grid> grid;
    };

    MODE mode_;
    double resolution_ = 1, resolution_inv_ = 1, hash_inv_ = 1;
    size_t capacity_ = 5000000, grid_capacity_ = 20;
    std::vector<Eigen::Vector3d> delta_;
    std::vector<size_t> current_idx_;
    std::list<size_t> cache_;
    std::unordered_map<size_t, Slot> storage_;

    size_t hash(const Eigen::Vector3d &p) const;
    void initDelta();
    void touch(Slot &slot);
};

// One-shot voxel downsample. Covariance uses the 3x3x3 neighbourhood of cell centroids (iG-LIO).
class FastVoxelMap
{
public:
    explicit FastVoxelMap(double resolution);
    void filter(const PointCloud::Ptr &cloud, std::vector<PointCov> &out);

private:
    double resolution_, hash_inv_;
    std::vector<Eigen::Vector3d> range_;
};
} // namespace lio
