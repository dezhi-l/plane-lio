#include "lio/neighbor_filter.h"
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <Eigen/Eigenvalues>

namespace lio
{
namespace
{
using Cell = std::array<int, 3>;

uint64_t cellKey(int x, int y, int z)
{
    constexpr uint64_t m = (1u << 21) - 1;
    return (uint64_t(x + (1 << 20)) & m) | ((uint64_t(y + (1 << 20)) & m) << 21) | ((uint64_t(z + (1 << 20)) & m) << 42);
}
} // namespace

// Emits centroid + plane-shaped covariance of `s` if the cell passes the PCA check; near the sensor
// four in-plane sub-samples are added.
void NeighborFilter::emit(const Stats &s, std::vector<PointCov> &outs) const
{
    if (s.n < 3)
        return;
    const Eigen::Vector3d mu = s.centroid();
    const Eigen::Matrix3d cov = (s.outer - s.sum * mu.transpose()) / double(s.n - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    const Eigen::Vector3d ev = es.eigenvalues().cwiseMax(0.0);
    const bool near = std::fabs(mu.x()) < p_.near_range || std::fabs(mu.y()) < p_.near_range;
    if (near && p_.pca_check && (ev[0] > p_.plane_ratio * ev[1] || ev[1] < p_.line_ratio * ev[2]))
        return; // spherical or line-like: no stable plane
    const Eigen::Matrix3d U = es.eigenvectors();
    const Eigen::Matrix3d reg = U * Eigen::Vector3d(1e-3, 1, 1).asDiagonal() * U.transpose(); // normal = col 0
    outs.emplace_back(mu, reg);
    if (!near)
        return;
    const double a = 0.5 * p_.sub_offset;
    for (int sx : {-1, 1})
        for (int sy : {-1, 1})
            outs.emplace_back(mu + a * (sx * U.col(1) + sy * U.col(2)), 0.5 * reg);
}

void NeighborFilter::filter(const PointCloud::Ptr &cloud, std::vector<PointCov> &outs) const
{
    outs.clear();
    if (!cloud || cloud->empty())
        return;
    const double inv = 1.0 / p_.grid;
    std::unordered_map<uint64_t, Stats> cells;
    std::unordered_map<uint64_t, Cell> index;
    for (const auto &pt : cloud->points)
    {
        const Eigen::Vector3d v(pt.x, pt.y, pt.z);
        const Cell c{int(std::floor(v.x() * inv)), int(std::floor(v.y() * inv)), int(std::floor(v.z() * inv))};
        const uint64_t key = cellKey(c[0], c[1], c[2]);
        Stats &s = cells[key];
        s.sum += v;
        s.outer += v * v.transpose();
        s.n++;
        index[key] = c;
    }
    std::vector<std::pair<double, uint64_t>> order; // nearest cells first so max_outs keeps the near field
    order.reserve(cells.size());
    for (const auto &kv : cells)
        order.emplace_back(kv.second.centroid().norm(), kv.first);
    std::sort(order.begin(), order.end());
    for (const auto &o : order)
    {
        if (outs.size() > p_.max_outs)
            break;
        const Stats &self = cells[o.second];
        if (int(self.n) >= p_.min_num)
        {
            emit(self, outs);
            continue;
        }
        Stats merged;
        int occupied = 0;
        const Cell &c = index[o.second];
        for (int dx = -p_.range; dx <= p_.range; dx++)
            for (int dy = -p_.range; dy <= p_.range; dy++)
                for (int dz = -p_.range; dz <= p_.range; dz++)
                {
                    auto it = cells.find(cellKey(c[0] + dx, c[1] + dy, c[2] + dz));
                    if (it == cells.end())
                        continue;
                    merged.add(it->second);
                    occupied++;
                }
        if (occupied >= p_.neighbor_min)
            emit(merged, outs);
    }
}
} // namespace lio
