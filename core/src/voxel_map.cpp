#include "lio/voxel_map.h"
#include <algorithm>
#include <cmath>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

namespace lio
{
namespace
{
constexpr size_t kHashP = 116101, kHashN = 10000000000ULL;

size_t voxelHash(const Eigen::Vector3d &p, double inv)
{
    double q[3];
    for (int i = 0; i < 3; ++i)
    {
        q[i] = p[i] * inv;
        if (q[i] < 0)
            q[i] -= 1.0;
    }
    const size_t x = static_cast<size_t>(q[0]), y = static_cast<size_t>(q[1]), z = static_cast<size_t>(q[2]);
    return (((z * kHashP) % kHashN + y) * kHashP) % kHashN + x;
}

Eigen::Matrix3d fixedCov(const Eigen::Matrix3d &cov)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d val(1, 1, 1e-3);
    return svd.matrixU() * val.asDiagonal() * svd.matrixV().transpose();
}
} // namespace

void VoxelMap::Grid::addPoint(const Eigen::Vector3d &point, bool insert)
{
    is_updated = false;
    points_num++;
    points_sum += point;
    centroid = points_sum / static_cast<double>(points_num);
    conv_sum += point * point.transpose();
    if (points.size() >= max_num * 2 || !insert)
        return;
    points.push_back(point);
}

void VoxelMap::Grid::updateConv()
{
    if (is_updated)
        return;
    is_updated = true;
    if (points_num < min_num)
    {
        is_valid = false;
        return;
    }
    conv = (conv_sum - points_sum * centroid.transpose()) / (static_cast<double>(points_num) - 1.0);
    is_valid = true;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(conv);
    normal = es.eigenvectors().col(0);
    planarity = es.eigenvalues()[0] / std::max(es.eigenvalues()[1], 1e-12);
    conv = fixedCov(conv);
}

size_t VoxelMap::hash(const Eigen::Vector3d &p) const { return voxelHash(p, hash_inv_); }

void VoxelMap::initDelta()
{
    // Same offsets and order as the old NEARBY_1 / 7 / 19 / 26 lists (26 includes the centre: 27 cells).
    static constexpr int8_t off[] = {
        0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,
        1, 1, 0, 1, -1, 0, 1, 0, 1, 1, 0, -1, 0, 1, 1, 0, 1, -1,
        -1, 1, 0, -1, -1, 0, -1, 0, 1, -1, 0, -1, 0, -1, 1, 0, -1, -1,
        1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1};
    const int n = mode_ == NEARBY_1 ? 1 : mode_ == NEARBY_7 ? 7 : mode_ == NEARBY_19 ? 19 : 27;
    delta_.clear();
    delta_.reserve(n);
    for (int i = 0; i < n; ++i)
        delta_.emplace_back(off[3 * i] * resolution_, off[3 * i + 1] * resolution_, off[3 * i + 2] * resolution_);
}

VoxelMap::VoxelMap(double resolution, size_t capacity, size_t grid_capacity, MODE mode)
    : mode_(mode), resolution_(resolution), hash_inv_(1.0 / resolution), capacity_(capacity), grid_capacity_(grid_capacity)
{
    initDelta();
}

bool VoxelMap::isSameGrid(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const
{
    auto bin = [&](double v) { return static_cast<int>(std::floor(v * resolution_inv_)); };
    return bin(a.x()) == bin(b.x()) && bin(a.y()) == bin(b.y()) && bin(a.z()) == bin(b.z());
}

void VoxelMap::touch(Slot &slot)
{
    cache_.splice(cache_.begin(), cache_, slot.it);
    slot.it = cache_.begin();
}

size_t VoxelMap::addCloud(const PointCloud::Ptr &cloud)
{
    size_t error_count = 0;
    if (!cloud || cloud->empty())
        return 0;
    current_idx_.clear();
    for (const auto &point : cloud->points)
    {
        const Eigen::Vector3d v(point.x, point.y, point.z);
        const size_t idx = hash(v);
        current_idx_.push_back(idx);
        auto it = storage_.find(idx);
        if (it == storage_.end())
        {
            auto grid = std::make_shared<Grid>(grid_capacity_);
            grid->addPoint(v, true);
            cache_.push_front(idx);
            storage_.insert({idx, Slot{cache_.begin(), grid}});
            if (storage_.size() >= capacity_)
            {
                storage_.erase(cache_.back());
                cache_.pop_back();
            }
            continue;
        }
        Slot &slot = it->second;
        if (!isSameGrid(slot.grid->centroid, v))
        {
            ++error_count;
            cache_.erase(slot.it);
            auto grid = std::make_shared<Grid>(grid_capacity_);
            grid->addPoint(v, true);
            cache_.push_front(idx);
            slot.it = cache_.begin();
            slot.grid = grid;
            continue;
        }
        const size_t n = slot.grid->points_num;
        if (n < 50)
            slot.grid->addPoint(v, n < grid_capacity_);
        touch(slot);
    }
    for (size_t idx : current_idx_)
        storage_[idx].grid->updateConv();
    return error_count;
}

bool VoxelMap::searchKNN(const Eigen::Vector3d &point, size_t k, double range, std::vector<Eigen::Vector3d> &results)
{
    std::vector<std::pair<Eigen::Vector3d, double>> cand;
    const double r2 = range * range;
    for (const auto &d : delta_)
    {
        const Eigen::Vector3d q = point + d;
        auto it = storage_.find(hash(q));
        if (it == storage_.end() || !isSameGrid(q, it->second.grid->centroid))
            continue;
        for (const auto &p : it->second.grid->points)
        {
            const double dist = (point - p).squaredNorm();
            if (dist < r2)
                cand.emplace_back(p, dist);
        }
    }
    if (cand.empty())
        return false;
    const auto closer = [](const auto &a, const auto &b) { return a.second < b.second; };
    if (cand.size() > k)
    {
        std::nth_element(cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k - 1), cand.end(), closer);
        cand.resize(k);
    }
    std::nth_element(cand.begin(), cand.begin(), cand.end(), closer);
    results.clear();
    for (const auto &c : cand)
        results.push_back(c.first);
    return true;
}

bool VoxelMap::getCentroidAndCovariance(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Matrix3d &cov)
{
    auto it = storage_.find(hash(point));
    if (it == storage_.end() || !it->second.grid->is_valid)
        return false;
    centroid = it->second.grid->centroid;
    cov = it->second.grid->conv;
    return true;
}

bool VoxelMap::searchPlane(const Eigen::Vector3d &point, Eigen::Vector3d &centroid, Eigen::Vector3d &normal, double &planarity)
{
    double best = 1e18;
    for (const auto &d : delta_)
    {
        const Eigen::Vector3d q = point + d;
        auto it = storage_.find(hash(q));
        if (it == storage_.end() || !it->second.grid->is_valid || !isSameGrid(q, it->second.grid->centroid))
            continue;
        const Grid &g = *it->second.grid;
        const double dist = (g.centroid - point).squaredNorm();
        if (dist >= best)
            continue;
        best = dist;
        centroid = g.centroid;
        normal = g.normal;
        planarity = g.planarity;
    }
    return best < 1e18;
}

FastVoxelMap::FastVoxelMap(double resolution) : resolution_(resolution), hash_inv_(1.0 / resolution)
{
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            for (int z = -1; z <= 1; ++z)
                range_.emplace_back(x * resolution_, y * resolution_, z * resolution_);
}

void FastVoxelMap::filter(const PointCloud::Ptr &cloud, std::vector<PointCov> &outs)
{
    outs.clear();
    if (!cloud || cloud->empty())
        return;
    struct Bin
    {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        size_t num = 0;
        void add(const Eigen::Vector3d &p)
        {
            num += 1;
            centroid += (p - centroid) / static_cast<double>(num);
        }
    };
    std::vector<Bin> bins;
    std::unordered_map<size_t, size_t> index;
    bins.reserve(cloud->size());
    for (const auto &p : cloud->points)
    {
        const Eigen::Vector3d v(p.x, p.y, p.z);
        const size_t id = voxelHash(v, hash_inv_);
        auto it = index.find(id);
        if (it == index.end())
        {
            index.emplace(id, bins.size());
            bins.push_back({});
            bins.back().add(v);
        }
        else
            bins[it->second].add(v);
    }
    for (const Bin &g : bins)
    {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        size_t n = 0;
        for (const auto &r : range_)
        {
            auto it = index.find(voxelHash(g.centroid + r, hash_inv_));
            if (it == index.end())
                continue;
            const Eigen::Vector3d &c = bins[it->second].centroid;
            sum += c;
            cov += c * c.transpose();
            ++n;
        }
        if (n >= 6)
        {
            const Eigen::Vector3d mu = sum / static_cast<double>(n);
            cov = fixedCov((cov - sum * mu.transpose()) / (static_cast<double>(n) - 1.0));
        }
        outs.emplace_back(g.centroid, cov);
    }
}
} // namespace lio
