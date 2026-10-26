#include "lio/keyframe.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

namespace lio
{
void writeTUM(const std::string &file, const std::vector<PoseStamped> &poses)
{
    std::ofstream f(file);
    f << std::fixed << std::setprecision(9);
    for (const auto &p : poses)
    {
        Eigen::Quaterniond q(p.rot);
        f << p.time << ' ' << p.pos.x() << ' ' << p.pos.y() << ' ' << p.pos.z() << ' '
          << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
}

void saveSession(const std::string &dir, const Keyframes &kfs)
{
    std::vector<PoseStamped> poses;
    std::ofstream bin(dir + "/keyframes.bin", std::ios::binary);
    for (const auto &k : kfs)
    {
        poses.push_back({k.time, k.global_rot, k.global_pos});
        const uint32_t n = k.cloud->size();
        bin.write(reinterpret_cast<const char *>(&n), sizeof(n));
        for (const auto &p : k.cloud->points)
        {
            const float v[4] = {p.x, p.y, p.z, p.intensity};
            bin.write(reinterpret_cast<const char *>(v), sizeof(v));
        }
    }
    writeTUM(dir + "/keyframes.tum", poses);
}

Keyframes loadSession(const std::string &dir)
{
    std::ifstream tum(dir + "/keyframes.tum"), bin(dir + "/keyframes.bin", std::ios::binary);
    if (!tum || !bin)
        throw std::runtime_error("cannot open session " + dir);
    Keyframes kfs;
    std::string line;
    while (std::getline(tum, line))
    {
        std::istringstream ss(line);
        Keyframe k;
        double x, y, z, qx, qy, qz, qw;
        ss >> k.time >> x >> y >> z >> qx >> qy >> qz >> qw;
        k.global_rot = k.local_rot = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
        k.global_pos = k.local_pos = Eigen::Vector3d(x, y, z);
        uint32_t n = 0;
        bin.read(reinterpret_cast<char *>(&n), sizeof(n));
        k.cloud.reset(new PointCloud);
        k.cloud->resize(n);
        for (uint32_t i = 0; i < n; i++)
        {
            float v[4];
            bin.read(reinterpret_cast<char *>(v), sizeof(v));
            auto &p = k.cloud->points[i];
            p.x = v[0], p.y = v[1], p.z = v[2], p.intensity = v[3];
        }
        kfs.push_back(k);
    }
    return kfs;
}

PointCloud::Ptr voxelDownsample(const PointCloud::Ptr &in, double leaf)
{
    PointCloud::Ptr out(new PointCloud);
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf, leaf, leaf);
    vg.setInputCloud(in);
    vg.filter(*out);
    return out;
}

PointCloud::Ptr buildMap(const Keyframes &kfs, double leaf, size_t first, size_t last)
{
    PointCloud::Ptr map(new PointCloud);
    last = std::min(last, kfs.size() - 1);
    for (size_t i = first; i <= last && i < kfs.size(); i++)
        *map += *transformCloud(*kfs[i].cloud, kfs[i].global_rot, kfs[i].global_pos);
    return leaf > 0 ? voxelDownsample(map, leaf) : map;
}

bool alignICP(const PointCloud::Ptr &src, const PointCloud::Ptr &tgt, const Eigen::Matrix4f &guess,
              double max_corr, bool score_within_corr, Eigen::Matrix4d &T, double &fitness)
{
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(max_corr);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    PointCloud aligned;
    icp.align(aligned, guess);
    fitness = score_within_corr ? icp.getFitnessScore(max_corr) : icp.getFitnessScore();
    T = icp.getFinalTransformation().cast<double>();
    return icp.hasConverged();
}
} // namespace lio
