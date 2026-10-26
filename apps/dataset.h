// Reader for the ROS-free offline format written by tools/bag_to_frames.py:
//   imu.txt   : t ax ay az gx gy gz   (one line per sample)
//   scans.bin : repeated { double t_begin; uint32 n; n x float32[5] (x y z intensity offset_ms) }
#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include "lio/types.h"

namespace lio
{
class OfflineDataset
{
public:
    OfflineDataset(const std::string &dir, double blind, int point_stride)
        : scans_(dir + "/scans.bin", std::ios::binary), blind2_(blind * blind), stride_(point_stride)
    {
        std::ifstream f(dir + "/imu.txt");
        if (!f || !scans_)
            throw std::runtime_error("cannot open dataset in " + dir);
        IMU m;
        while (f >> m.timestamp >> m.acc.x() >> m.acc.y() >> m.acc.z() >> m.gyro.x() >> m.gyro.y() >> m.gyro.z())
            imus_.push_back(m);
        std::ifstream fw(dir + "/wheel.txt"); // optional: t vx vy vz (body frame, m/s)
        WheelOdom w;
        while (fw >> w.timestamp >> w.vel.x() >> w.vel.y() >> w.vel.z())
            wheels_.push_back(w);
    }

    // Next scan with all IMU samples up to the scan end time.
    bool next(MeasureGroup &mg)
    {
        double t0;
        uint32_t n;
        if (!scans_.read(reinterpret_cast<char *>(&t0), sizeof(t0)) || !scans_.read(reinterpret_cast<char *>(&n), sizeof(n)))
            return false;
        std::vector<float> buf(5 * size_t(n));
        scans_.read(reinterpret_cast<char *>(buf.data()), buf.size() * sizeof(float));
        mg.lidar.reset(new PointCloud);
        mg.lidar->reserve(n / stride_ + 1);
        float max_off = 0;
        for (uint32_t i = 0; i < n; i += stride_)
        {
            const float *v = &buf[5 * size_t(i)];
            if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < blind2_)
                continue;
            PointType p;
            p.x = v[0], p.y = v[1], p.z = v[2], p.intensity = v[3], p.curvature = v[4];
            max_off = std::max(max_off, v[4]);
            mg.lidar->push_back(p);
        }
        mg.lidar_time_begin = t0;
        mg.lidar_time_end = t0 + max_off / 1000.0;
        mg.imus.clear();
        while (imu_idx_ < imus_.size() && imus_[imu_idx_].timestamp <= mg.lidar_time_end)
            mg.imus.push_back(imus_[imu_idx_++]);
        mg.wheels.clear();
        while (wheel_idx_ < wheels_.size() && wheels_[wheel_idx_].timestamp <= mg.lidar_time_end)
            mg.wheels.push_back(wheels_[wheel_idx_++]);
        return true;
    }

private:
    std::ifstream scans_;
    std::deque<IMU> imus_;
    std::deque<WheelOdom> wheels_;
    size_t imu_idx_ = 0;
    size_t wheel_idx_ = 0;
    double blind2_;
    int stride_;
};
} // namespace lio
