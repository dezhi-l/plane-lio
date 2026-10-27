// Offline LIO + loop closure.
// usage: lio_mapping <config.yaml> <dataset_dir> <out_dir> [--residual point_to_plane|gicp]
//                    [--start s] [--end s] [--degen ratio] [--wheel] [--nhc] [--zupt] [--bias-limit] [--bias-rw] [--dynamic] [--no-loop]
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include "dataset.h"
#include "lio/lio_builder.h"
#include "lio/loop_closure.h"

using namespace lio;

namespace
{
template <typename T>
void get(const YAML::Node &n, const char *k, T &v)
{
    if (n && n[k])
        v = n[k].as<T>();
}

struct Job
{
    std::string data_dir, out, residual = "point_to_plane";
    LIOParams lp;
    LoopParams cp;
    double blind = 0.5, start = 0, end = 1e18;
    int stride = 3;
};

// Wheel / NHC / ZUPT / bias-limit keys under `lio:` (all optional).
void loadMotion(const YAML::Node &l, MotionParams &m)
{
    get(l, "wheel_enable", m.wheel_enable);
    get(l, "nhc_enable", m.nhc_enable);
    get(l, "wheel_gain", m.wheel_gain);
    get(l, "nhc_gain", m.nhc_gain);
    get(l, "wheel_scale", m.wheel_scale);
    get(l, "wheel_max_dt", m.wheel_max_dt);
    get(l, "motion_huber", m.huber_k);
    get(l, "zupt_enable", m.zupt_enable);
    get(l, "zupt_speed", m.zupt_speed);
    get(l, "zupt_count", m.zupt_count);
    get(l, "zupt_gyro_thresh", m.zupt_gyro_thresh);
    get(l, "zupt_vel_gain", m.zupt_vel_gain);
    get(l, "zupt_gyro_gain", m.zupt_gyro_gain);
    get(l, "zupt_acc_gain", m.zupt_acc_gain);
    get(l, "bias_limit_enable", m.bias_limit_enable);
    get(l, "bg_max", m.bg_max);
    get(l, "ba_max", m.ba_max);
    get(l, "bias_limit_gain", m.bias_limit_gain);
    if (l && l["wheel_ext_rot"])
    {
        auto r = l["wheel_ext_rot"].as<std::vector<double>>(); // wheel/base -> IMU, row-major
        m.ext_rot = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
    }
    if (l && l["wheel_ext_pos"])
    {
        auto t = l["wheel_ext_pos"].as<std::vector<double>>();
        m.ext_pos = Eigen::Vector3d(t[0], t[1], t[2]);
    }
}

void loadLio(const YAML::Node &l, Job &job)
{
    get(l, "residual", job.residual);
    get(l, "scan_resolution", job.lp.scan_resolution);
    get(l, "map_resolution", job.lp.map_resolution);
    get(l, "max_iterations", job.lp.max_iterations);
    get(l, "p2plane_gain", job.lp.p2plane_gain);
    get(l, "p2plane_robust_delta", job.lp.p2plane_robust_delta);
    get(l, "plane_thresh", job.lp.plane_thresh);
    get(l, "knn", job.lp.knn);
    get(l, "gicp_gain", job.lp.gicp_gain);
    get(l, "degen_ratio", job.lp.degen_ratio);
    loadMotion(l, job.lp.motion);
    if (l && l["dynamic"])
    {
        get(l["dynamic"], "enable", job.lp.dynamic.enable);
        get(l["dynamic"], "appear_thresh", job.lp.dynamic.appear_thresh);
        get(l["dynamic"], "azimuth_bins", job.lp.dynamic.azimuth_bins);
        get(l["dynamic"], "elev_bins", job.lp.dynamic.elev_bins);
    }
    get(l, "extrinsic_est_en", job.lp.extrinsic_est_en);
    get(l, "align_gravity", job.lp.align_gravity);
    int mode = 1;
    get(l, "near_mode", mode);
    job.lp.near_mode = mode == 7 ? VoxelMap::NEARBY_7 : mode == 19 ? VoxelMap::NEARBY_19 : mode == 26 ? VoxelMap::NEARBY_26 : VoxelMap::NEARBY_1;
    get(l, "acc_cov", job.lp.imu_noise.acc_cov);
    get(l, "gyro_cov", job.lp.imu_noise.gyro_cov);
    get(l, "acc_bias_cov", job.lp.imu_noise.acc_bias_cov);
    get(l, "gyro_bias_cov", job.lp.imu_noise.gyro_bias_cov);
    get(l, "bias_random_walk", job.lp.imu_noise.proper_bias_rw);
    if (l && l["imu_ext_rot"])
    {
        auto r = l["imu_ext_rot"].as<std::vector<double>>();
        job.lp.imu_ext_rot = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
    }
    if (l && l["imu_ext_pos"])
    {
        auto t = l["imu_ext_pos"].as<std::vector<double>>();
        job.lp.imu_ext_pos = Eigen::Vector3d(t[0], t[1], t[2]);
    }
}

void loadLoop(const YAML::Node &c, LoopParams &cp)
{
    get(c, "enable", cp.enable);
    get(c, "key_rot_thresh", cp.key_rot_thresh);
    get(c, "key_dist_thresh", cp.key_dist_thresh);
    get(c, "time_thresh", cp.time_thresh);
    get(c, "index_thresh", cp.index_thresh);
    get(c, "search_radius", cp.search_radius);
    get(c, "submap_half_window", cp.submap_half_window);
    get(c, "icp_max_corr", cp.icp_max_corr);
    get(c, "icp_fitness_thresh", cp.icp_fitness_thresh);
}

void applyCli(int argc, char **argv, Job &job)
{
    for (int i = 4; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--residual" && i + 1 < argc)
            job.residual = argv[++i];
        else if (a == "--start" && i + 1 < argc)
            job.start = std::stod(argv[++i]);
        else if (a == "--end" && i + 1 < argc)
            job.end = std::stod(argv[++i]);
        else if (a == "--degen" && i + 1 < argc)
            job.lp.degen_ratio = std::stod(argv[++i]);
        else if (a == "--wheel")
            job.lp.motion.wheel_enable = true;
        else if (a == "--nhc")
            job.lp.motion.nhc_enable = true;
        else if (a == "--zupt")
            job.lp.motion.zupt_enable = true;
        else if (a == "--bias-limit")
            job.lp.motion.bias_limit_enable = true;
        else if (a == "--bias-rw")
            job.lp.imu_noise.proper_bias_rw = true;
        else if (a == "--dynamic")
            job.lp.dynamic.enable = true;
        else if (a == "--no-loop")
            job.cp.enable = false;
    }
}

Job loadJob(int argc, char **argv)
{
    const YAML::Node cfg = YAML::LoadFile(argv[1]);
    Job job;
    job.data_dir = argv[2];
    job.out = argv[3];
    get(cfg["preprocess"], "blind", job.blind);
    get(cfg["preprocess"], "point_stride", job.stride);
    loadLio(cfg["lio"], job);
    loadLoop(cfg["loop_closure"], job.cp);
    applyCli(argc, argv, job);
    job.lp.residual = parseResidual(job.residual);
    return job;
}

void writeOutputs(const std::string &out, const std::string &residual, const std::vector<PoseStamped> &odom,
                  const std::vector<PoseStamped> &global, const LoopClosure &loop, double lio_ms, size_t n_scans)
{
    writeTUM(out + "/odom.tum", odom);
    writeTUM(out + "/global.tum", global);
    saveSession(out, loop.keyframes());
    std::ofstream lf(out + "/loops.txt");
    for (auto &p : loop.loops())
        lf << p.first << ' ' << p.second << '\n';
    PointCloud::Ptr map = buildMap(loop.keyframes(), 0.1);
    if (!map->empty())
        pcl::io::savePCDFileBinaryCompressed(out + "/map.pcd", *map);
    const double ms = n_scans ? lio_ms / n_scans : 0;
    std::ofstream st(out + "/stats.txt");
    st << "residual " << residual << "\nscans " << n_scans << "\nkeyframes " << loop.keyframes().size()
       << "\nloops " << loop.loops().size() << "\nlio_ms_per_scan " << ms << '\n';
    std::cout << "done: " << n_scans << " scans, " << loop.keyframes().size() << " keyframes, "
              << loop.loops().size() << " loops, " << ms << " ms/scan (LIO)\n";
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: lio_mapping <config.yaml> <dataset_dir> <out_dir> [--residual r] [--start s] [--end s] [--degen ratio] [--wheel] [--nhc] [--zupt] [--bias-limit] [--bias-rw] [--dynamic] [--no-loop]\n";
        return 1;
    }
    const Job job = loadJob(argc, argv);
    std::filesystem::create_directories(job.out);
    OfflineDataset data(job.data_dir, job.blind, job.stride);
    LIOBuilder lio(job.lp);
    LoopClosure loop(job.cp);
    std::vector<PoseStamped> odom, global;
    MeasureGroup mg;
    double t_first = -1, lio_ms = 0;
    size_t n_scans = 0;
    while (data.next(mg))
    {
        if (t_first < 0)
            t_first = mg.lidar_time_begin;
        const double rel = mg.lidar_time_begin - t_first;
        if (rel < job.start)
            continue;
        if (rel > job.end)
            break;
        const auto tic = std::chrono::steady_clock::now();
        if (!lio.process(mg))
            continue;
        lio_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
        n_scans++;
        const auto &s = lio.state();
        odom.push_back({mg.lidar_time_end, s.rot, s.pos});
        loop.addPose(mg.lidar_time_end, s.rot, s.pos, lio.cloudBody());
        PoseStamped gp{mg.lidar_time_end};
        loop.toGlobal(s.rot, s.pos, gp.rot, gp.pos);
        global.push_back(gp);
    }
    writeOutputs(job.out, job.residual, odom, global, loop, lio_ms, n_scans);
    std::cout << "degenerate_updates " << lio.degenerate_updates
              << " dynamic_dropped " << lio.dynamic_dropped << " zupt_scans " << lio.zupt_scans << '\n';
    const auto &fs = lio.state();
    std::cout << "final_bg " << fs.bg.transpose() << " final_ba " << fs.ba.transpose() << '\n';
    return 0;
}
