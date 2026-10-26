// Offline LIO + loop closure on a converted dataset.
// usage: lio_mapping <config.yaml> <dataset_dir> <out_dir> [--residual point_to_plane|gicp]
//                    [--start s] [--end s] [--no-loop]
// Outputs in <out_dir>: odom.tum (LIO, every scan), keyframes.tum/.bin (loop-closed session),
//                       global.tum (every scan, loop-corrected), loops.txt, map.pcd, stats.txt
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

template <typename T>
static void get(const YAML::Node &n, const char *k, T &v) { if (n[k]) v = n[k].as<T>(); }

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: lio_mapping <config.yaml> <dataset_dir> <out_dir> [--residual r] [--start s] [--end s] [--no-loop]\n";
        return 1;
    }
    const YAML::Node cfg = YAML::LoadFile(argv[1]);
    const std::string data_dir = argv[2], out = argv[3];
    std::filesystem::create_directories(out);

    LIOParams lp;
    LoopParams cp;
    double blind = 0.5, start = 0, end = 1e18;
    int stride = 3;
    std::string residual = "point_to_plane";
    get(cfg["preprocess"], "blind", blind);
    get(cfg["preprocess"], "point_stride", stride);
    const YAML::Node l = cfg["lio"];
    get(l, "residual", residual);
    get(l, "scan_resolution", lp.scan_resolution);
    get(l, "map_resolution", lp.map_resolution);
    get(l, "max_iterations", lp.max_iterations);
    get(l, "p2plane_gain", lp.p2plane_gain);
    get(l, "p2plane_robust_delta", lp.p2plane_robust_delta);
    get(l, "plane_thresh", lp.plane_thresh);
    get(l, "knn", lp.knn);
    get(l, "gicp_gain", lp.gicp_gain);
    get(l, "extrinsic_est_en", lp.extrinsic_est_en);
    get(l, "align_gravity", lp.align_gravity);
    int mode = 1;
    get(l, "near_mode", mode);
    lp.near_mode = mode == 7 ? VoxelMap::NEARBY_7 : mode == 19 ? VoxelMap::NEARBY_19 : mode == 26 ? VoxelMap::NEARBY_26 : VoxelMap::NEARBY_1;
    get(l, "acc_cov", lp.imu_noise.acc_cov);
    get(l, "gyro_cov", lp.imu_noise.gyro_cov);
    if (l["imu_ext_rot"])
    {
        auto r = l["imu_ext_rot"].as<std::vector<double>>();
        lp.imu_ext_rot = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
    }
    if (l["imu_ext_pos"])
    {
        auto t = l["imu_ext_pos"].as<std::vector<double>>();
        lp.imu_ext_pos = Eigen::Vector3d(t[0], t[1], t[2]);
    }
    const YAML::Node c = cfg["loop_closure"];
    get(c, "enable", cp.enable);
    get(c, "key_rot_thresh", cp.key_rot_thresh);
    get(c, "key_dist_thresh", cp.key_dist_thresh);
    get(c, "time_thresh", cp.time_thresh);
    get(c, "index_thresh", cp.index_thresh);
    get(c, "search_radius", cp.search_radius);
    get(c, "submap_half_window", cp.submap_half_window);
    get(c, "icp_max_corr", cp.icp_max_corr);
    get(c, "icp_fitness_thresh", cp.icp_fitness_thresh);
    for (int i = 4; i < argc; i++)
    {
        std::string a = argv[i];
        if (a == "--residual" && i + 1 < argc) residual = argv[++i];
        else if (a == "--start" && i + 1 < argc) start = std::stod(argv[++i]);
        else if (a == "--end" && i + 1 < argc) end = std::stod(argv[++i]);
        else if (a == "--no-loop") cp.enable = false;
    }
    lp.residual = parseResidual(residual);

    OfflineDataset data(data_dir, blind, stride);
    LIOBuilder lio(lp);
    LoopClosure loop(cp);
    std::vector<PoseStamped> odom, global;
    MeasureGroup mg;
    double t_first = -1, lio_ms = 0;
    size_t n_scans = 0;
    while (data.next(mg))
    {
        if (t_first < 0) t_first = mg.lidar_time_begin;
        const double rel = mg.lidar_time_begin - t_first;
        if (rel < start) continue;
        if (rel > end) break;
        auto tic = std::chrono::steady_clock::now();
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
        if (n_scans % 200 == 0)
            std::cout << "scan " << n_scans << " t=" << rel << "s corr=" << lio.lastNumCorrespondences()
                      << " pos=" << s.pos.transpose() << std::endl;
    }
    writeTUM(out + "/odom.tum", odom);
    writeTUM(out + "/global.tum", global);
    saveSession(out, loop.keyframes());
    std::ofstream lf(out + "/loops.txt");
    for (auto &p : loop.loops()) lf << p.first << ' ' << p.second << '\n';
    PointCloud::Ptr map = buildMap(loop.keyframes(), 0.1);
    if (!map->empty()) pcl::io::savePCDFileBinaryCompressed(out + "/map.pcd", *map);
    std::ofstream st(out + "/stats.txt");
    st << "residual " << residual << "\nscans " << n_scans << "\nkeyframes " << loop.keyframes().size()
       << "\nloops " << loop.loops().size() << "\nlio_ms_per_scan " << (n_scans ? lio_ms / n_scans : 0) << '\n';
    std::cout << "done: " << n_scans << " scans, " << loop.keyframes().size() << " keyframes, "
              << loop.loops().size() << " loops, " << (n_scans ? lio_ms / n_scans : 0) << " ms/scan (LIO)\n";
    return 0;
}
