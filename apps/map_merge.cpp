// Merge two sessions produced by lio_mapping into one map.
// usage: map_merge <session_a_dir> <session_b_dir> <out_dir>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include "lio/map_merge.h"

using namespace lio;

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "usage: map_merge <session_a> <session_b> <out_dir>\n";
        return 1;
    }
    const Keyframes a = loadSession(argv[1]), b = loadSession(argv[2]);
    std::filesystem::create_directories(argv[3]);
    const std::string out = argv[3];
    MergeResult r = mergeSessions(a, b, MergeParams{});
    if (!r.success)
        return 2;
    Keyframes ka(r.merged.begin(), r.merged.begin() + r.size_a), kb(r.merged.begin() + r.size_a, r.merged.end());
    saveSession(out, r.merged);
    std::vector<PoseStamped> pa, pb;
    for (auto &k : ka) pa.push_back({k.time, k.global_rot, k.global_pos});
    for (auto &k : kb) pb.push_back({k.time, k.global_rot, k.global_pos});
    writeTUM(out + "/session_a.tum", pa);
    writeTUM(out + "/session_b.tum", pb);
    pcl::io::savePCDFileBinaryCompressed(out + "/merged_map.pcd", *buildMap(r.merged, 0.1));
    std::ofstream st(out + "/stats.txt");
    st << "init_fitness " << r.init_fitness << "\ninter_session_loops " << r.inter_loops << "\nT_a_b\n" << r.T_a_b << '\n';
    std::cout << "merged " << a.size() << " + " << b.size() << " keyframes, inter-session loops " << r.inter_loops << '\n';
    return 0;
}
