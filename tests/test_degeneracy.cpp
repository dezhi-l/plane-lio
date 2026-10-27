// Degeneracy analysis, solution remapping projector, velocity damping and the neighbour filter.
#include <iostream>
#include <random>
#include <pcl/point_types.h>
#include "lio/degeneracy.h"
#include "lio/neighbor_filter.h"

using namespace lio;

namespace
{
int fails = 0;
void check(bool ok, const char *what)
{
    std::cout << (ok ? "[ ok ] " : "[FAIL] ") << what << "\n";
    fails += !ok;
}

// Corridor-like information: strong in y/z, weak along x (translation); rotation well-constrained.
kf::Matrix12d corridorInfo()
{
    kf::Matrix12d H = kf::Matrix12d::Zero();
    H.topLeftCorner<3, 3>().diagonal() << 1e-2, 5e3, 6e3; // x is weak
    H.block<3, 3>(3, 3).diagonal() << 4e3, 5e3, 6e3;
    return H;
}

void testDetection()
{
    DegenParams p;
    p.ratio = 1e-3;
    DegenInfo d = analyzeDegeneracy(corridorInfo(), p);
    check(d.n_t == 1 && d.n_r == 0, "relative ratio flags exactly the weak translation direction");
    check(std::abs(d.dir_t.col(0).x()) > 0.999, "weak direction is the x axis");
    DegenParams q;
    q.abs_min = 1.0;
    check(analyzeDegeneracy(corridorInfo(), q).n_t == 1, "absolute threshold flags the same direction");
    check(!analyzeDegeneracy(corridorInfo(), DegenParams()).any(), "detection off by default");

    kf::Matrix12d P = wellConstrainedProjector(analyzeDegeneracy(corridorInfo(), p));
    check(std::abs(P(0, 0)) < 1e-9 && std::abs(P(1, 1) - 1.0) < 1e-9 && std::abs(P(9, 9) - 1.0) < 1e-9,
          "projector removes x, keeps y and the extrinsic block");
}

void testNeighborFilter()
{
    std::mt19937 g(1);
    std::normal_distribution<float> n(0.f, 0.01f);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    PointCloud::Ptr cloud(new PointCloud);
    for (int i = 0; i < 400; ++i) // a plane z = 2 (+noise) in front of the sensor
    {
        PointType pt;
        pt.x = u(g);
        pt.y = u(g);
        pt.z = 2.f + n(g);
        cloud->push_back(pt);
    }
    NeighborFilterParams p;
    p.enable = true;
    p.near_range = 0.0; // no sub-sampling: one output per occupied cell
    std::vector<PointCov> outs;
    NeighborFilter(p).filter(cloud, outs);
    bool planar = !outs.empty();
    for (const auto &o : outs) planar = planar && std::abs(o.point.z() - 2.0) < 0.05;
    check(planar, "neighbour filter outputs plane points (z ~ 2)");
    std::cout << "  outputs " << outs.size() << " of " << cloud->size() << "\n";
    check(outs.size() < cloud->size() / 3, "neighbour filter compresses the scan");
}
} // namespace

int main()
{
    testDetection();
    testNeighborFilter();
    std::cout << (fails ? "FAILED\n" : "all passed\n");
    return fails ? 1 : 0;
}
