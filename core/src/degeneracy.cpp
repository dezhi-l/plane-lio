#include "lio/degeneracy.h"
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace lio
{
namespace
{
// Weakest-first eigen decomposition of a 3x3 information block; returns the number of degenerate directions.
int analyzeBlock(const Eigen::Matrix3d &H, const DegenParams &p, Eigen::Vector3d &ev, Eigen::Matrix3d &dir)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H);
    ev = es.eigenvalues();
    dir = es.eigenvectors();
    const double thr = std::max(p.ratio * ev[2], p.abs_min);
    int n = 0;
    for (int i = 0; i < 3; i++)
        n += ev[i] < thr;
    return std::min(n, 3);
}
} // namespace

DegenInfo analyzeDegeneracy(const kf::Matrix12d &H, const DegenParams &p)
{
    DegenInfo d;
    d.n_t = analyzeBlock(H.block<3, 3>(0, 0), p, d.ev_t, d.dir_t);
    d.n_r = analyzeBlock(H.block<3, 3>(3, 3), p, d.ev_r, d.dir_r);
    return d;
}

kf::Matrix12d wellConstrainedProjector(const DegenInfo &d)
{
    kf::Matrix12d P = kf::Matrix12d::Identity();
    P.block<3, 3>(0, 0) = d.dir_t.rightCols(3 - d.n_t) * d.dir_t.rightCols(3 - d.n_t).transpose();
    P.block<3, 3>(3, 3) = d.dir_r.rightCols(3 - d.n_r) * d.dir_r.rightCols(3 - d.n_r).transpose();
    return P;
}

kf::Observation velocityDampingObs(const DegenInfo &d, const kf::State &s, const Eigen::Vector3d &v_pred, double gain)
{
    kf::Observation o(d.n_t);
    for (int i = 0; i < d.n_t; i++)
    {
        const Eigen::Vector3d u = d.dir_t.col(i);
        o.r[i] = u.dot(s.vel - v_pred);
        o.J.block<1, 3>(i, 12) = u.transpose();
        o.w[i] = gain;
    }
    return o;
}
} // namespace lio
