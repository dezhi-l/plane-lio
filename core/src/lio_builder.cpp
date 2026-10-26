#include "lio/lio_builder.h"
#include <stdexcept>
#include <sophus/so3.hpp>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace lio
{
namespace
{
// Fit a plane to the kNN with PCA. Returns false if the neighbourhood is not planar.
bool fitPlane(const std::vector<Eigen::Vector3d> &pts, double thresh, Eigen::Vector3d &n, double &d)
{
    if (pts.size() < 3)
        return false;
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (const auto &p : pts)
        c += p;
    c /= double(pts.size());
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &p : pts)
        cov += (p - c) * (p - c).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    n = es.eigenvectors().col(0);
    d = -n.dot(c);
    for (const auto &p : pts)
        if (std::fabs(n.dot(p) + d) > thresh)
            return false;
    return true;
}

// Cauchy kernel: rho(s) = c^2 log(1 + s/c^2), returns rho'(s) = IRLS weight for squared error s.
inline double cauchyWeight(double s, double c) { return 1.0 / (1.0 + s / (c * c)); }

// Upstream iG-LIO Cauchy (rho, rho', rho'') used by the GICP residual.
inline Eigen::Vector3d cauchyRho(double e, double delta)
{
    const double d2 = delta * delta;
    if (e <= d2)
        return {e, 1.0, 0.0};
    const double se = std::sqrt(e), r1 = delta / se;
    return {2 * se * delta - d2, r1, -0.5 * r1 / e};
}
} // namespace

Residual parseResidual(const std::string &s)
{
    if (s == "point_to_plane" || s == "p2plane")
        return Residual::POINT_TO_PLANE;
    if (s == "gicp")
        return Residual::GICP;
    throw std::runtime_error("unknown residual: " + s);
}

LIOBuilder::LIOBuilder(const LIOParams &p) : params_(p)
{
    kf_ = std::make_shared<kf::IESKF>(p.max_iterations);
    kf_->setMeasurement([this](const kf::State &s, kf::MeasurementSystem &m) { measurement(s, m); });
    imu_ = std::make_unique<IMUProcessor>(kf_, p.imu_noise, p.imu_ext_rot, p.imu_ext_pos, p.align_gravity);
    map_ = std::make_unique<VoxelMap>(p.map_resolution, p.map_capacity, p.grid_capacity, p.near_mode);
    scan_filter_ = std::make_unique<FastVoxelMap>(p.scan_resolution);
}

bool LIOBuilder::process(const MeasureGroup &meas)
{
    if (!imu_->process(meas, cloud_lidar_))
        return false;
    if (!map_initialized_)
    {
        map_->addCloud(cloudWorld());
        key_rot_ = state().rot;
        key_pos_ = state().pos;
        map_initialized_ = true;
        return true;
    }
    scan_filter_->filter(cloud_lidar_, scan_points_);
    kf_->update();
    frame_count_++;
    const Eigen::Matrix3d dR = state().rot.transpose() * key_rot_;
    const bool is_key = frame_count_ < 10 || cloud_lidar_->size() < 1000 ||
                        Sophus::SO3d(Eigen::Quaterniond(dR).normalized()).log().norm() > params_.key_rot_thresh ||
                        (state().pos - key_pos_).norm() > params_.key_pos_thresh;
    if (is_key)
    {
        map_->addCloud(cloudWorld());
        key_rot_ = state().rot;
        key_pos_ = state().pos;
    }
    return true;
}

void LIOBuilder::measurement(const kf::State &s, kf::MeasurementSystem &m)
{
    if (params_.residual == Residual::GICP)
        gicp(s, m);
    else
        pointToPlane(s, m);
}

void LIOBuilder::addJacobianExtrinsic(const kf::State &s, const Eigen::Vector3d &p_lidar,
                                      Eigen::Matrix<double, -1, 12> &J) const
{
    // d(p_world)/d(rot_ext), d(p_world)/d(pos_ext); J currently holds n^T-weighting in cols 0..2.
    const Eigen::MatrixXd A = J.leftCols<3>();
    J.block(0, 6, J.rows(), 3) = -A * s.rot * s.rot_ext * Sophus::SO3d::hat(p_lidar);
    J.block(0, 9, J.rows(), 3) = A * s.rot;
}

void LIOBuilder::pointToPlane(const kf::State &s, kf::MeasurementSystem &m)
{
    const int n = static_cast<int>(scan_points_.size());
    // Data association is refreshed during the first iterations, then frozen (like upstream).
    if (m.iter_num < 3)
    {
        planes_.resize(n);
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++)
        {
            Plane &pl = planes_[i];
            pl.valid = false;
            pl.point = scan_points_[i].point;
            const Eigen::Vector3d pw = s.rot * (s.rot_ext * pl.point + s.pos_ext) + s.pos;
            std::vector<Eigen::Vector3d> nn;
            nn.reserve(params_.knn);
            if (!map_->searchKNN(pw, params_.knn, 5.0, nn) || int(nn.size()) < params_.knn)
                continue;
            pl.valid = fitPlane(nn, params_.plane_thresh, pl.normal, pl.d);
        }
    }
    Eigen::Matrix<double, -1, 12> J(1, 12);
    for (int i = 0; i < n; i++)
    {
        const Plane &pl = planes_[i];
        if (!pl.valid)
            continue;
        const Eigen::Vector3d pb = s.rot_ext * pl.point + s.pos_ext;
        const Eigen::Vector3d pw = s.rot * pb + s.pos;
        const double r = pl.normal.dot(pw) + pl.d;
        if (std::fabs(r) > 5.0 * params_.p2plane_robust_delta)
            continue;
        J.setZero();
        J.block<1, 3>(0, 0) = pl.normal.transpose();
        J.block<1, 3>(0, 3) = -pl.normal.transpose() * s.rot * Sophus::SO3d::hat(pb);
        if (params_.extrinsic_est_en)
            addJacobianExtrinsic(s, pl.point, J);
        const double w = params_.p2plane_gain * cauchyWeight(r * r, params_.p2plane_robust_delta);
        m.H += J.transpose() * w * J;
        m.b += J.transpose() * w * r;
    }
}

void LIOBuilder::gicp(const kf::State &s, kf::MeasurementSystem &m)
{
    if (m.iter_num < 3)
    {
        gicp_corr_.clear();
        Eigen::Vector3d mean_b;
        Eigen::Matrix3d cov_b;
        for (const PointCov &pc : scan_points_)
        {
            const Eigen::Vector3d pw = s.rot * (s.rot_ext * pc.point + s.pos_ext) + s.pos;
            for (const Eigen::Vector3d &r : map_->searchRange())
            {
                const Eigen::Vector3d q = pw + r;
                if (map_->getCentroidAndCovariance(q, mean_b, cov_b) && map_->isSameGrid(q, mean_b))
                    gicp_corr_.push_back({pc.point, mean_b, pc.cov, cov_b});
            }
        }
    }
    Eigen::Matrix<double, -1, 12> J(3, 12);
    for (const GICPCorr &c : gicp_corr_)
    {
        const Eigen::Vector3d pb = s.rot_ext * c.mean_a + s.pos_ext;
        const Eigen::Vector3d pw = s.rot * pb + s.pos;
        const Eigen::Matrix3d RR = s.rot * s.rot_ext;
        const Eigen::Matrix3d omega = (c.cov_b + RR * c.cov_a * RR.transpose()).inverse();
        const Eigen::Vector3d e = pw - c.mean_b; // sign flipped vs upstream, together with J, so the normal equations match
        const double chi2 = e.dot(omega * e);
        if (m.iter_num > 2 && chi2 > 7.815)
            continue;
        const Eigen::Vector3d rho = cauchyRho(chi2, 10.0);
        J.setZero();
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J.block<3, 3>(0, 3) = -s.rot * Sophus::SO3d::hat(pb);
        if (params_.extrinsic_est_en)
            addJacobianExtrinsic(s, c.mean_a, J);
        const Eigen::Matrix3d info = params_.gicp_gain * (rho[1] * omega + 2.0 * rho[2] * omega * e * e.transpose() * omega);
        m.H += J.transpose() * info * J;
        m.b += params_.gicp_gain * rho[1] * J.transpose() * omega * e;
    }
}

PointCloud::Ptr LIOBuilder::cloudBody() const
{
    return transformCloud(*cloud_lidar_, state().rot_ext, state().pos_ext);
}

PointCloud::Ptr LIOBuilder::cloudWorld() const
{
    const kf::State &s = state();
    return transformCloud(*cloud_lidar_, s.rot * s.rot_ext, s.rot * s.pos_ext + s.pos);
}

PointCloud::Ptr transformCloud(const PointCloud &in, const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    PointCloud::Ptr out(new PointCloud);
    out->reserve(in.size());
    for (const auto &p : in.points)
    {
        const Eigen::Vector3d q = R * Eigen::Vector3d(p.x, p.y, p.z) + t;
        PointType o = p;
        o.x = q.x();
        o.y = q.y();
        o.z = q.z();
        out->push_back(o);
    }
    return out;
}
} // namespace lio
