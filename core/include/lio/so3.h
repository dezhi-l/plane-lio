// Minimal SO(3) helpers (replaces the Sophus dependency of upstream).
#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lio::so3
{
inline Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

inline Eigen::Matrix3d exp(const Eigen::Vector3d &w)
{
    const double th = w.norm();
    if (th < 1e-10)
        return Eigen::Matrix3d::Identity() + hat(w);
    return Eigen::AngleAxisd(th, w / th).toRotationMatrix();
}

inline Eigen::Vector3d log(const Eigen::Matrix3d &R)
{
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

// Left Jacobian of SO(3); right Jacobian Jr(w) = Jl(w)^T = Jl(-w).
inline Eigen::Matrix3d leftJacobian(const Eigen::Vector3d &w)
{
    const double th = w.norm();
    const Eigen::Matrix3d W = hat(w);
    if (th < 1e-6)
        return Eigen::Matrix3d::Identity() + 0.5 * W;
    const double th2 = th * th;
    return Eigen::Matrix3d::Identity() + (1 - std::cos(th)) / th2 * W + (th - std::sin(th)) / (th2 * th) * W * W;
}

inline Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &w) { return leftJacobian(w).transpose(); }
} // namespace lio::so3
