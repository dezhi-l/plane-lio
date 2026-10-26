// Iterated error-state Kalman filter (Gauss-Newton form, as in upstream iG-LIO_SAM_LC).
// State (23 dof): pos, rot, rot_ext, pos_ext, vel, bg, ba, gravity (S2, 2 dof).
#pragma once
#include <functional>
#include <Eigen/Core>

namespace lio::kf
{
constexpr double GRAVITY = 9.81;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix12d = Eigen::Matrix<double, 12, 12>;
using Matrix23d = Eigen::Matrix<double, 23, 23>;
using Vector23d = Eigen::Matrix<double, 23, 1>;
using Vector24d = Eigen::Matrix<double, 24, 1>;
using Matrix23x12d = Eigen::Matrix<double, 23, 12>;
using Matrix3x2d = Eigen::Matrix<double, 3, 2>;
using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

// Normal equations of the measurement model.
// H/b: first 12 states (pos, rot, rot_ext, pos_ext) from LiDAR.
// H_vel/b_vel: world-frame velocity (states 12..14) from wheel / NHC.
struct MeasurementSystem
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix12d H = Matrix12d::Zero();
    Vector12d b = Vector12d::Zero();
    Eigen::Matrix3d H_vel = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b_vel = Eigen::Vector3d::Zero();
    // Cross terms: d(r_vel)/d(rot) so H_cross accumulates J_rot^T W J_vel etc. via addVel.
    Eigen::Matrix<double, 3, 3> H_rot_vel = Eigen::Matrix3d::Zero(); // rot block cols 3..5 <-> vel
    Eigen::Matrix<double, 3, 1> b_rot = Eigen::Vector3d::Zero();     // extra rot RHS from vel residual
    size_t iter_num = 0;
    bool valid = true;
};

struct State
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rot_ext = Eigen::Matrix3d::Identity();
    Eigen::Vector3d pos_ext = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d(0.0, 0.0, -GRAVITY);

    void initG(const Eigen::Vector3d &dir) { g = dir.normalized() * GRAVITY; }
    void operator+=(const Vector23d &delta);
    void operator+=(const Vector24d &delta);
    Vector23d operator-(const State &other) const;
    Matrix3x2d getBx() const;
    Matrix3x2d getMx() const;
    Matrix3x2d getMx(const Eigen::Vector2d &res) const;
    Matrix2x3d getNx() const;
};

struct Input
{
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
};

using MeasurementFunc = std::function<void(const State &, MeasurementSystem &)>;

class IESKF
{
public:
    explicit IESKF(size_t max_iter = 5) : max_iter_(max_iter) { P_.setIdentity(); }
    State &x() { return x_; }
    const State &x() const { return x_; }
    Matrix23d &P() { return P_; }
    void setMeasurement(MeasurementFunc f) { func_ = std::move(f); }
    void predict(const Input &inp, double dt, const Matrix12d &Q);
    void update();

private:
    size_t max_iter_;
    double eps_ = 1e-3;
    State x_;
    Matrix23d P_;
    MeasurementFunc func_;
};
} // namespace lio::kf
