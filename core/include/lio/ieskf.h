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

// Generic stacked observation r = h(x) - z with Jacobian J = dr/d(error state) (23 columns,
// right perturbation on SO3) and per-row information w = 1/sigma^2.
struct Observation
{
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    Eigen::VectorXd w;
    Observation(int rows = 0) : r(Eigen::VectorXd::Zero(rows)), J(Eigen::MatrixXd::Zero(rows, 23)), w(Eigen::VectorXd::Zero(rows)) {}
};

// Normal equations of the measurement model.
// H/b: first 12 states (pos, rot, rot_ext, pos_ext) from LiDAR.
// H_full/b_full: information of every non-LiDAR observation over the full 23-dof error state
// (wheel / NHC / ZUPT / bias limits), accumulated as J^T W J and J^T W r via add().
struct MeasurementSystem
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix12d H = Matrix12d::Zero();
    Vector12d b = Vector12d::Zero();
    Matrix23d H_full = Matrix23d::Zero();
    Vector23d b_full = Vector23d::Zero();
    size_t iter_num = 0;
    bool valid = true;

    // huber_k > 0: rows with |r|/sigma > huber_k are down-weighted (IRLS Huber).
    void add(const Observation &o, double huber_k = 0.0);
    void clearFull()
    {
        H_full.setZero();
        b_full.setZero();
    }
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
    // Q = diag(gyro, acc, gyro-bias, acc-bias). proper_bias_rw: bias noise is a continuous random walk,
    // var = Q*dt (G = sqrt(dt)*I); otherwise the legacy per-sample form var = Q*dt^2.
    void setProperBiasRandomWalk(bool v) { proper_bias_rw_ = v; }
    void predict(const Input &inp, double dt, const Matrix12d &Q);
    void update();

private:
    size_t max_iter_;
    bool proper_bias_rw_ = false;
    double eps_ = 1e-3;
    State x_;
    Matrix23d P_;
    MeasurementFunc func_;
};
} // namespace lio::kf
