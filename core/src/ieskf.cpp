#include "lio/ieskf.h"
#include <sophus/so3.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace lio::kf
{
namespace
{
Eigen::Matrix3d Exp(const Eigen::Vector3d &w) { return Sophus::SO3d::exp(w).matrix(); }
Eigen::Vector3d Log(const Eigen::Matrix3d &R) { return Sophus::SO3d(Eigen::Quaterniond(R).normalized()).log(); }
Eigen::Matrix3d Hat(const Eigen::Vector3d &w) { return Sophus::SO3d::hat(w); }
// Jr(w) = Jl(-w). Sophus 1.22 exposes leftJacobian only; it matches the previous hand-rolled Jl.
Eigen::Matrix3d Jr(const Eigen::Vector3d &w) { return Sophus::SO3d::leftJacobian(-w); }

template <class Vec>
void retract(State &s, const Vec &d, const Eigen::Vector3d &dg)
{
    s.pos += d.template segment<3>(0);
    s.rot *= Exp(d.template segment<3>(3));
    s.rot_ext *= Exp(d.template segment<3>(6));
    s.pos_ext += d.template segment<3>(9);
    s.vel += d.template segment<3>(12);
    s.bg += d.template segment<3>(15);
    s.ba += d.template segment<3>(18);
    s.g = Exp(dg) * s.g;
}
} // namespace

void State::operator+=(const Vector23d &d) { retract(*this, d, getBx() * d.segment<2>(21)); }
void State::operator+=(const Vector24d &d) { retract(*this, d, d.segment<3>(21)); }

Vector23d State::operator-(const State &o) const
{
    Vector23d d = Vector23d::Zero();
    d.segment<3>(0) = pos - o.pos;
    d.segment<3>(3) = Log(o.rot.transpose() * rot);
    d.segment<3>(6) = Log(o.rot_ext.transpose() * rot_ext);
    d.segment<3>(9) = pos_ext - o.pos_ext;
    d.segment<3>(12) = vel - o.vel;
    d.segment<3>(15) = bg - o.bg;
    d.segment<3>(18) = ba - o.ba;
    const double v_sin = (Hat(g) * o.g).norm();
    const double v_cos = g.dot(o.g);
    const double theta = std::atan2(v_sin, v_cos);
    Eigen::Vector2d res = Eigen::Vector2d::Zero();
    if (v_sin < 1e-11)
    {
        if (std::fabs(theta) > 1e-11)
            res << M_PI, 0;
    }
    else
        res = theta / v_sin * o.getBx().transpose() * Hat(o.g) * g;
    d.segment<2>(21) = res;
    return d;
}

Matrix3x2d State::getBx() const
{
    Matrix3x2d r;
    r << -g[1], -g[2],
        GRAVITY - g[1] * g[1] / (GRAVITY + g[0]), -g[2] * g[1] / (GRAVITY + g[0]),
        -g[2] * g[1] / (GRAVITY + g[0]), GRAVITY - g[2] * g[2] / (GRAVITY + g[0]);
    return r / GRAVITY;
}

Matrix3x2d State::getMx() const { return -Hat(g) * getBx(); }

Matrix3x2d State::getMx(const Eigen::Vector2d &res) const
{
    const Matrix3x2d bx = getBx();
    const Eigen::Vector3d bu = bx * res;
    return -Exp(bu) * Hat(g) * Jr(bu) * bx;
}

Matrix2x3d State::getNx() const { return 1 / GRAVITY / GRAVITY * getBx().transpose() * Hat(g); }

void IESKF::predict(const Input &inp, double dt, const Matrix12d &Q)
{
    const Eigen::Vector3d w = (inp.gyro - x_.bg) * dt;
    Vector24d delta = Vector24d::Zero();
    delta.segment<3>(0) = x_.vel * dt;
    delta.segment<3>(3) = w;
    delta.segment<3>(12) = (x_.rot * (inp.acc - x_.ba) + x_.g) * dt;

    Matrix23d F = Matrix23d::Identity();
    F.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(3, 3) = Exp(-w);
    F.block<3, 3>(3, 15) = -Jr(w) * dt;
    F.block<3, 3>(12, 3) = -x_.rot * Hat(inp.acc - x_.ba) * dt;
    F.block<3, 3>(12, 18) = -x_.rot * dt;
    F.block<3, 2>(12, 21) = x_.getMx() * dt;
    F.block<2, 2>(21, 21) = x_.getNx() * x_.getMx();

    Matrix23x12d G = Matrix23x12d::Zero();
    G.block<3, 3>(3, 0) = -Jr(w) * dt;
    G.block<3, 3>(12, 3) = -x_.rot * dt;
    G.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;
    G.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;
    x_ += delta;
    P_ = F * P_ * F.transpose() + G * Q * G.transpose();
}

void IESKF::update()
{
    const State x_pred = x_;
    const Matrix23d P_inv = P_.inverse();
    MeasurementSystem meas;
    Matrix23d H;
    Vector23d delta = Vector23d::Zero();
    for (size_t i = 0; i < max_iter_; i++)
    {
        meas.H.setZero();
        meas.b.setZero();
        func_(x_, meas);
        delta = x_ - x_pred;
        Matrix23d J = Matrix23d::Identity();
        J.block<3, 3>(3, 3) = Jr(delta.segment<3>(3));
        J.block<3, 3>(6, 6) = Jr(delta.segment<3>(6));
        J.block<2, 2>(21, 21) = x_.getNx() * x_pred.getMx(delta.segment<2>(21));
        H = J.transpose() * P_inv * J;
        Vector23d b = J.transpose() * P_inv * delta;
        H.block<12, 12>(0, 0) += meas.H;
        b.head<12>() += meas.b;
        delta = -H.ldlt().solve(b);
        x_ += delta;
        meas.iter_num++;
        if (delta.cwiseAbs().maxCoeff() < eps_)
            break;
    }
    Matrix23d L = Matrix23d::Identity();
    L.block<3, 3>(3, 3) = Jr(delta.segment<3>(3));
    L.block<3, 3>(6, 6) = Jr(delta.segment<3>(6));
    L.block<2, 2>(21, 21) = x_.getNx() * x_pred.getMx(delta.segment<2>(21));
    P_ = L * H.inverse() * L.transpose();
}
} // namespace lio::kf
