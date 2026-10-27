#include "lio/motion_obs.h"
#include <algorithm>
#include <sophus/so3.hpp>

namespace lio
{
namespace
{
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
// Error-state column offsets (see ieskf.h): rot 3, vel 12, bg 15, ba 18, gravity 21.
constexpr int kRot = 3, kVel = 12, kBg = 15, kBa = 18, kGrav = 21;

Mat3 hat(const Vec3 &v) { return Sophus::SO3d::hat(v); }

// Appends `n` rows to o and returns the index of the first new row.
int grow(kf::Observation &o, int n)
{
    const int k = int(o.r.size());
    o.r.conservativeResizeLike(Eigen::VectorXd::Zero(k + n));
    o.w.conservativeResizeLike(Eigen::VectorXd::Zero(k + n));
    o.J.conservativeResizeLike(Eigen::MatrixXd::Zero(k + n, 23));
    return k;
}

// Identity observation of a 3-block: r = x_block - z.
void appendBlock(kf::Observation &o, int col, const Vec3 &x, const Vec3 &z, double w)
{
    const int k = grow(o, 3);
    o.r.segment<3>(k) = x - z;
    o.J.block<3, 3>(k, col).setIdentity();
    o.w.segment<3>(k).setConstant(w);
}
} // namespace

kf::Observation wheelNhcObs(const kf::State &s, const MotionParams &p, const MotionInput &in)
{
    const Mat3 Rbi = p.ext_rot.transpose();
    const Vec3 v_i = s.rot.transpose() * s.vel;
    const Vec3 omega = in.gyro - s.bg;
    const Vec3 v_b = Rbi * (v_i + omega.cross(p.ext_pos));
    const Vec3 z(p.wheel_scale * in.wheel_vel.x(), 0.0, 0.0);
    // Full 3-row model, then keep the enabled rows.
    Eigen::Matrix<double, 3, 23> J = Eigen::Matrix<double, 3, 23>::Zero();
    J.block<3, 3>(0, kRot) = Rbi * hat(v_i);               // right perturbation R*Exp(dtheta)
    J.block<3, 3>(0, kVel) = Rbi * s.rot.transpose();
    J.block<3, 3>(0, kBg) = Rbi * hat(p.ext_pos);          // (w - bg) x l  ->  +hat(l) dbg
    kf::Observation o;
    const bool use[3] = {p.wheel_enable && in.wheel_valid, p.nhc_enable, p.nhc_enable};
    const double gain[3] = {p.wheel_gain, p.nhc_gain, p.nhc_gain};
    for (int i = 0; i < 3; i++)
    {
        if (!use[i])
            continue;
        const int k = grow(o, 1);
        o.r[k] = v_b[i] - z[i];
        o.J.row(k) = J.row(i);
        o.w[k] = gain[i];
    }
    return o;
}

kf::Observation zuptObs(const kf::State &s, const MotionParams &p, const MotionInput &in)
{
    kf::Observation o;
    appendBlock(o, kVel, s.vel, Vec3::Zero(), p.zupt_vel_gain);
    // At rest the gyro reads only its bias: h = bg, z = gyro.
    appendBlock(o, kBg, s.bg, in.gyro, p.zupt_gyro_gain);
    if (p.zupt_acc_gain > 0)
    {
        // At rest the world acceleration is zero: R(acc - ba) + g = 0.
        const int k = grow(o, 3);
        const Vec3 f = in.acc - s.ba;
        o.r.segment<3>(k) = s.rot * f + s.g;
        o.J.block<3, 3>(k, kRot) = -s.rot * hat(f);
        o.J.block<3, 3>(k, kBa) = -s.rot;
        o.J.block<3, 2>(k, kGrav) = s.getMx();
        o.w.segment<3>(k).setConstant(p.zupt_acc_gain);
    }
    return o;
}

kf::Observation biasLimitObs(const kf::State &s, const MotionParams &p)
{
    kf::Observation o;
    for (int blk = 0; blk < 2; blk++)
    {
        const Vec3 &b = blk == 0 ? s.bg : s.ba;
        const double lim = blk == 0 ? p.bg_max : p.ba_max;
        for (int i = 0; i < 3; i++)
        {
            const double excess = b[i] - std::max(-lim, std::min(lim, b[i]));
            if (excess == 0.0)
                continue;
            const int k = grow(o, 1);
            o.r[k] = excess;
            o.J(k, (blk == 0 ? kBg : kBa) + i) = 1.0;
            o.w[k] = p.bias_limit_gain;
        }
    }
    return o;
}

void addMotionObservations(const kf::State &s, const MotionParams &p, const MotionInput &in, kf::MeasurementSystem &m)
{
    if (p.wheel_enable || p.nhc_enable)
        m.add(wheelNhcObs(s, p, in), p.huber_k);
    if (p.zupt_enable && in.standstill)
        m.add(zuptObs(s, p, in));
    if (p.bias_limit_enable)
        m.add(biasLimitObs(s, p));
}
} // namespace lio
