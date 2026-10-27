// Analytic Jacobians of the wheel / NHC / ZUPT / bias-limit observations vs central differences
// on the 23-dof error state (State::operator+= retraction, right perturbation on SO3, S2 gravity).
#include <iostream>
#include <random>
#include <sophus/so3.hpp>
#include "lio/motion_obs.h"

using namespace lio;

namespace
{
using Obs = kf::Observation;
using Func = Obs (*)(const kf::State &, const MotionParams &, const MotionInput &);

Eigen::Vector3d randVec(std::mt19937 &g, double s)
{
    std::normal_distribution<double> n(0.0, s);
    return {n(g), n(g), n(g)};
}

kf::State randomState(std::mt19937 &g)
{
    kf::State s;
    s.rot = Sophus::SO3d::exp(randVec(g, 0.8)).matrix();
    s.vel = randVec(g, 1.0);
    s.bg = randVec(g, 0.3);   // beyond bg_max = 0.2 for most axes -> limit rows are active
    s.ba = randVec(g, 1.5);
    s.initG(Eigen::Vector3d(0.1, -0.2, -1.0));
    return s;
}

// max |J_analytic - J_numeric| over all rows/cols.
double maxError(Func f, const kf::State &s, const MotionParams &p, const MotionInput &in, int &rows)
{
    const Obs o = f(s, p, in);
    rows = std::max(rows, int(o.r.size()));
    if (o.r.size() == 0)
        return 0.0; // bias inside its limits: no rows
    double err = 0.0;
    const double h = 1e-6;
    for (int c = 0; c < 23; c++)
    {
        kf::Vector23d d = kf::Vector23d::Zero();
        d[c] = h;
        kf::State sp = s, sm = s;
        sp += d;
        const kf::Vector23d dm = -d;
        sm += dm;
        const Obs op = f(sp, p, in), om = f(sm, p, in);
        if (op.r.size() != o.r.size() || om.r.size() != o.r.size())
            continue; // hinge switched active set (never at these states)
        const Eigen::VectorXd num = (op.r - om.r) / (2 * h);
        err = std::max(err, (num - o.J.col(c)).cwiseAbs().maxCoeff());
    }
    return err;
}

Obs wheel(const kf::State &s, const MotionParams &p, const MotionInput &i) { return wheelNhcObs(s, p, i); }
Obs zupt(const kf::State &s, const MotionParams &p, const MotionInput &i) { return zuptObs(s, p, i); }
Obs limit(const kf::State &s, const MotionParams &p, const MotionInput &) { return biasLimitObs(s, p); }
} // namespace

int main()
{
    std::mt19937 g(7);
    MotionParams p;
    p.wheel_enable = p.nhc_enable = true;
    p.wheel_scale = 1.03;
    p.zupt_acc_gain = 100.0;
    p.ext_rot = Sophus::SO3d::exp(randVec(g, 0.5)).matrix();
    p.ext_pos = Eigen::Vector3d(0.3, -0.1, -0.4);
    MotionInput in;
    in.wheel_vel = Eigen::Vector3d(1.2, 0.0, 0.0);
    in.wheel_valid = true;
    in.gyro = randVec(g, 0.5);
    in.acc = randVec(g, 3.0);
    MotionParams po = p; // omnidirectional base: lateral speed is observed (non-zero vy)
    po.wheel_omni = true;
    MotionInput io = in;
    io.wheel_vel.y() = -0.4;
    double worst = 0.0;
    int rows[3] = {0, 0, 0};
    for (int trial = 0; trial < 50; trial++)
    {
        const kf::State s = randomState(g);
        worst = std::max({worst, maxError(wheel, s, p, in, rows[0]),
                          maxError(wheel, s, po, io, rows[0]), maxError(zupt, s, p, in, rows[1]),
                          maxError(limit, s, p, in, rows[2])});
    }
    std::cout << "rows wheel/nhc=" << rows[0] << " zupt=" << rows[1] << " bias_limit=" << rows[2]
              << "  max |J_analytic - J_numeric| over 50 random states = " << worst << '\n';
    const bool ok = worst < 1e-6;
    std::cout << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
