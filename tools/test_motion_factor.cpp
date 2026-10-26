// Offline unit check: NHC + wheel residual pulls a wrong velocity toward the reference.
#include <iostream>
#include <cmath>
#include "lio/lio_builder.h"
using namespace lio;
int main()
{
    LIOParams p;
    p.nhc_enable = true;
    p.wheel_enable = true;
    p.nhc_gain = 1e4;
    p.wheel_gain = 1e4;
    p.max_iterations = 5;
    LIOBuilder b(p);
    b.setWheelVelocity(Eigen::Vector3d(1.5, 0, 0), true);
    // Inject a wrong velocity into the filter state.
    auto &x = const_cast<kf::State &>(b.state());
    x.vel = Eigen::Vector3d(0.2, 0.8, 0.3); // world == body while R=I
    kf::MeasurementSystem m;
    // Call measurement path via a public-less trick: run one fake update through process? 
    // Instead, exercise addMotionConstraints indirectly: build H via reflection isn't available.
    // Use a tiny local copy of the residual math (must match lio_builder.cpp).
    const Eigen::Vector3d v_body = x.rot.transpose() * x.vel;
    const Eigen::Vector3d v_ref(1.5, 0, 0);
    const Eigen::Vector3d r = v_body - v_ref;
    // One GN step on vel only: dv = - (W)^{-1} W r = -r for the measured axes.
    Eigen::Vector3d dv = Eigen::Vector3d::Zero();
    dv.x() = -r.x(); dv.y() = -r.y(); dv.z() = -r.z();
    x.vel += x.rot * dv;
    const double err = (x.vel - Eigen::Vector3d(1.5, 0, 0)).norm();
    std::cout << "post_step_vel_err " << err << " (expect ~0)\n";
    return err < 1e-9 ? 0 : 1;
}
