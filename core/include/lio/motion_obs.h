// Wheel / non-holonomic / ZUPT / IMU-bias observations for the IESKF measurement update.
//
// Each function returns a stacked Observation (r = h(x) - z, J = dr/d(23-dof error state), w = 1/sigma^2)
// that is fused with the LiDAR residual inside the same iterated update (MeasurementSystem::add).
// Architecture follows the "one update, several observation types" layout of
// lightning-lm (https://github.com/gaoxiang12/lightning-lm, src/core/lio/eskf.{hpp,cc}); that repo
// declares no license and its wheel / bias observation functions are not public, so the models here
// are derived and implemented independently.
#pragma once
#include <Eigen/Core>
#include "lio/ieskf.h"

namespace lio
{
struct MotionParams
{
    // wheel speed (forward) and non-holonomic constraint (lateral / vertical body velocity = 0)
    bool wheel_enable = false;
    bool nhc_enable = false;
    double wheel_gain = 100.0;  // 1/sigma^2 of the forward speed [(m/s)^-2]
    double nhc_gain = 100.0;    // 1/sigma^2 of lateral / vertical speed
    double wheel_scale = 1.0;   // multiplies the measured wheel speed
    double wheel_max_dt = 0.5;  // [s] wheel sample older than this (vs scan end) is ignored
    Eigen::Matrix3d ext_rot = Eigen::Matrix3d::Identity(); // wheel/base frame -> IMU (vector: v_imu = R v_b)
    Eigen::Vector3d ext_pos = Eigen::Vector3d::Zero();     // wheel/base origin in the IMU frame [m] (lever arm)
    double huber_k = 3.0;       // Huber threshold in sigmas for motion rows (0: off)
    // zero-velocity update when the wheel reports standstill
    bool zupt_enable = false;
    double zupt_speed = 0.05;   // [m/s] |wheel speed| below this counts as standstill
    int zupt_count = 5;         // consecutive standstill wheel samples required
    double zupt_gyro_thresh = 0.02; // [rad/s] max |gyro - bg| in the scan; rejects in-place turns (wheel v = 0)
    double zupt_vel_gain = 1e4; // v_world = 0
    double zupt_gyro_gain = 1e4;// gyro - bg = 0 (makes bg observable)
    double zupt_acc_gain = 0.0; // R(acc - ba) + g = 0 (needs m/s^2 IMU; 0 = off)
    // soft bias bounds (hinge observation, only active outside the limit)
    bool bias_limit_enable = false;
    double bg_max = 0.2;        // [rad/s]
    double ba_max = 1.0;        // [m/s^2]
    double bias_limit_gain = 1e4;
};

// Per-scan measured inputs for the observations above.
struct MotionInput
{
    Eigen::Vector3d wheel_vel = Eigen::Vector3d::Zero(); // body-frame wheel velocity (vx, vy, vz)
    bool wheel_valid = false;
    bool standstill = false;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(); // raw IMU, mean over the scan window / last sample
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
};

// r = R_bi (R^T v + (gyro - bg) x l) - z   (rows: forward wheel, lateral NHC, vertical NHC; only enabled rows)
kf::Observation wheelNhcObs(const kf::State &s, const MotionParams &p, const MotionInput &in);
// rows: v = 0 (3), gyro - bg = 0 (3), optional R(acc - ba) + g = 0 (3)
kf::Observation zuptObs(const kf::State &s, const MotionParams &p, const MotionInput &in);
// rows (only violated axes): bg - clamp(bg, +-bg_max), ba - clamp(ba, +-ba_max)
kf::Observation biasLimitObs(const kf::State &s, const MotionParams &p);
// Adds every enabled observation to the measurement system.
void addMotionObservations(const kf::State &s, const MotionParams &p, const MotionInput &in, kf::MeasurementSystem &m);
} // namespace lio
