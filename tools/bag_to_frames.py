#!/usr/bin/env python3
"""Convert a ROS1/ROS2 bag with LiDAR PointCloud2 + IMU (+ optional wheel /odom)
into the offline format read by apps/lio_mapping:
  imu.txt   : t ax ay az gx gy gz
  scans.bin : { float64 t_begin; uint32 n; n x float32[5] (x y z intensity offset_ms) }
  wheel.txt : t vx vy vz   (body-frame linear velocity from nav_msgs/Odometry, optional)
"""
import argparse, struct
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

DT = {7: np.float32, 8: np.float64, 2: np.uint8, 6: np.uint32, 4: np.uint16}

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('out')
ap.add_argument('--lidar', default='/eve/lidar3d')
ap.add_argument('--imu', default='/eve/livox/imu')
ap.add_argument('--odom', default='', help='nav_msgs/Odometry topic for wheel velocity (e.g. /odom)')
ap.add_argument('--acc-scale', type=float, default=9.81, help='multiply acc (Livox reports g)')
ap.add_argument('--time-field', default='timestamp')
ap.add_argument('--time-unit', type=float, default=1e-9,
                help='seconds per unit of the per-point time field (Livox driver2: ns)')
ap.add_argument('--no-time-field', action='store_true',
                help='use header stamp only (no per-point timestamp); offset_ms = 0')
ap.add_argument('--max-scans', type=int, default=0, help='stop after N lidar frames (0=all)')
ap.add_argument('--skip-scans', type=int, default=0, help='skip first N lidar frames')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
topics = {a.lidar, a.imu}
if a.odom:
    topics.add(a.odom)
n_scan = n_wheel = 0
with AnyReader([Path(a.bag)]) as r, open(out / 'imu.txt', 'w') as fi, open(out / 'scans.bin', 'wb') as fs, \
     open(out / 'wheel.txt', 'w') as fw:
    conns = [c for c in r.connections if c.topic in topics]
    for c, _, raw in r.messages(connections=conns):
        m = r.deserialize(raw, c.msgtype)
        if c.topic == a.imu:
            la, av = m.linear_acceleration, m.angular_velocity
            s = a.acc_scale
            fi.write(f'{stamp(m.header):.9f} {la.x*s:.6f} {la.y*s:.6f} {la.z*s:.6f} {av.x:.6f} {av.y:.6f} {av.z:.6f}\n')
            continue
        if a.odom and c.topic == a.odom:
            # Prefer twist.twist.linear (body frame for typical differential-drive odom).
            tw = m.twist.twist.linear
            t = stamp(m.header) if hasattr(m, 'header') and m.header.stamp.sec else stamp(getattr(m, 'header', m))
            try:
                t = stamp(m.header)
            except Exception:
                t = 0.0
            fw.write(f'{t:.9f} {tw.x:.6f} {tw.y:.6f} {tw.z:.6f}\n')
            n_wheel += 1
            continue
        # LiDAR PointCloud2
        dtype = np.dtype({'names': [f.name for f in m.fields], 'formats': [DT[f.datatype] for f in m.fields],
                          'offsets': [f.offset for f in m.fields], 'itemsize': m.point_step})
        p = np.frombuffer(m.data.tobytes() if hasattr(m.data, 'tobytes') else bytes(m.data), dtype=dtype,
                          count=m.width * m.height)
        ok = np.isfinite(p['x']) & np.isfinite(p['y']) & np.isfinite(p['z'])
        p = p[ok]
        if len(p) == 0:
            continue
        intens = p['intensity'] if 'intensity' in p.dtype.names else np.zeros(len(p), np.float32)
        if a.no_time_field or a.time_field not in p.dtype.names:
            t0 = stamp(m.header)
            arr = np.stack([p['x'], p['y'], p['z'], intens, np.zeros(len(p))], 1).astype(np.float32)
        else:
            t = p[a.time_field].astype(np.float64) * a.time_unit
            # Relative-within-scan (e.g. Ouster `t`) vs absolute timestamp field.
            if float(t.max()) - float(t.min()) < 1.0 and float(t.max()) < 1e9:
                t0 = stamp(m.header)
                off_ms = (t - float(t.min())) * 1e3
            else:
                t0 = float(t.min())
                off_ms = (t - t0) * 1e3
            arr = np.stack([p['x'], p['y'], p['z'], intens, off_ms], 1).astype(np.float32)
            order = np.argsort(arr[:, 4], kind='stable')
            arr = arr[order]
        n_seen = getattr(a, '_seen', 0) + 1; a._seen = n_seen
        if n_seen <= a.skip_scans:
            continue
        fs.write(struct.pack('<dI', t0, len(arr))); fs.write(arr.tobytes())
        n_scan += 1
        if n_scan == 1:
            print('first scan: header', stamp(m.header), 'point t0', t0, 'span ms', float(arr[:, 4].max()))
        if a.max_scans and n_scan >= a.max_scans:
            break
print('scans', n_scan, 'wheel', n_wheel)
