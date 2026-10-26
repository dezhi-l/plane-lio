#!/usr/bin/env python3
"""Convert a ROS1/ROS2 bag with a Livox PointCloud2 (x,y,z,intensity,...,timestamp) and an IMU topic
into the offline format read by apps/lio_mapping:
  imu.txt   : t ax ay az gx gy gz   (acc in m/s^2, gyro in rad/s)
  scans.bin : repeated { float64 t_begin; uint32 n; n x float32[5] (x, y, z, intensity, offset_ms) }
"""
import argparse, struct
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

DT = {7: np.float32, 8: np.float64, 2: np.uint8, 6: np.uint32, 4: np.uint16}

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('out')
ap.add_argument('--lidar', default='/eve/lidar3d'); ap.add_argument('--imu', default='/eve/livox/imu')
ap.add_argument('--acc-scale', type=float, default=9.81, help='multiply acc (Livox reports g)')
ap.add_argument('--time-field', default='timestamp'); ap.add_argument('--time-unit', type=float, default=1e-9,
                help='seconds per unit of the per-point time field (Livox driver2: ns)')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
stamp = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
with AnyReader([Path(a.bag)]) as r, open(out / 'imu.txt', 'w') as fi, open(out / 'scans.bin', 'wb') as fs:
    conns = [c for c in r.connections if c.topic in (a.lidar, a.imu)]
    n_scan = 0
    for c, _, raw in r.messages(connections=conns):
        m = r.deserialize(raw, c.msgtype)
        if c.topic == a.imu:
            la, av = m.linear_acceleration, m.angular_velocity
            s = a.acc_scale
            fi.write(f'{stamp(m.header):.9f} {la.x*s:.6f} {la.y*s:.6f} {la.z*s:.6f} {av.x:.6f} {av.y:.6f} {av.z:.6f}\n')
            continue
        dtype = np.dtype({'names': [f.name for f in m.fields], 'formats': [DT[f.datatype] for f in m.fields],
                          'offsets': [f.offset for f in m.fields], 'itemsize': m.point_step})
        p = np.frombuffer(m.data.tobytes() if hasattr(m.data, 'tobytes') else bytes(m.data), dtype=dtype,
                          count=m.width * m.height)
        ok = np.isfinite(p['x']) & np.isfinite(p['y']) & np.isfinite(p['z'])
        p = p[ok]
        if len(p) == 0:
            continue
        t = p[a.time_field].astype(np.float64) * a.time_unit
        t0 = float(t.min())
        arr = np.stack([p['x'], p['y'], p['z'], p['intensity'], ((t - t0) * 1e3)], 1).astype(np.float32)
        order = np.argsort(arr[:, 4], kind='stable')
        fs.write(struct.pack('<dI', t0, len(arr))); fs.write(arr[order].tobytes())
        n_scan += 1
        if n_scan == 1:
            print('first scan: header', stamp(m.header), 'point t0', t0, 'span ms', arr[:, 4].max())
print('scans', n_scan)
