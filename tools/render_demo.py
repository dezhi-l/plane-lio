#!/usr/bin/env python3
"""Offscreen demo video: (1) LIO + loop-closure mapping, growing top-down map;
(2) multi-session merge: session B before/after alignment into session A.
usage: render_demo.py <run_dir> <sess_a> <sess_b> <merged_dir> <gt.tum> <out.mp4>"""
import struct, subprocess, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def load(d):
    tum = np.loadtxt(f'{d}/keyframes.tum')
    clouds = []
    with open(f'{d}/keyframes.bin', 'rb') as f:
        for row in tum:
            n, = struct.unpack('<I', f.read(4))
            p = np.frombuffer(f.read(16 * n), np.float32).reshape(-1, 4)[:, :3].astype(float)
            R = Rot.from_quat(row[4:8]).as_matrix()
            clouds.append(p @ R.T + row[1:4])
    return tum, clouds

def sub(c, k=4): return c[::k]
run, sa, sb, mg, gt_path, out = sys.argv[1:7]
W, H, FPS = 960, 540, 20
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}',
                       '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '28', '-preset', 'slow', out],
                      stdin=subprocess.PIPE)
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
def emit():
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())

# --- part 1: mapping
tum, clouds = load(run)
odom = np.loadtxt(f'{run}/odom.tum')
allp = np.vstack(clouds); allp = allp[(allp[:, 2] > -0.3) & (allp[:, 2] < 2.0)]
lim = (np.percentile(allp[:, 0], [1, 99]), np.percentile(allp[:, 1], [1, 99]))
frames = 260
for i in range(frames):
    k = max(1, int(len(clouds) * (i + 1) / frames))
    fig.clf(); ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
    pts = np.vstack([sub(c) for c in clouds[:k]]); pts = pts[(pts[:, 2] > -0.3) & (pts[:, 2] < 2.0)]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=pts[:, 2], cmap='viridis', vmin=-0.3, vmax=2.0, linewidths=0)
    t_end = tum[k - 1, 0]; o = odom[odom[:, 0] <= t_end]
    ax.plot(o[:, 1], o[:, 2], 'r-', lw=1.2); ax.plot(o[-1:, 1], o[-1:, 2], 'ro', ms=5)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'LIO (point-to-plane) + loop closure  |  keyframe {k}/{len(clouds)}  t={t_end - odom[0,0]:.0f}s', fontsize=12)
    emit()

# --- part 2: merge
ta, ca = load(sa); tb, cb = load(sb); tm, cm = load(mg)
A = np.vstack([sub(c, 6) for c in ca]); Bb = np.vstack([sub(c, 6) for c in cb])
Bm = np.vstack([sub(c, 6) for c in cm[len(ca):]])
keep = lambda P: (P[:, 2] > -0.3) & (P[:, 2] < 2.0)
mb = keep(Bb) & keep(Bm); A = A[keep(A)]; Bb = Bb[mb]; Bm = Bm[mb]
U = np.vstack([A, Bb, Bm]); LX = np.percentile(U[:, 0], [0.5, 99.5]); LY = np.percentile(U[:, 1], [0.5, 99.5])
def panel(B, traj_b, title, n=40):
    for _ in range(n):
        fig.clf(); ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
        ax.scatter(A[:, 0], A[:, 1], s=0.2, c='tab:blue', alpha=0.5, linewidths=0)
        ax.scatter(B[:, 0], B[:, 1], s=0.2, c='tab:orange', alpha=0.35, linewidths=0)
        ax.set_xlim(*LX); ax.set_ylim(*LY)
        ax.plot(ta[:, 1], ta[:, 2], 'b-', lw=1); ax.plot(traj_b[:, 1], traj_b[:, 2], '-', c='darkorange', lw=1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); fig.suptitle(title, fontsize=12); emit()
panel(Bb, tb, 'Multi-session merge: session A (blue) and B (orange) in their own frames', 50)
# interpolate B points from own frame to merged frame (same point order) for a smooth transition
for i in range(40):
    a = (i + 1) / 40; panel(Bb * (1 - a) + Bm * a, tb[:, :4] * (1 - a) + tm[len(ca):, :4] * a, 'Global alignment + inter-session loop closures', 1)
panel(Bm, tm[len(ca):], 'Merged map (joint pose graph)', 60)
ff.stdin.close(); ff.wait()
