#!/usr/bin/env bash
# Reproduce the A/B evaluation: tools/run_eval.sh <dataset_dir> <gt.tum> <runs_dir>
set -e
D=$1; GT=$2; R=$3; B=$(dirname "$0")/../build; C=$(dirname "$0")/../config/mid360.yaml
for r in point_to_plane gicp; do $B/lio_mapping $C $D $R/$r --residual $r | tail -n 1; done
for r in point_to_plane gicp; do for t in odom global; do
  echo "== $r $t APE (SE3 aligned)"; evo_ape tum $GT $R/$r/$t.tum -a --t_max_diff 0.02 | grep -E "rmse|max"
done
  echo "== $r odom RPE 1 m (trans / rot deg)"
  evo_rpe tum $GT $R/$r/odom.tum -a --t_max_diff 0.02 --delta 1 --delta_unit m | grep rmse
  evo_rpe tum $GT $R/$r/odom.tum -a --t_max_diff 0.02 --delta 1 --delta_unit m --pose_relation angle_deg | grep rmse
done
