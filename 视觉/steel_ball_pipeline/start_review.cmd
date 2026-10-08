@echo off
cd /d D:\University\NUEDC\steel_ball_pipeline
python scripts\review_labels.py --dataset D:\University\NUEDC\datasets\steel_ball_v4_incremental --pending-only --scale 2
pause
