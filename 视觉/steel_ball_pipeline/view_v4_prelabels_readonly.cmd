@echo off
cd /d D:\University\NUEDC\steel_ball_pipeline
echo Read-only viewer. This does not accept or change any label.
python scripts\inspect_prelabels.py --dataset D:\University\NUEDC\datasets\steel_ball_v4_incremental --pending-only --scale 2
if errorlevel 1 pause
