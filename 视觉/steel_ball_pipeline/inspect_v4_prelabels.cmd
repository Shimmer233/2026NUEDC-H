@echo off
cd /d D:\University\NUEDC\steel_ball_pipeline
echo Editable review: drag with the left mouse button to replace the ball box.
echo Enter/Space = accept, N = empty, R = restore auto box, A/D = move, Q = save and quit.
python scripts\review_labels.py --dataset D:\University\NUEDC\datasets\steel_ball_v4_incremental --pending-only --scale 2
pause
