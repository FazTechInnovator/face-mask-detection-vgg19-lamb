@echo off
setlocal

call venv\Scripts\activate.bat
python src\webcam_detect.py --auto_camera --display_width 480 --predict_every 0.8 --smooth_window 3 --min_confidence 0.90 --min_margin 0.25 %*
