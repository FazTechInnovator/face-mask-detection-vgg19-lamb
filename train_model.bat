@echo off
setlocal

call venv\Scripts\activate.bat
python src\train_cached.py --epochs 60 --class_weight_mode sqrt --reuse_features --occlusion_negatives_per_image 2
