@echo off
setlocal

call venv\Scripts\activate.bat
python src\prepare_dataset.py --download --reset
