@echo off
setlocal

if not exist venv (
  python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
call download_model.bat

echo.
echo Setup complete. Activate with:
echo venv\Scripts\activate.bat
