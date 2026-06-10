@echo off
setlocal

if not exist outputs (
  mkdir outputs
)

if exist outputs\best_model.keras (
  echo Model already exists: outputs\best_model.keras
  exit /b 0
)

echo Downloading trained model...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='https://github.com/FazTechInnovator/face-mask-detection-vgg19-lamb/raw/main/outputs/best_model.keras';" ^
  "$out='outputs/best_model.keras';" ^
  "Invoke-WebRequest -Uri $url -OutFile $out"

if not exist outputs\best_model.keras (
  echo Model download failed.
  exit /b 1
)

echo Model downloaded to outputs\best_model.keras
