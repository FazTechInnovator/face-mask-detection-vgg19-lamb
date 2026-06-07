# GitHub Share Guide

Use this when uploading the project so a friend can download and run it without
retraining.

## What To Upload

Upload the project using Git, not by dragging the entire folder into GitHub.
The `.gitignore` file keeps heavy local folders out of the repo.

Included for demo:

```text
src/
config.yaml
requirements.txt
*.bat
README.md
HOW_TO_RUN_WINDOWS.md
FRIEND_HANDOFF.md
outputs/best_model.keras
outputs/class_names.json
outputs/evaluation/
```

Not included:

```text
venv/
raw_datasets/
dataset/
outputs/feature_cache/
duplicate backup models
```

## Upload Commands

Run these from the project folder:

```powershell
git init
git add .
git status
git commit -m "Add face mask detection project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub account and repo.

## Friend Run Commands

Your friend should install 64-bit Python 3.11 first. Then they should download
the repo, open the project folder in PowerShell, and run:

```powershell
setup_windows.bat
run_streamlit.bat
```

For webcam:

```powershell
run_webcam.bat
```

If webcam is slow:

```powershell
run_webcam.bat --predict_every 1.0 --display_width 420
```

If the wrong camera opens:

```powershell
run_webcam.bat --camera 1
```

## Large Model Note

`outputs/best_model.keras` is about 77 MB. GitHub allows files under 100 MB, but
may show a large-file warning. If GitHub refuses the upload, put that model file
in a GitHub Release or Google Drive and tell your friend to download it into:

```text
outputs/best_model.keras
```
