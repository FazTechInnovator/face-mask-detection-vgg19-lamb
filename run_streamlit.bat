@echo off
setlocal

call venv\Scripts\activate.bat
streamlit run src\streamlit_app.py
