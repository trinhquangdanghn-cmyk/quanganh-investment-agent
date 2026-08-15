@echo off
title QuangAnh Investment Agent
echo Dang khoi dong QuangAnh Investment Agent...
if not exist venv (
    echo Dang tao moi truong ao...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)
streamlit run app.py
pause