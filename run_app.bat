@echo off
rem Double-click to start Parlianote. Needs Python and the packages in requirements.txt.
cd /d "%~dp0"
python -m streamlit run app/app.py
pause
