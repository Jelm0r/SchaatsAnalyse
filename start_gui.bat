@echo off
REM Starts the GUI under the YOLO venv (Python 3.11 with torch/ultralytics/PySide6).
"%~dp0.venv-yolo\Scripts\python.exe" "%~dp0skate_gui.py" %*
