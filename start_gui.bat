@echo off
REM Start de GUI onder de YOLO-venv (Python 3.11 met torch/ultralytics/PySide6).
"%~dp0.venv-yolo\Scripts\python.exe" "%~dp0skate_gui.py" %*
