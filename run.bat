@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Install (or repair) the portable Python + libraries inside this folder when something is missing
set NEED=0
if not exist "python\python.exe" set NEED=1
if "%NEED%"=="0" "python\python.exe" -c "import faster_whisper, sounddevice, keyboard, pyperclip; from PySide6 import QtWidgets, QtNetwork" >nul 2>nul || set NEED=1
if "%NEED%"=="1" (
  echo Installing portable Python and libraries into this folder ^(first run, needs internet^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (echo. & echo Setup FAILED. & pause & exit /b 1)
)
rem pythonw = no console window. Errors, if any, are written to crash.log
start "" "python\pythonw.exe" voicenote.py
