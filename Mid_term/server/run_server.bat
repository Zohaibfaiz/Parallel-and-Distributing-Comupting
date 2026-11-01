@echo off
title CSC-334: Remote Worker Server Node (Computer B)
cd /d "%~dp0"
echo ======================================================================
echo   CSC-334: REMOTE WORKER SERVER NODE (COMPUTER B)
echo ======================================================================
echo   [1] Launch Modern Server GUI Dashboard (Recommended)
echo   [2] Launch Command-Line Console Daemon (0.0.0.0:5000)
echo ======================================================================
set /p choice="Select Option [1 or 2] (Default 1): "
if "%choice%"=="2" (
    echo Starting Server CLI Daemon on 0.0.0.0:5000...
    python -u server_daemon.py --host 0.0.0.0 --port 5000
) else (
    echo Starting Modern Server GUI Dashboard...
    python server_gui.py
)
pause
