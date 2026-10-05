@echo off
REM Start the GPU worker daemon on Windows.  Edit the token below (must match the client).
cd /d "%~dp0\.."
if "%OFFLOAD_TOKEN%"=="" set OFFLOAD_TOKEN=offload-secret
python server\daemon.py --host 0.0.0.0 --port 5050 %*
pause
