@echo off
cd /d "%~dp0"
echo Starting AI Talk server (stays running)...
echo Open: http://127.0.0.1:8000
echo Press Ctrl+C only when you want to stop.
echo.
".venv\Scripts\python.exe" run_server.py
pause
