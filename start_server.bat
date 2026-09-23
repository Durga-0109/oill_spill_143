@echo off
echo Starting OceanGuard AI Backend Server...
echo To create a new database, run: create_database.bat
if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8000"
if not defined RELOAD set "RELOAD=false"
cd /d "%~dp0backend"
python -m uvicorn main:app --host %HOST% --port %PORT% --reload
pause
