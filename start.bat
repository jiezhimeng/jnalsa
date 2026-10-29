@echo off
setlocal
cd /d "%~dp0"
if defined JIANGNAN_PYTHON goto custom_python
if exist "toolkit\python.exe" (
    "toolkit\python.exe" gui.py --host 127.0.0.1
    goto done
)
if exist ".runtime\Scripts\python.exe" (
    ".runtime\Scripts\python.exe" gui.py --host 127.0.0.1
    goto done
)
echo Python runtime missing. Follow README.md to create .runtime or set JIANGNAN_PYTHON.
exit /b 1
:custom_python
"%JIANGNAN_PYTHON%" gui.py --host 127.0.0.1
:done
set "start_result=%errorlevel%"
if not "%start_result%"=="0" pause
exit /b %start_result%
