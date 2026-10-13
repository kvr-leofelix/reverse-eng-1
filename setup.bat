@echo off
setlocal EnableExtensions
echo.
echo ==============================================
echo   CS213 - Reverse Engineering Lab Setup
echo ==============================================
echo.

if not exist "%~dp0requirements.txt" (
    echo ERROR: requirements.txt was not found.
    echo Keep setup.bat and requirements.txt in the same folder.
    pause
    exit /b 1
)

py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10 was not found.
    echo Please install Python 3.10 and run this script again.
    echo Make sure "Add Python to PATH" is selected during installation.
    pause
    exit /b 1
)

echo Python 3.10 found:
py -3.10 --version
echo.

if exist "%~dp0venv\Scripts\python.exe" (
    echo Existing virtual environment found. Reusing it.
) else (
    echo Creating virtual environment...
    py -3.10 -m venv "%~dp0venv"
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo Upgrading pip...
"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Could not upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo This may take several minutes. Please do not close this window.
echo.

"%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo ==============================================
    echo   INSTALLATION FAILED
    echo ==============================================
    echo.
    echo Please take a screenshot of this window and
    echo contact the CS213 TA team.
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================
echo   SETUP COMPLETED SUCCESSFULLY
echo ==============================================
echo.
echo To activate later:
echo     venv\Scripts\activate
echo.
echo To start Jupyter:
echo     venv\Scripts\activate
echo     jupyter notebook
echo.
pause
