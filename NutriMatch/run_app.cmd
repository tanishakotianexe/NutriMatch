@echo off
REM NutriMatch launcher
REM Picks the first Python that can actually run the app (streamlit + scikit-learn),
REM because the .joblib models in data/ are exported for that environment.
setlocal
cd /d "%~dp0"

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import streamlit, sklearn, joblib, numpy, pandas" >nul 2>nul
if errorlevel 1 (
    echo.
    echo NutriMatch needs a Python with streamlit + scikit-learn installed.
    echo Tried: %PY%
    echo Install them with:  "%PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Starting NutriMatch with:
echo   %PY%
echo Open http://localhost:8501 in your browser. Close this window to stop the site.
echo.
"%PY%" -m streamlit run "%~dp0app\app.py"
pause
