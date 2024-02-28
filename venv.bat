@echo off

if exist "%~dp0venv\" (
  echo Abriendo VENV.
) else (
  echo Creando...
  md "%~dp0venv"
  python -m venv "%~dp0\venv"
  if not exist "%~dp0main.py" echo.>"%~dp0main.py"
  echo Listo
)

set /p var1=Open app? [Y/n]?:

if "%var1%"=="" "%~dp0venv\Scripts\python" main.py
if "%var1%"=="y" "%~dp0venv\Scripts\python" main.py
if "%var1%"=="Y" "%~dp0venv\Scripts\python" main.py

set /p var2=Open VENV? [y/N]?:

if "%var2%"=="y" cmd /k "cd /d %~dp0venv\Scripts & Activate & cd /d %~dp0"
if "%var2%"=="Y" cmd /k "cd /d %~dp0venv\Scripts & Activate & cd /d %~dp0"