@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem  HKJC CLI launcher with terminal-level reload
rem  - 不修改 cli.py
rem  - 結束 CLI 後可按 R 立刻重開（載入最新 .py）
rem ============================================================

cd /d "%~dp0"

rem 可選：自動啟用虛擬環境（依你實際路徑調整／刪除）
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

set "PYTHON_EXE=python"
where %PYTHON_EXE% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 找不到 python，請先安裝或加入 PATH。
    pause
    exit /b 1
)

if not exist "cli.py" (
    echo [ERROR] 目前目錄找不到 cli.py：
    echo   %CD%
    pause
    exit /b 1
)

:MAIN_LOOP
cls
echo ============================================================
echo   HKJC CLI Launcher
echo   工作目錄: %CD%
echo   時間: %DATE% %TIME%
echo ============================================================
echo.
echo   正在啟動 cli.py ...
echo   （在 CLI 內選 0 退出後，可在此重載最新程式碼）
echo.

rem -B：不寫 .pyc，減少「改了檔卻還在跑舊 bytecode」的機率
%PYTHON_EXE% -B cli.py %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================================
echo   cli.py 已結束  (exit code = %EXITCODE%)
echo ============================================================
echo.
echo   [R] 重載並重新啟動 cli.py（載入你剛改的程式碼）
echo   [Q] 離開 launcher
echo.

:ASK
set "CHOICE="
set /p "CHOICE=請選擇 R / Q: "

if /i "%CHOICE%"=="R" (
    echo.
    echo [RELOAD] 重新啟動 cli.py ...
    echo.
    goto MAIN_LOOP
)
if /i "%CHOICE%"=="Q" (
    echo 再見。
    exit /b %EXITCODE%
)
if "%CHOICE%"=="" goto ASK

echo 無效輸入，請輸入 R 或 Q。
goto ASK