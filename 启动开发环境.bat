@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   启动 learning-python 开发环境
echo ============================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [错误] 找不到 .venv\Scripts\activate.bat
    echo        请先在本目录执行:  python -m venv .venv
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo   虚拟环境 : %VIRTUAL_ENV%
echo   当前目录 : %CD%
echo.
echo   常用命令 :
echo     python day5_minimal.py            运行脚本
echo     python -m pip list               查看已装的包
echo     python -m pip install 包名        装包
echo     deactivate                        退出虚拟环境
echo.
echo   关掉这个窗口即可退出。
echo ============================================================
echo.

cmd /k
