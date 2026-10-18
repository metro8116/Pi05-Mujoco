@echo off
rem Install the local collection and replay environment with Conda.
rem
rem Create lerobot_pi05 from environment.yml when absent, then install
rem requirements.txt using the configured package mirror. Require Python
rem 3.12, disable user site-packages, and check dependency consistency.
rem Run from Anaconda Prompt on Windows; activate lerobot_pi05 after
rem installation to collect demonstrations or replay the recorded data.
rem Usage: install_local.bat

setlocal
rem Keep user-level Python packages out of this Conda environment.
set "PYTHONNOUSERSITE=1"
cd /d "%~dp0"

where conda >nul 2>&1
if errorlevel 1 (
    echo Open Anaconda Prompt, then run this script again.
    exit /b 1
)

call conda run --no-capture-output -n lerobot_pi05 python --version >nul 2>&1
if errorlevel 1 (
    call conda env create -f environment.yml
    if errorlevel 1 goto failed
)

call conda env config vars set -n lerobot_pi05 PYTHONNOUSERSITE=1
if errorlevel 1 goto failed
call conda run --no-capture-output -n lerobot_pi05 python -s -c "import site, sys; assert sys.version_info[:2] == (3, 12), 'lerobot_pi05 requires Python 3.12'; assert site.ENABLE_USER_SITE is False, 'User site-packages must be disabled'; print('Python:', sys.executable); print('User site enabled:', site.ENABLE_USER_SITE)"
if errorlevel 1 goto failed
call conda run --no-capture-output -n lerobot_pi05 python -s -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto failed
call conda run --no-capture-output -n lerobot_pi05 python -s -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto failed
call conda run --no-capture-output -n lerobot_pi05 python -s -m pip check
if errorlevel 1 goto failed

echo.
echo Installed. Re-activate the environment in your terminal:
echo conda deactivate
echo conda activate lerobot_pi05
echo Collect: python 1.collect_data_pick_place.py
echo Replay:  python 2.replay_data_pick_place.py
exit /b 0

:failed
echo Installation failed. See the error above; the existing environment was not deleted.
exit /b 1
