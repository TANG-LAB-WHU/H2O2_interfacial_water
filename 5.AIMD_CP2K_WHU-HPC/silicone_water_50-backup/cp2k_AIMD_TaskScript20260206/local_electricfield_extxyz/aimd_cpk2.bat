@echo off
REM ============================================
REM CP2K AIMD Workflow Automation Script
REM ============================================
REM This script performs the following:
REM 1. Extracts the last frame from geo_opt trajectory (Python)
REM 2. Updates silicone_water_box_optimized.xyz
REM 3. Runs CP2K AIMD via Docker Compose (current directory mounted)
REM ============================================
REM Environment: Interfacial_H2O (conda)
REM ============================================

echo ============================================
echo CP2K AIMD Workflow Automation
echo ============================================

REM Step 1: Extract last frame from geo_opt trajectory using Python
echo.
echo [Step 1] Extracting last frame from trajectory...
python extract_last_frame.py silicone_water_50_geo_opt-pos-1.xyz silicone_water_box_optimized.xyz

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to extract last frame
    exit /b 1
)

REM Step 2: Run CP2K AIMD via Docker Compose
echo.
echo [Step 2] Starting CP2K AIMD calculation via Docker...
echo.

docker-compose up --abort-on-container-exit

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Docker returned non-zero exit code. Check aimd.log for details.
) else (
    echo.
    echo ============================================
    echo AIMD calculation completed successfully.
    echo Output: aimd.log
    echo ============================================
)

echo.
echo [Done] Results are available in current directory.

REM Step 3: Cleanup Docker containers
echo.
echo [Cleanup] Removing Docker containers...
docker compose -f docker-compose.yml down --remove-orphans 2>nul
echo.
