@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Batch script for Water Box NVT Prerelaxation (AUTOMATED WORKFLOW)
REM =============================================================================
REM Purpose: 
REM   1. Auto-download and convert MACE models
REM   2. Run LAMMPS simulation to relax water box at 298K
REM =============================================================================

REM -----------------------------------------------------------------------------
REM 1. Configuration Section (Edit these to change models)
REM -----------------------------------------------------------------------------
REM [HOW TO CHOOSE]: 
REM 1. MACE_MODEL_TYPE: 
REM    - Use 'mp' (Materials Project) for almost all cases (includes Silicon).
REM    - Use 'off' ONLY for pure organic liquids/polymers without inorganic surfaces.
set "MACE_MODEL_TYPE=mp"

REM 2. MACE_MODEL_SIZE:
REM    - Use 'mh-1' to get Multi-Head capability (Highest Accuracy, SOTA).
REM    - Use 'medium-mpa-0' if you want the classic stable baseline from Alexandria.
set "MACE_MODEL_SIZE=mh-1"

REM 3. MACE_MODEL_HEAD (ONLY for MODEL_SIZE=mh-1 / Ignore for others):
REM    -------------------------------------------------------------------------
REM    - omat_pbe      [RECOMMENDED]: Best general choice! SOTA stability for 
REM                      Si-O-C-H mixed interfaces (e.g. PDMS on Silicon).
REM    - matpes_r2scan [SURFACES]: Higher precision for solid surface defects.
REM    - oc20_usemppbe [ADSORPTION]: Expert at studying how molecules STICK to 
REM                      a solid surface (e.g. PDMS anchor points).
REM    - omol          [ORGANIC]: Focused on organic/biological chemistry.
REM    - spice_wB97M   [SOLVENTS]: Good for liquid-phase/solvent interactions.
REM    - mp_pbe_refit_add [LEGACY]: Standard Materials Project baseline.
REM    -------------------------------------------------------------------------
REM    * NOTE: If using mpa-0 or OFF23, set this to empty ""
set "MACE_MODEL_HEAD=omol"

REM -----------------------------------------------------------------------------
REM Setup Paths
REM -----------------------------------------------------------------------------
set "MODEL_DIR=mace_pretained_models"
set "CONVERT_SCRIPT=convert_mace_to_mliap.py"

REM Determine filenames based on selection (MATCHING download_mace_model.py logic)
if "%MACE_MODEL_TYPE%"=="mp" (
    if "%MACE_MODEL_SIZE%"=="mh-1" (
        set "RAW_MODEL_NAME=mace-mh-1.model"
    ) else if "%MACE_MODEL_SIZE%"=="medium-mpa-0" (
        set "RAW_MODEL_NAME=mace-mpa-0-medium.model"
    ) else (
        set "RAW_MODEL_NAME=mace-!MACE_MODEL_SIZE!.model"
    )
) else if "%MACE_MODEL_TYPE%"=="off" (
    set "RAW_MODEL_NAME=MACE-OFF23_!MACE_MODEL_SIZE!.model"
) else if "%MACE_MODEL_TYPE%"=="anicc" (
    set "RAW_MODEL_NAME=ani500k_large_CC.model"
) else if "%MACE_MODEL_TYPE%"=="omol" (
    set "RAW_MODEL_NAME=MACE-omol-0-extra-large-1024.model"
)

REM Converted PT name logic
if not "!MACE_MODEL_HEAD!"=="" (
    set "PT_MODEL_NAME=!RAW_MODEL_NAME!_!MACE_MODEL_HEAD!.pt"
) else (
    set "PT_MODEL_NAME=!RAW_MODEL_NAME!.pt"
)

set "RAW_MODEL_PATH=!MODEL_DIR!/!RAW_MODEL_NAME!"
set "PT_MODEL_PATH=!MODEL_DIR!/!PT_MODEL_NAME!"

REM -----------------------------------------------------------------------------
REM Logging Setup
REM -----------------------------------------------------------------------------
for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"`) do set "timestamp=%%i"
set "LOG_FILE=run_prerelaxation_!timestamp!.log"

echo ========================================== >> "!LOG_FILE!"
echo Water Box NVT Prerelaxation Started >> "!LOG_FILE!"
echo Model: !MACE_MODEL_TYPE! / !MACE_MODEL_SIZE! >> "!LOG_FILE!"
echo Head:  !MACE_MODEL_HEAD! >> "!LOG_FILE!"
echo Started at: %date% %time% >> "!LOG_FILE!"
echo ========================================== >> "!LOG_FILE!"

REM -----------------------------------------------------------------------------
REM Step 0: Ensure Data File exists
REM -----------------------------------------------------------------------------
if not exist "water_box.lmpdat" (
    echo [ERROR] water_box.lmpdat not found. Please build the water box first. >> "!LOG_FILE!"
    pause
    exit /b 1
) else (
    echo [INFO] water_box.lmpdat found. >> "!LOG_FILE!"
)

REM -----------------------------------------------------------------------------
REM Step 1: Check/Download Model
REM -----------------------------------------------------------------------------
if not exist "!RAW_MODEL_PATH!" (
    echo [INFO] Raw model not found. Triggering automated download... | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"
    docker compose run --rm mace_model_download !MACE_MODEL_TYPE! !MACE_MODEL_SIZE! 2>&1 | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"
    if errorlevel 1 (
        echo [ERROR] Model download failed. Check Internet connection. >> "!LOG_FILE!"
        pause
        exit /b 1
    )
) else (
    echo [INFO] Raw model found at: !RAW_MODEL_PATH! >> "!LOG_FILE!"
)

REM -----------------------------------------------------------------------------
REM Step 2: Check/Convert Model
REM -----------------------------------------------------------------------------
if not exist "!PT_MODEL_PATH!" (
    echo [INFO] MLIAP PT model not found. Triggering conversion... | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"
    
    set "HEAD_ARG="
    if not "!MACE_MODEL_HEAD!"=="" set "HEAD_ARG=--head !MACE_MODEL_HEAD!"
    
    docker compose run --rm mace_model_conversion python3 /workspace/!CONVERT_SCRIPT! /workspace/!RAW_MODEL_PATH! /workspace/!PT_MODEL_PATH! !HEAD_ARG! 2>&1 | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"
    
    if errorlevel 1 (
        echo [ERROR] Model conversion failed. >> "!LOG_FILE!"
        pause
        exit /b 1
    )
) else (
    echo [INFO] Converted MLIAP model found at: !PT_MODEL_PATH! >> "!LOG_FILE!"
)

REM -----------------------------------------------------------------------------
REM Step 3: Run LAMMPS Simulation
REM -----------------------------------------------------------------------------
echo [INFO] Starting LAMMPS simulation with model: !PT_MODEL_PATH! | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"

REM Run using variable injection for the model path
docker compose run --rm lammps_prerelaxation lmp -k on g 1 -sf kk -pk kokkos neigh half newton on -var MODEL_PT_PATH !PT_MODEL_PATH! -in lammps_prerelaxtion.inp 2>&1 | powershell -Command "$input | ForEach-Object { Add-Content -Path '!LOG_FILE!' -Value $_; Write-Host $_ }"

if errorlevel 1 (
    echo [ERROR] LAMMPS simulation failed. >> "!LOG_FILE!"
    pause
    exit /b 1
)

echo. >> "!LOG_FILE!"
echo [SUCCESS] Workflow completed at: %date% %time% >> "!LOG_FILE!"
echo Final density and dimensions should be checked in the summary block below or in !LOG_FILE! >> "!LOG_FILE!"
echo.
echo All tasks finished. See !LOG_FILE! for full history.
pause
