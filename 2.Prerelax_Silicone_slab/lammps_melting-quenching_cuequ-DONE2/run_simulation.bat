@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM Batch script for PDMS melt-quench densification
REM =============================================================================
REM Purpose: Run LAMMPS simulation to achieve target density ~1.10 g/cm^3
REM 
REM Workflow:
REM   1. Convert XYZ to LAMMPS data format
REM   2. Convert MACE model to MLIAP format (if needed)
REM   3. Run melt-quench simulation (443K -> 300K)
REM 
REM Chemistry context:
REM   - Curing agent: 2,5-dimethyl-2,5-di(tert-butylperoxy)hexane (1%)
REM   - Curing conditions: 170C (443K), 10 min
REM   - Target density: 1.10 g/cm^3
REM =============================================================================

REM Generate timestamp for log file
for /f "tokens=1-3 delims=/- " %%a in ("%date%") do (
    set "year=%%c"
    set "month=%%a"
    set "day=%%b"
)
if "!month:~1,1!"=="" set "month=0!month!"
if "!day:~1,1!"=="" set "day=0!day!"
for /f "tokens=1-3 delims=:." %%a in ("%time%") do (
    set "hour=%%a"
    set "min=%%b"
    set "sec=%%c"
)
if "!hour:~1,1!"=="" set "hour=0!hour!"
if "!min:~1,1!"=="" set "min=0!min!"
if "!sec:~1,1!"=="" set "sec=0!sec!"
set "timestamp=!day!-!month!-!year!_!hour!-!min!-!sec!"
set "LOG_FILE=run_melt_quench_!timestamp!.log"

REM Start logging
echo ========================================== >> "!LOG_FILE!"
echo PDMS Melt-Quench Densification Simulation >> "!LOG_FILE!"
echo Target density: 1.10 g/cm^3 >> "!LOG_FILE!"
echo Using MACE Machine Learning Force Field >> "!LOG_FILE!"
echo Log file: !LOG_FILE! >> "!LOG_FILE!"
echo Started at: %date% %time% >> "!LOG_FILE!"
echo ========================================== >> "!LOG_FILE!"
echo. >> "!LOG_FILE!"

REM Check docker-compose availability
set DOCKER_COMPOSE_CMD=docker-compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: docker-compose not found >> "!LOG_FILE!"
        echo Please install Docker Desktop >> "!LOG_FILE!"
        pause
        exit /b 1
    ) else (
        set DOCKER_COMPOSE_CMD=docker compose
    )
)
echo Using: !DOCKER_COMPOSE_CMD! >> "!LOG_FILE!"

REM -----------------------------------------------------------------------------
REM Step 1: Convert XYZ to LAMMPS data format
REM -----------------------------------------------------------------------------
echo Step 1: Converting XYZ to LAMMPS data format... >> "!LOG_FILE!"

set "INPUT_XYZ=silicone_slab.xyz"
set "OUTPUT_LMPDAT=silicone_slab.lmpdat"

if exist "!INPUT_XYZ!" (
    python xyz_to_lammps.py "!INPUT_XYZ!" "!OUTPUT_LMPDAT!" >> "!LOG_FILE!" 2>&1
    if errorlevel 1 (
        echo ERROR: Failed to convert XYZ file >> "!LOG_FILE!"
        echo ERROR: Failed to convert XYZ file. Check !LOG_FILE! for details.
        pause
        exit /b 1
    )
    echo XYZ conversion completed successfully. >> "!LOG_FILE!"
) else (
    echo WARNING: !INPUT_XYZ! not found >> "!LOG_FILE!"
    echo Please ensure silicone_slab.xyz exists in current directory >> "!LOG_FILE!"
    echo WARNING: !INPUT_XYZ! not found. Check !LOG_FILE! for details.
    pause
    exit /b 1
)
echo. >> "!LOG_FILE!"

REM -----------------------------------------------------------------------------
REM Step 2: Convert MACE model to MLIAP format
REM -----------------------------------------------------------------------------
echo Step 2: Converting MACE model to MLIAP format... >> "!LOG_FILE!"

set "DEFAULT_MODEL=mace_pretained_models\mace-mpa-0-medium.model"
if exist "%DEFAULT_MODEL%" (
    echo Using default model: %DEFAULT_MODEL% >> "!LOG_FILE!"
    echo Converting model in Docker container... >> "!LOG_FILE!"
    %DOCKER_COMPOSE_CMD% run --rm mace_model_conversion >> "!LOG_FILE!" 2>&1
    if errorlevel 1 (
        echo ERROR: Model conversion failed >> "!LOG_FILE!"
        echo ERROR: Model conversion failed. Check !LOG_FILE! for details.
        pause
        exit /b 1
    )
    echo. >> "!LOG_FILE!"
    echo Model conversion completed successfully! >> "!LOG_FILE!"
    echo Output file: mace_pretained_models\mace-mpa-0-medium.model-mliap_lammps.pt >> "!LOG_FILE!"
) else (
    echo ERROR: Default model not found at: %DEFAULT_MODEL% >> "!LOG_FILE!"
    echo Please download the model first using: >> "!LOG_FILE!"
    echo   python mace_pretained_models\download_mace_model.py mp medium-mpa-0 >> "!LOG_FILE!"
    echo ERROR: MACE model not found. Check !LOG_FILE! for details.
    pause
    exit /b 1
)
echo. >> "!LOG_FILE!"

REM -----------------------------------------------------------------------------
REM Step 3: Run melt-quench simulation
REM -----------------------------------------------------------------------------
echo Step 3: Running melt-quench simulation... >> "!LOG_FILE!"
echo This may take a while... >> "!LOG_FILE!"

!DOCKER_COMPOSE_CMD! run --rm lammps_melt_quench >> "!LOG_FILE!" 2>&1
if errorlevel 1 (
    echo ERROR: Melt-quench simulation failed >> "!LOG_FILE!"
    echo ERROR: Melt-quench simulation failed. Check !LOG_FILE! for details.
    pause
    exit /b 1
)
echo. >> "!LOG_FILE!"
echo Melt-quench simulation completed. >> "!LOG_FILE!"

REM -----------------------------------------------------------------------------
REM Summary
REM -----------------------------------------------------------------------------
echo ========================================== >> "!LOG_FILE!"
echo Simulation completed! >> "!LOG_FILE!"
echo Completed at: %date% %time% >> "!LOG_FILE!"
echo. >> "!LOG_FILE!"
echo Output files: >> "!LOG_FILE!"
echo   - densified_pdms.lmpdat (LAMMPS data) >> "!LOG_FILE!"
echo   - densified_pdms.xyz (XYZ format) >> "!LOG_FILE!"
echo   - densified_pdms.restart (restart file) >> "!LOG_FILE!"
echo   - trajectory.xyz (full trajectory) >> "!LOG_FILE!"
echo   - melt_quench_log.lammps (LAMMPS log) >> "!LOG_FILE!"
echo ========================================== >> "!LOG_FILE!"

echo.
echo All output has been saved to: !LOG_FILE!
pause
