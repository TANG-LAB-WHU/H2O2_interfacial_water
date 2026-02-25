@echo off
cd /d "%~dp0"
echo Starting CP2K GEO_OPT with electric field ...
docker compose -f docker-compose-cp2k.yml up --abort-on-container-exit
echo.
echo CP2K calculation finished. Check geo_opt.log for results.
pause
