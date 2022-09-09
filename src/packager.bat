@echo off
setlocal
title Python Packager - Thunderbird Neutron Data Converter
echo Packaging... Please wait...
call ../venv/Scripts/activate
call :Timestamp yyyy,mm,dd,hh,mn,ss,ms,tz
set "logfile=packaging_logs/packaging_%yyyy%%mm%%DD%T%hh%%mn%%ss%_%ms%%tz%.log"
echo - Packaging starts - %yyyy%/%mm%/%DD% %hh%:%mn%:%ss% - >%logfile%
echo -- Generating spec file -- >>%logfile%
pyi-makespec --onefile ^
    --add-data="configuration/default_config.yaml;configuration" ^
    --splash="splash.png" ^
    neutron_data_converter.py >>%logfile% 2>&1
echo -- Editing spec file for splash screen -- >>%logfile%
rem Splash screen requires values to be set in specfile,
rem but has no way to set these from command line
python specfile_editor.py >>%logfile%
echo -- Packaging converter -- >>%logfile%
pyinstaller --clean --noconfirm neutron_data_converter.spec >>%logfile% 2>&1
call :Timestamp yyyy,mm,DD,hh,mn,ss,ms,tz
echo - Packaging ends - %yyyy%/%mm%/%DD% %hh%:%mn%:%dd% - >>%logfile%
call deactivate
echo Packaging complete.
echo The packaged converter can be found at %~dp0dist\neutron_data_converter.exe
echo Press any key to finish...
pause>nul
exit /B %ERRORLEVEL%
:Timestamp 
rem Generates timestamp data from local time
rem Return vars: yyyy,mm,DD,hh,mn,ss,ms,tz
set "ts="
for /f "skip=1 delims=" %%A in ('wmic os get localdatetime') do if not defined ts set "ts=%%A"
set "%~1=%ts:~0,4%"
set "%~2=%ts:~4,2%"
set "%~3=%ts:~6,2%"
set "%~4=%ts:~8,2%"
set "%~5=%ts:~10,2%"
set "%~6=%ts:~12,2%"
set "%~7=%ts:~15,3%"
set "tzsign=%ts:~21,1%"
set "offset=%ts:~22,3%"
set /A "tzhh=%offset%/60"
set "tzhh=0%tzhh%"
set "tzhh=%tzhh:~-2%"
set /A "tzmm=%offset% %% 60"
set "tzmm=0%tzmm%"
set "tzmm=%tzmm:~-2%"
set "%~8=%tzsign%%tzhh%%tzmm%"
exit /B 0