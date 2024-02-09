@echo off

set maindir=%cd%
:: echo %maindir%
%~dp0\.venv\Scripts\activate && jupyter lab --notebook-dir=%~dp0 & deactivate 