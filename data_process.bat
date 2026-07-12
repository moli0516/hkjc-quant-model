@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Scraping data...
python "scraper/data_manager.py"

cd /d "%~dp0"
echo Cleaning data...
echo Start flatten...
python "data/data_cleaning_flatten.py"
echo Finished flatten!

echo Start normalized races...
python "data/data_cleaning_normalized_races.py"
echo Finished normalized races

echo Start normalized horses...
python "data/data_cleaning_normalized_horses.py"
echo Finished normalized horses

echo Finished executing all data process script
pause