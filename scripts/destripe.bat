:: If you rename this .bat file, do not change it to "pystripe.bat"
:: Windows will get confused and think this script is calling itself
call activate pystripe
pystripe --input "%CD%" --sigma 6.0 --level 8 --wavelet db3
call deactivate
