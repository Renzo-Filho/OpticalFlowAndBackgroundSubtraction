@echo off
echo ==============================================
echo   Configurando atalho para o Windows...
echo ==============================================
echo.

REM 1. Descobre o caminho absoluto exato de onde esta pasta esta agora
set "PASTA_ATUAL=%~dp0"
set "ALVO=%~dp0rodar_demo.bat"
set "ATALHO=%USERPROFILE%\Desktop\Demo Visao Computacional.lnk"
set "ICONE=%~dp0assets\icons\logo.ico"

REM 2. Usa o PowerShell para gerar um atalho real (.lnk) apontando para o lugar certo
powershell -Command "$wshell = New-Object -ComObject WScript.Shell; $shortcut = $wshell.CreateShortcut('%ATALHO%'); $shortcut.TargetPath = '%ALVO%'; $shortcut.WorkingDirectory = '%PASTA_ATUAL%'; if (Test-Path '%ICONE%') { $shortcut.IconLocation = '%ICONE%' }; $shortcut.Save()"

echo [Sucesso] Atalho criado na sua Area de Trabalho!
echo.
echo NOTA SOBRE O ICONE:
echo O Windows exige arquivos .ico para atalhos. Transforme o seu 'input.png' 
echo em 'icone.ico', coloque nesta pasta e rode este setup novamente para o 
echo icone aparecer.
echo.
pause