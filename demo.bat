@echo off
echo ==============================================
echo   Iniciando Demo de Visao Computacional...
echo ==============================================
echo.

REM Muda para o diretorio onde este script esta
cd /d "%~dp0"

REM 1. Verifica/Cria o Ambiente Virtual
if not exist "venv" (
    echo [1/3] Criando ambiente virtual isolado...
    python -m venv venv
) else (
    echo [1/3] Ambiente virtual encontrado.
)

REM 2. Ativa e instala/atualiza dependencias
echo [2/3] Verificando e instalando dependencias (isso pode levar alguns segundos na primeira vez)...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

REM 3. Roda o aplicativo
echo [3/3] Abrindo a exposicao!
echo.
python src/main.py

echo.
echo O aplicativo foi encerrado.
pause