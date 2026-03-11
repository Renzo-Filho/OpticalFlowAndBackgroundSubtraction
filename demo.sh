#!/bin/bash
echo "=============================================="
echo "  Iniciando Demo de Visão Computacional..."
echo "=============================================="
echo ""

# Muda para o diretório onde este script está
cd "$(dirname "$0")"

# 1. Verifica/Cria o Ambiente Virtual
if [ ! -d "venv" ]; then
    echo "[1/3] Criando ambiente virtual isolado..."
    python3 -m venv .venv
else
    echo "[1/3] Ambiente virtual encontrado."
fi

# 2. Ativa e instala/atualiza dependências
echo "[2/3] Verificando e instalando dependências (isso pode levar alguns segundos na primeira vez)..."
source venv/bin/activate
pip install -r requirements.txt --quiet

# 3. Roda o aplicativo
echo "[3/3] Abrindo a exposição!"
echo ""
python3 src/main.py