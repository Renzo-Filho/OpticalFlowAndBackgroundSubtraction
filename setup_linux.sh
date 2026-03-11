#!/bin/bash

# 1. Descobre o caminho absoluto exato de onde esta pasta está agora
PASTA_ATUAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ARQUIVO_DESKTOP="$PASTA_ATUAL/DemoVisao.desktop"

# 2. Escreve o arquivo .desktop dinamicamente
cat <<EOF > "$ARQUIVO_DESKTOP"
[Desktop Entry]
Version=1.0
Name=Demo Visão Computacional
Comment=Exposição de Optical Flow
Exec=bash "$PASTA_ATUAL/demo.sh"
Path=$PASTA_ATUAL
Icon=$PASTA_ATUAL/assets/icons/logo.png 
Terminal=true
Type=Application
Categories=Application;
EOF

# 3. Dá as permissões de execução para o atalho e para o script principal
chmod +x "$ARQUIVO_DESKTOP"
chmod +x "$PASTA_ATUAL/demo.sh"

# 4. (Opcional) Copia o atalho para a Área de Trabalho do usuário atual
cp "$ARQUIVO_DESKTOP" ~/Desktop/ 2>/dev/null
chmod +x ~/Desktop/DemoVisao.desktop 2>/dev/null

echo "Atalho configurado com sucesso para o caminho: $PASTA_ATUAL"
echo "Você já pode fechar esta janela."
sleep 3