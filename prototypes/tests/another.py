import cv2
import numpy as np
import mediapipe as mp

# Inicialização da Câmera
cap = cv2.VideoCapture(4)

# -------------------------------------------------------------------
# OPÇÃO 1: OpenCV Tradicional de Ultra-Baixo Custo (CPU)
# Excelente para velocidade bruta, mas gera máscaras ruidosas
# Pode ser MOG2, KNN ou CNT (para extrema rapidez)
# -------------------------------------------------------------------
# bgs_algo = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
# Para extrema velocidade em sistemas embarcados, utiliza-se o CNT:
bgs_algo = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True)

# -------------------------------------------------------------------
# OPÇÃO 2: MediaPipe Selfie Segmentation (IA Edge / MobileNetV3)
# Excelente balanço entre semântica de IA e execução leve em CPU
# -------------------------------------------------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# model_selection=1 utiliza a variante "Landscape" (144x256), a mais rápida disponível
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # --- Processamento OpenCV ---
    # Aplica a subtração estatística do pixel (Rápido, mas ignora semântica)
    mask_opencv = bgs_algo.apply(frame)
    # Limpeza morfológica básica para reduzir ruído sal e pimenta
    mask_opencv = cv2.morphologyEx(mask_opencv, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # --- Processamento MediaPipe ---
    # Requer conversão de cores BGR para RGB para a rede neural
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(frame_rgb)
    
    # O MediaPipe retorna uma matriz de confiança (float32). 
    # Usamos o limiar de 0.5 para binarizar ou mantemos fracionário para blending suave.
    mask_mediapipe = np.stack((results.segmentation_mask,) * 3, axis=-1)
    condition = mask_mediapipe > 0.5
    
    # Substituição de fundo com MediaPipe
    bg_image = np.zeros(frame.shape, dtype=np.uint8) # Fundo preto / Chroma
    bg_image[:] = (0, 255, 0) # Fundo Verde
    
    # A fusão cria o efeito final da demonstração
    output_mediapipe = np.where(condition, frame, bg_image)
    
    # Exibição
    cv2.imshow('OpenCV CNT (Binario)', mask_opencv)
    cv2.imshow('MediaPipe (Semantico Leve)', output_mediapipe)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()