import cv2
import numpy as np
import mediapipe as mp
import time

# Atalhos para as classes do MediaPipe
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Variável global para armazenar a máscara gerada assincronamente
current_mask = None

# Função de callback que recebe o resultado do processamento
def save_result(result,
                output_image: mp.Image,
                timestamp_ms: int):
    global current_mask
    if result.category_mask is not None:
        # A CORREÇÃO ESTÁ AQUI: O .copy() garante que não vamos perder 
        # os dados quando a memória do MediaPipe for reciclada.
        current_mask = result.category_mask.numpy_view().copy()

# Configura as opções do segmentador apontando para o Selfie Segmenter
options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='selfie_segmenter_landscape.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    output_category_mask=True,
    result_callback=save_result
)

# Inicializa o segmentador
with ImageSegmenter.create_from_options(options) as segmenter:
    # Abre a webcam padrão (0)
    cap = cv2.VideoCapture(4)
    
    # Define a resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Pressione 'ESC' para sair.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignorando frame vazio da câmera.")
            continue

        # Inverte a imagem lateralmente (efeito espelho)
        frame = cv2.flip(frame, 1)

        # O MediaPipe exige que a imagem esteja em RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Gera um timestamp em milissegundos para o frame atual
        frame_timestamp_ms = int(time.time() * 1000)

        # Envia a imagem para processamento assíncrono
        segmenter.segment_async(mp_image, frame_timestamp_ms)

        # Processa e aplica o efeito se já tivermos uma máscara salva
        if current_mask is not None:
            # Redimensiona a máscara para o tamanho do frame atual
            mask_resized = cv2.resize(current_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Categoria > 0 garante que tudo que não for fundo puro será mantido nítido
            condition = mask_resized > 0
            condition_3d = np.stack((condition,) * 3, axis=-1)

            # Aplica o desfoque gaussiano no frame original (fundo borrado)
            blurred_frame = cv2.GaussianBlur(frame, (85, 85), 0)

            # Combina a pessoa com o fundo borrado
            output_frame = np.where(condition_3d, blurred_frame, frame)
        else:
            output_frame = frame

        # Mostra o resultado
        cv2.imshow('Selfie Segmentation - Tempo Real', output_frame)

        # Encerra se a tecla ESC for pressionada
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()