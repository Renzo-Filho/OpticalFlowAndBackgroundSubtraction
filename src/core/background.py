import cv2
import numpy as np
import time
import mediapipe as mp
import os

class BackgroundProcessor:
    # Constantes dos Modos
    MODE_OTSU = 0
    MODE_GRABCUT = 1
    MODE_AI_SELFIE = 2
    MODE_AI_DEEPLAB = 3

    def __init__(self):
        self.mode = self.MODE_OTSU
        self.bg_base = None       
        
        # Pesos do Modo Estático (Otsu e GrabCut)
        self.weight_y = 0.3
        self.weight_cr = 1.2
        self.weight_cb = 1.2

        # Memória exigida pelo GrabCut
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)

        # Memória das IAs
        self.current_ai_mask = None
        self.seg_selfie = None
        self.seg_deeplab = None
        
        self._init_ai_models()

    def _init_ai_models(self):
        """Inicializa os modelos assíncronos do MediaPipe."""
        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Callback universal para atualizar a máscara assíncrona
        def _save_result(result, output_image: mp.Image, timestamp_ms: int):
            if result.category_mask is not None:
                self.current_ai_mask = result.category_mask.numpy_view().copy()

        # Tenta carregar o Selfie Segmenter
        if os.path.exists('selfie_segmenter_landscape.tflite'):
            options_selfie = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path='selfie_segmenter_landscape.tflite'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                output_category_mask=True,
                result_callback=_save_result
            )
            self.seg_selfie = ImageSegmenter.create_from_options(options_selfie)
        else:
            print("Aviso: 'selfie_segmenter_landscape.tflite' nao encontrado.")

        # Tenta carregar o DeepLab
        if os.path.exists('deeplabv3.tflite'):
            options_deeplab = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path='deeplabv3.tflite'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                output_category_mask=True,
                result_callback=_save_result
            )
            self.seg_deeplab = ImageSegmenter.create_from_options(options_deeplab)
        else:
            print("Aviso: 'deeplabv3.tflite' nao encontrado.")

    def capture_static_model(self, cap, num_frames=100):
        print("Capturando background... Por favor, saia da frente.")
        acc = None
        count = 0
        time.sleep(1.0)
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            f_f = frame.astype(np.float32)
            acc = f_f if acc is None else acc + f_f
            count += 1
            
        if acc is not None:
            self.bg_base = (acc / count).astype(np.uint8)
            print("Background Capturado!")
            return True
        return False

    def get_mask(self, frame, flow):
        """O Cérebro roteador. Chama o método de recorte correto."""
        if self.mode == self.MODE_OTSU:
            return self._mask_otsu(frame)
        elif self.mode == self.MODE_GRABCUT:
            return self._mask_grabcut(frame)
        elif self.mode in [self.MODE_AI_SELFIE, self.MODE_AI_DEEPLAB]:
            return self._mask_ai(frame)
            
        # Fallback
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # ==========================================
    # 1. MÉTODO OTSU (Rápido e Clássico)
    # ==========================================
    def _mask_otsu(self, frame):
        """ Uses YCrCb difference and Otsu thresholding. """
        if self.bg_base is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        f_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b_ycc = cv2.cvtColor(self.bg_base, cv2.COLOR_BGR2YCrCb)
        diff = cv2.absdiff(f_ycc, b_ycc)

        score = self.weight_y * diff[..., 0] + self.weight_cr * diff[..., 1] + self.weight_cb * diff[..., 2]
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Limpeza Morfológica Específica para o Otsu
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:2]:
                if cv2.contourArea(cnt) > 500:
                    cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return cv2.GaussianBlur(filled_mask, (11, 11), 0)

    # ==========================================
    # 2. MÉTODO GRABCUT (Otimização de Grafos)
    # ==========================================
    def _mask_grabcut(self, frame):
        if self.bg_base is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        h, w = frame.shape[:2]
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        b_ycc = cv2.cvtColor(cv2.resize(self.bg_base, None, fx=scale, fy=scale), cv2.COLOR_BGR2YCrCb)
        f_ycc = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YCrCb)

        diff = cv2.absdiff(f_ycc, b_ycc)
        score = self.weight_y * diff[..., 0] + self.weight_cr * diff[..., 1] + self.weight_cb * diff[..., 2]
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)

        # Monta o mapa de probabilidades para o GrabCut
        gc_mask = np.full(small_frame.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[score_u8 > 20] = cv2.GC_PR_FGD
        gc_mask[score_u8 > 80] = cv2.GC_FGD
        gc_mask[score_u8 < 5]  = cv2.GC_BGD

        has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
        has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))

        if has_bg and has_fg:
            cv2.grabCut(small_frame, gc_mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            mask_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        else:
            mask_small = np.zeros(small_frame.shape[:2], np.uint8)

        # Retorna para o tamanho original e suaviza as bordas
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(mask, (7, 7), 0)

    # ==========================================
    # 3. MÉTODOS DE INTELIGÊNCIA ARTIFICIAL
    # ==========================================
    def _mask_ai(self, frame):
        h, w = frame.shape[:2]
        
        # O MediaPipe exige RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)

        # Envia a imagem apenas para a IA que estiver ativa no momento
        try:
            if self.mode == self.MODE_AI_SELFIE and self.seg_selfie:
                self.seg_selfie.segment_async(mp_image, ts)
            elif self.mode == self.MODE_AI_DEEPLAB and self.seg_deeplab:
                self.seg_deeplab.segment_async(mp_image, ts)
        except Exception:
            pass # Ignora erros de timestamp repetido caso a câmera engasgue

        # Retorna a máscara processada de forma assíncrona (se já houver alguma)
        if self.current_ai_mask is not None:
            mask_resized = cv2.resize(self.current_ai_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # === CORREÇÃO DA INVERSÃO ===
            if self.mode == self.MODE_AI_SELFIE:
                # O Selfie Segmenter mapeia a Pessoa como 0 e o Fundo como 1 (ou 255)
                # Queremos Pessoa = 255 (Branco) e Fundo = 0 (Preto)
                binary_mask = np.where(mask_resized == 0, 255, 0).astype(np.uint8)
            else:
                # O DeepLab mapeia o Fundo como 0 e a Pessoa/Objetos como > 0 (ex: 15)
                binary_mask = np.where(mask_resized > 0, 255, 0).astype(np.uint8)
                
            return binary_mask
        else:
            return np.zeros((h, w), dtype=np.uint8)