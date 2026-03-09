import cv2
import numpy as np
import time
import mediapipe as mp
import os
from scipy import ndimage  # <--- IMPORTAÇÃO ADICIONADA PARA O FILL HOLES

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

        def _save_result(result, output_image: mp.Image, timestamp_ms: int):
            if result.category_mask is not None:
                self.current_ai_mask = result.category_mask.numpy_view().copy()

        if os.path.exists('selfie_segmenter_landscape.tflite'):
            options_selfie = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path='selfie_segmenter_landscape.tflite'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                output_category_mask=True,
                result_callback=_save_result
            )
            self.seg_selfie = ImageSegmenter.create_from_options(options_selfie)

        if os.path.exists('deeplabv3.tflite'):
            options_deeplab = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path='deeplabv3.tflite'),
                running_mode=VisionRunningMode.LIVE_STREAM,
                output_category_mask=True,
                result_callback=_save_result
            )
            self.seg_deeplab = ImageSegmenter.create_from_options(options_deeplab)

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
            
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # ==========================================
    # O SEU PÓS-PROCESSAMENTO ROBUSTO
    # ==========================================
    def _post_process(self, mask, min_area_threshold=1000):
        """
        Advanced morphological reconstruction to fix 'missing limbs'.
        Strategy: Bridge Gaps -> Filter by Bottom Edge & Min Area -> Fill Holes.
        """
        h, w = mask.shape[:2]

        # 1. Morphological Closing (The "Bridge")
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)

        # 2. Encontrar componentes conectadas e suas estatísticas
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 3. Identificar IDs: Toca a borda inferior, mas NÃO toca a borda superior
        bottom_labels = set(labels[h - 1, :]) - {0}
        top_labels = set(labels[0, :]) - {0}
        
        target_labels = bottom_labels - top_labels 

        # 4. Filtrar os componentes pela área mínima
        valid_labels = [
            label for label in target_labels 
            if stats[label, cv2.CC_STAT_AREA] >= min_area_threshold
        ]
        
        # 5. Construir a máscara base filtrada
        filtered_mask = np.zeros_like(mask)
        if valid_labels:
            filtered_mask[np.isin(labels, valid_labels)] = 255
            
        # 6. Tapar os buracos internos (Fill Holes)
        filtered_mask = ndimage.binary_fill_holes(filtered_mask).astype(np.uint8) * 255

        # 7. Aplicar a dilatação
        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(filtered_mask, kernel_dilation, iterations=1)
                
        # 8. Anti-aliasing para garantir bordas suaves nos efeitos (Opcional, mas recomendado)
        return cv2.GaussianBlur(dilated_mask, (11, 11), 0)

    # ==========================================
    # 1. MÉTODO OTSU (Rápido e Clássico)
    # ==========================================
    def _mask_otsu(self, frame):
        if self.bg_base is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        f_ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        b_ycc = cv2.cvtColor(self.bg_base, cv2.COLOR_BGR2YCrCb)
        diff = cv2.absdiff(f_ycc, b_ycc)

        score = self.weight_y * diff[..., 0] + self.weight_cr * diff[..., 1] + self.weight_cb * diff[..., 2]
        score_u8 = np.clip(score, 0, 255).astype(np.uint8)
        
        # Gera a máscara bruta
        _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplica a sua função de pós-processamento
        return self._post_process(mask)

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

        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.GaussianBlur(mask, (7, 7), 0)

    # ==========================================
    # 3. MÉTODOS DE INTELIGÊNCIA ARTIFICIAL
    # ==========================================
    def _mask_ai(self, frame):
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)

        try:
            if self.mode == self.MODE_AI_SELFIE and self.seg_selfie:
                self.seg_selfie.segment_async(mp_image, ts)
            elif self.mode == self.MODE_AI_DEEPLAB and self.seg_deeplab:
                self.seg_deeplab.segment_async(mp_image, ts)
        except Exception:
            pass

        if self.current_ai_mask is not None:
            mask_resized = cv2.resize(self.current_ai_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            if self.mode == self.MODE_AI_SELFIE:
                binary_mask = np.where(mask_resized == 0, 255, 0).astype(np.uint8)
            else:
                binary_mask = np.where(mask_resized > 0, 255, 0).astype(np.uint8)
                
            return self._post_process(binary_mask)
        else:
            return np.zeros((h, w), dtype=np.uint8)