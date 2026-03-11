import cv2
import numpy as np
import time
import mediapipe as mp
import os
import sys
from scipy import ndimage 

class BackgroundProcessor:
    # Constantes dos Modos (Agora apenas 2)
    MODE_OTSU = 0
    MODE_AI_SELFIE = 1

    def __init__(self):
        self.mode = self.MODE_AI_SELFIE
        self.bg_base = None       
        
        # Pesos do Modo Estático (Otsu)
        self.weight_y = 0.3
        self.weight_cr = 1.2
        self.weight_cb = 1.2

        # Memória da IA
        self.current_ai_mask = None
        self.seg_selfie = None
        
        self._init_ai_models()

    def get_resource_path(self, relative_path):
        """ Retorna o caminho absoluto do arquivo... """
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        return os.path.join(base_dir, relative_path)

    def _init_ai_models(self):
        """Inicializa o modelo assíncrono do MediaPipe."""
        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        def _save_result(result, output_image: mp.Image, timestamp_ms: int):
            if result.category_mask is not None:
                self.current_ai_mask = result.category_mask.numpy_view().copy()

        model_path = self.get_resource_path('assets/models/selfie_segmenter_landscape.tflite')

        if os.path.exists(model_path):
            options_selfie = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                output_category_mask=True,
                result_callback=_save_result
            )
            self.seg_selfie = ImageSegmenter.create_from_options(options_selfie)
        else:
            print(f"AVISO: Arquivo do modelo de IA não encontrado em -> {model_path}")

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
        elif self.mode == self.MODE_AI_SELFIE:
            return self._mask_ai(frame)
            
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # ==========================================
    # PÓS-PROCESSAMENTO ROBUSTO
    # ==========================================
    def _post_process(self, mask, min_area_threshold=1000):
        h, w = mask.shape[:2]

        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_bridge, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        bottom_labels = set(labels[h - 1, :]) - {0}
        top_labels = set(labels[0, :]) - {0}
        target_labels = bottom_labels - top_labels 

        valid_labels = [
            label for label in target_labels 
            if stats[label, cv2.CC_STAT_AREA] >= min_area_threshold
        ]
        
        filtered_mask = np.zeros_like(mask)
        if valid_labels:
            filtered_mask[np.isin(labels, valid_labels)] = 255
            
        filtered_mask = ndimage.binary_fill_holes(filtered_mask).astype(np.uint8) * 255

        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(filtered_mask, kernel_dilation, iterations=1)
                
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
        
        _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._post_process(mask)

    # ==========================================
    # 2. MÉTODO DE INTELIGÊNCIA ARTIFICIAL (Selfie)
    # ==========================================
    def _mask_ai(self, frame):
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)

        try:
            if self.seg_selfie:
                self.seg_selfie.segment_async(mp_image, ts)
        except Exception:
            pass

        if self.current_ai_mask is not None:
            mask_resized = cv2.resize(self.current_ai_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Inverte a máscara (Selfie segmenter usa 0 para pessoa)
            binary_mask = np.where(mask_resized == 0, 255, 0).astype(np.uint8)
            return self._post_process(binary_mask)
        else:
            return np.zeros((h, w), dtype=np.uint8)