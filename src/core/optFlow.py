import cv2
import numpy as np

class OpticalFlowEngine:
    def __init__(self, scale=0.25):
        self.scale = scale
        self.prev_gray = None
        
        # Inicializa o DIS com o preset mais leve possível (ULTRAFAST)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        
        # Método fixo para logs e compatibilidade
        # self.method = "DIS_ULTRAFAST"
        self.method = "DIS_FAST"

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)

        # 1. Downscale para performance em tempo real
        prev_small = cv2.resize(self.prev_gray, None, fx=self.scale, fy=self.scale)
        curr_small = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

        # 2. Calcula o Fluxo com DIS
        flow_small = self.dis.calc(prev_small, curr_small, None)
        flow_small = cv2.GaussianBlur(flow_small, (15, 15), 5.0)

        # 3. Limite mínimo de movimento (reduz ruído de câmera)
        mag_threshold = 1.5
        u = flow_small[..., 0]
        v = flow_small[..., 1]
        magnitude = np.hypot(u, v)

        # Zera os vetores irrelevantes
        flow_small[magnitude < mag_threshold] = 0
        
        # Amplifica o movimento restante
        flow_small = flow_small * 2.5 #FIXME

        # 4. Upscale de volta para a resolução original
        h, w = gray.shape
        flow = cv2.resize(flow_small, (w, h))
        
        # Ajusta a magnitude após o redimensionamento
        flow *= (1.0 / self.scale)

        self.prev_gray = gray.copy()
        return flow