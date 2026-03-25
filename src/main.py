import os
import cv2
import time
import numpy as np
import tkinter as tk
from core.optFlow import OpticalFlowEngine
from core.background import BackgroundProcessor
from core.pose import PoseProcessor
from effects.pose_effects import FlowBenderEffect, NeonSkeletonEffect
from effects.overlay import MathChromaKeyEffect
from effects.geometry import ArrowEffect, GridWarpEffect, DelaunayConstellationEffect
from effects.physics import WaveEquationEffect, KineticParticleEffect, FluidPaintEffect
from effects.filters import CartoonEffect, HeatmapEffect, NegativeEffect, CyberGlitchEffect, NeonSilhouetteEffect
from effects.temporal import TimeTunnelEffect, DrosteTunnelEffect
from effects.debug import ShowMaskEffect
from utils.hud import HUD

class ExhibitionApp:
    def __init__(self):
        # 1. Initialize Camera
        self.cap = cv2.VideoCapture(0) # FIXME
        ret, frame = self.cap.read()
        if not ret: raise RuntimeError("Could not initialize camera.")
        
        # 2. Engine Setup
        self.flow_engine = OpticalFlowEngine(scale=0.5)
        self.bg_processor = BackgroundProcessor()
        self.pose_processor = PoseProcessor()

        # HUD names
        self.mask_modes_names = [
            "Static: YCrCb + Otsu", 
            "AI: Selfie Segmenter"
        ]
        
        # 3. Effects Playlist
        self.effects = [
            ShowMaskEffect(),
            MathChromaKeyEffect(),
            FlowBenderEffect(),
            NeonSkeletonEffect(color=(255, 50, 255)),
            HeatmapEffect(),     # Adds a colorful, thermal-camera vibe
            CartoonEffect(),     # Adds a comic-book aesthetic
            NegativeEffect(),    # Classic high-contrast 
            WaveEquationEffect(damping=0.98),
            DelaunayConstellationEffect(max_points=200),
            KineticParticleEffect(),
            CyberGlitchEffect(),
            NeonSilhouetteEffect(color=(0, 255, 255)), # Amarelo, ou (255, 255, 0) para Ciano
            TimeTunnelEffect(max_clones=10, frame_delay=15),
            #DrosteTunnelEffect(scale_factor=0.94), # Faster recession
            DrosteTunnelEffect(scale_factor=0.98), # Slow, hypnotic recession
            FluidPaintEffect(decay=0.985),
            GridWarpEffect(step=40, amplitude=10.0),
            ArrowEffect(step=30)
        ]
        self.current_idx = 0
        self.effect_duration = 30.0
        self.start_time = time.time()
        
        # 4. Window & HUD Setup
        h, w = frame.shape[:2]
        self.hud = HUD(w, h)
        self.hud.active = False
        self.window_name = "Exhibition"

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(base_dir, 'assets', 'icons', 'ime.png')
        self.logo_raw = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        self.logo = None
        
        if self.logo_raw is not None:
            target_height = int(h * 0.08) 
            aspect_ratio = self.logo_raw.shape[1] / self.logo_raw.shape[0]
            target_width = int(target_height * aspect_ratio * 1.1)
            self.logo = cv2.resize(self.logo_raw, (target_width, target_height), interpolation=cv2.INTER_AREA)
        else:
            print(f"[AVISO] Logo não encontrada no caminho: {logo_path}")
            
        root = tk.Tk()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()
        self.target_ratio = self.screen_w / self.screen_h
        root.destroy() 

        # Como vamos enviar a imagem já na proporção certa, não precisamos mais do FREERATIO
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_w, self.screen_h)

        #cv2.createTrackbar("Peso Y", self.window_name, 30, 300, self.update_weights)
        #cv2.createTrackbar("Peso Cr", self.window_name, 120, 300, self.update_weights)
        #cv2.createTrackbar("Peso Cb", self.window_name, 120, 300, self.update_weights)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) # Mirror view

            cam_h, cam_w = frame.shape[:2]
            cam_ratio = cam_w / cam_h

            if cam_ratio < self.target_ratio:
                # Câmera mais "quadrada" que a tela. Cortamos o topo e a base.
                new_h = int(cam_w / self.target_ratio)
                y_offset = (cam_h - new_h) // 2
                frame = frame[y_offset:y_offset+new_h, :]
            elif cam_ratio > self.target_ratio:
                # Câmera mais "larga" que a tela. Cortamos as laterais.
                new_w = int(cam_h * self.target_ratio)
                x_offset = (cam_w - new_w) // 2
                frame = frame[:, x_offset:x_offset+new_w]
            
            # --- Processing ---
            t0 = time.time()
            flow = self.flow_engine.update(frame)
            t1 = time.time()
            
            mask = self.bg_processor.get_mask(frame, flow)

            pose = self.pose_processor.process(frame)
                        
            # --- Auto Rotation ---
            elapsed = time.time() - self.start_time
            if elapsed > self.effect_duration:
                self.next_effect()
                elapsed = 0

            # --- Render Effect ---
            current_effect = self.effects[self.current_idx]
            try:
                output = current_effect.apply(frame, flow, mask, pose=pose)
            except Exception as e:
                print(f"Error in {current_effect.name}: {e}")
                output = frame

            display_frame = output.copy()

            # --- HUD & Status Logic ---
            # Pega o nome do modo atual da nossa lista
            method_label = f"MODO: {self.mask_modes_names[self.bg_processor.mode]}"

            status_parts = []
            
            # Alerta se estivermos em modo estático e não houver background
            if self.bg_processor.mode == 0 and self.bg_processor.bg_base is None:
                status_parts.append("No BG captured (Press 'b')")

            full_status = " | ".join(status_parts)

            self.hud.render(
                display_frame, 
                current_effect.name, 
                method_label, 
                remaining_time=(self.effect_duration - elapsed),
                extra_info=full_status
            )

            if self.logo is not None:
                out_h, out_w = display_frame.shape[:2]
                logo_h, logo_w = self.logo.shape[:2]
                
                # Margem relativa: 3% da largura e altura da tela
                margin_x = int(out_w * 0.02)
                margin_y = int(out_h * 0.02)
                
                x_pos = margin_x
                y_pos = out_h - logo_h - margin_y
                
                self._overlay_transparent(display_frame, self.logo, x_pos, y_pos)

            cv2.imshow(self.window_name, display_frame)
            
            if self.handle_input():
                break

        self.cleanup()

    def next_effect(self):
        self.current_idx = (self.current_idx + 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def last_effect(self):
        self.current_idx = (self.current_idx - 1) % len(self.effects)
        self.effects[self.current_idx].reset()
        self.start_time = time.time()

    def handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]: return True
        elif key == ord('n'): self.next_effect()
        elif key == ord('l'): self.last_effect()
        elif key == ord('m'): self.bg_processor.mode = (self.bg_processor.mode + 1) % 2   
        elif key == ord('b'): self.bg_processor.capture_static_model(self.cap)
        elif key == ord('d'): self.hud.toggle()
        elif key == ord('r'): self.effects[self.current_idx].reset()
        return False
    
    def cleanup(self):
        print("Finalizando e resetando configurações...")
        
        # Comandos v4l2 só funcionam no Linux
        if os.name == 'posix':
            try:
                os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=3") 
                os.system("v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=1")
            except Exception:
                pass

        if self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        
    def update_weights(self, val):
        """Callback chamado sempre que um slider é movido pelo mouse."""
        # Lê os valores atuais (0 a 300) da janela
        y_val = cv2.getTrackbarPos("Peso Y", self.window_name)
        cr_val = cv2.getTrackbarPos("Peso Cr", self.window_name)
        cb_val = cv2.getTrackbarPos("Peso Cb", self.window_name)
        
        # Divide por 100 para voltar a ter casas decimais (ex: 120 vira 1.2)
        # e atualiza o processador de background
        self.bg_processor.weight_y = y_val / 100.0
        self.bg_processor.weight_cr = cr_val / 100.0
        self.bg_processor.weight_cb = cb_val / 100.0
        
        print(f"Pesos Atualizados -> Y: {self.bg_processor.weight_y}, Cr: {self.bg_processor.weight_cr}, Cb: {self.bg_processor.weight_cb}")

    def _overlay_transparent(self, background, overlay, x, y):
        """Cola uma imagem com canal Alpha (transparência) sobre o fundo."""
        if overlay is None:
            return
        
        h, w = background.shape[:2]
        sh, sw = overlay.shape[:2]

        y1, y2 = max(0, y), min(h, y + sh)
        x1, x2 = max(0, x), min(w, x + sw)

        oy1, oy2 = max(0, -y), sh - max(0, (y + sh) - h)
        ox1, ox2 = max(0, -x), sw - max(0, (x + sw) - w)

        # Retorna se a imagem estiver completamente fora da tela
        if y1 >= y2 or x1 >= x2:
            return

        # Mistura a imagem usando o canal Alpha se ele existir (PNG transparente)
        if overlay.shape[2] == 4:
            alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0
            background[y1:y2, x1:x2] = (alpha * overlay[oy1:oy2, ox1:ox2, :3] + 
                                        (1.0 - alpha) * background[y1:y2, x1:x2])
        else:
            background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]

if __name__ == "__main__":
    app = ExhibitionApp()
    try:
        app.run()
    finally:
        # The 'finally' block ensures cleanup runs even if the code crashes
        app.cleanup()