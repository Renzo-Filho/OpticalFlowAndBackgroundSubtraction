import os
import cv2
import time
import numpy as np
import tkinter as tk
import io

from core.optFlow import OpticalFlowEngine
from core.background import BackgroundProcessor
from core.pose import PoseProcessor
from effects.pose_effects import FlowBenderEffect, NeonSkeletonEffect, KamehamehaEffect, KamehamehaEffect2
from effects.geometry import ArrowEffect, GridWarpEffect, DelaunayConstellationEffect, ShatteredGlassEffect
from effects.physics import WaveEquationEffect, KineticParticleEffect, FluidPaintEffect, GlowingWaveEffect, NavierStokesFluidEffect, NavierStokesRealityEffect
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
            "AI: Selfie Segmenter",
            "AI: Pose Segmenter"
        ]
        
        # 3. Effects Playlist
        self.effects = [
            ShowMaskEffect(),
            ######## FILTROS
            HeatmapEffect(),
            CartoonEffect(),
            NegativeEffect(),

            ######### POSE
            KamehamehaEffect2(),
            FlowBenderEffect(),
            NeonSkeletonEffect(color=(255, 50, 255)),
            NeonSilhouetteEffect(color=(0, 255, 255)), 

            ######### CLONES
            TimeTunnelEffect(max_clones=10, frame_delay=15),
            DrosteTunnelEffect(scale_factor=0.98), 

            ######## FLUID 
            FluidPaintEffect(decay=0.985),
            NavierStokesFluidEffect(),
            NavierStokesRealityEffect(),
            WaveEquationEffect(damping=0.98),
            GlowingWaveEffect(),

            ######## GEOMETRY 
            GridWarpEffect(step=40, amplitude=8.0),
            ArrowEffect(step=30),
            ShatteredGlassEffect(),
            DelaunayConstellationEffect(max_points=200),

            ######## OTHERS 
            KineticParticleEffect()
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
        
        # --- CARREGAMENTO DE ASSETS GLOBAIS ---
        # A. Logo (Top Left)
        logo_path = os.path.join(base_dir, 'assets', 'icons', 'ime.png')
        self.logo_raw = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        self.logo = None
        if self.logo_raw is not None:
            target_height = int(h * 0.08) 
            aspect_ratio = self.logo_raw.shape[1] / self.logo_raw.shape[0]
            target_width = int(target_height * aspect_ratio * 1.1)
            self.logo = cv2.resize(self.logo_raw, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # B. QR Code (Bottom Right)
        qr_path = os.path.join(base_dir, 'assets', 'icons', 'demo-qrcode.png')
        self.qr_raw = cv2.imread(qr_path, cv2.IMREAD_UNCHANGED)
        self.qr_code = None
        if self.qr_raw is not None:
            qr_height = int(h * 0.15)
            aspect_ratio = self.qr_raw.shape[1] / self.qr_raw.shape[0]
            qr_width = int(qr_height * aspect_ratio)
            self.qr_code = cv2.resize(self.qr_raw, (qr_width, qr_height), interpolation=cv2.INTER_AREA)
        else:
            print(f"[AVISO] QR Code não encontrado no caminho: {qr_path}")

        # C. Equações (Top Right)
        self.equation_sprites = {}
        self._generate_latex_sprites(w, h) # Passamos a altura da câmera para controle de tamanho
        
        # Setup final da Janela
        root = tk.Tk()
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()
        self.target_ratio = self.screen_w / self.screen_h
        root.destroy() 

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_w, self.screen_h)

    def _generate_latex_sprites(self, screen_w, screen_h):
        """Renderiza as equações matemáticas mapeadas com limite de altura e largura."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERRO] Instale o matplotlib (pip install matplotlib) para renderizar LaTeX.")
            return

        formulas_map = {
            "SHOW_MASK": r"M(x,y) = |I_{\mathrm{curr}} - I_{\mathrm{bg}}| > \tau",
            "COLORFUL_HEAT": r"H = \mathrm{ColorMap}(I_{\mathrm{gray}})",
            "CARTOON_FILTER": r"I_{\mathrm{out}} = \mathrm{Bilateral}(I) \odot \mathrm{Edge}(I)",
            "NEGATIVE_FILTER": r"I_{\mathrm{out}} = 255 - I_{\mathrm{in}}",
            "KAMEHAMEHA": r"E = \frac{1}{2}mv^2 + \int P \, dt",
            "FLOW_BENDER": r"\mathbf{F} = m\mathbf{a} + q(\mathbf{v} \times \mathbf{B})",
            "NEON_SKELETON": r"S = \bigcup \overline{P_i P_j}",
            "NEON_SILHOUETTE": r"\nabla M = (M \oplus B) - (M \ominus B)",
            "TIME_TUNNEL": r"T(t) = \sum_{k=0}^{N} \alpha^k I(t - k \Delta t)",
            "DROSTE_TUNNEL": r"D(x) = D(s \cdot x) + I_{\mathrm{curr}}(x)",
            "FLUID_PAINT_BG": r"\frac{\partial C}{\partial t} = -(\mathbf{v} \cdot \nabla)C",
            "SMOOTH_FLUID_SIM": r"\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = \nu \nabla^2 \mathbf{v}",
            "FLUID_REALITY": r"\rho \frac{D\mathbf{v}}{Dt} = -\nabla p + \mu \nabla^2 \mathbf{v}",
            "WAVE_EQUATION": r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u",
            "NEON_WAVE": r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t}",
            "GRID_WARP": r"\mathbf{P}' = \mathbf{P} + A \cdot \mathbf{v}(\mathbf{P})",
            "ARROWS": r"\nabla I \cdot \mathbf{v} + \frac{\partial I}{\partial t} = 0",
            "SHATTERED_GLASS": r"V_i = \{ x \mid d(x, P_i) \leq d(x, P_j) \}",
            "DELAUNAY_CONSTELLATION": r"\mathrm{DT}(P) \Leftrightarrow \mathrm{Circ}(\Delta) \cap P = \emptyset",
            "KINETIC_PARTICLES": r"\frac{d\mathbf{p}}{dt} = \mathbf{v}(\mathbf{p}, t)"
        }

        fig = plt.figure(figsize=(8, 2), dpi=200) 
        
        print("[INFO] Renderizando equações matemáticas...")
        
        # --- DEFINIÇÃO DE LIMITES ---
        # Altura ideal: 6% da tela. Largura máxima permitida: 35% da tela.
        target_h = max(30, int(screen_h * 0.06))
        max_w = int(screen_w * 0.35) 
        pad_x, pad_y = 20, 15 

        for effect_name, tex in formulas_map.items():
            fig.clf()
            fig.patch.set_alpha(0.0) 
            
            plt.text(0.5, 0.5, f"${tex}$", size=24, color="#FFFFFF", ha='center', va='center')
            plt.axis('off')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, pad_inches=0.0)
            buf.seek(0)
            
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            sprite = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED) 
            
            if sprite is not None:
                alpha_channel = sprite[:, :, 3] 
                y_idx, x_idx = np.where(alpha_channel > 0) 
                
                if len(y_idx) > 0 and len(x_idx) > 0:
                    y_min, y_max = np.min(y_idx), np.max(y_idx)
                    x_min, x_max = np.min(x_idx), np.max(x_idx)
                    sprite = sprite[y_min:y_max+1, x_min:x_max+1]
                    
                    # --- NOVO CÁLCULO DE REDIMENSIONAMENTO ---
                    sh, sw = sprite.shape[:2]
                    
                    scale_h = target_h / float(sh)
                    scale_w = max_w / float(sw)
                    
                    # Usa a escala mais restritiva para garantir que caiba em ambos os limites
                    scale = min(scale_h, scale_w)
                    
                    new_w = int(sw * scale)
                    new_h = int(sh * scale)
                    
                    resized_text = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    box_w = new_w + 2 * pad_x
                    box_h = new_h + 2 * pad_y
                    
                    box = np.zeros((box_h, box_w, 4), dtype=np.float32) 
                    box[:, :] = (30, 30, 30, 160) 
                    
                    alpha_t = resized_text[:, :, 3:4] / 255.0
                    
                    for c in range(3):
                        box[pad_y:pad_y+new_h, pad_x:pad_x+new_w, c] = \
                            alpha_t[:, :, 0] * resized_text[:, :, c] + \
                            (1.0 - alpha_t[:, :, 0]) * box[pad_y:pad_y+new_h, pad_x:pad_x+new_w, c]
                    
                    box_alpha = box[pad_y:pad_y+new_h, pad_x:pad_x+new_w, 3] / 255.0
                    combined_alpha = alpha_t[:, :, 0] + box_alpha - (alpha_t[:, :, 0] * box_alpha)
                    box[pad_y:pad_y+new_h, pad_x:pad_x+new_w, 3] = combined_alpha * 255
                    
                    self.equation_sprites[effect_name] = box.astype(np.uint8)
            else:
                print(f"[AVISO] Falha ao decodificar a equação para: {effect_name}")
        
        plt.close(fig)
        print("[INFO] Matriz de Equações mapeadas pronta e envelopada!")

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            cam_h, cam_w = frame.shape[:2]
            cam_ratio = cam_w / cam_h

            if cam_ratio < self.target_ratio:
                new_h = int(cam_w / self.target_ratio)
                y_offset = (cam_h - new_h) // 2
                frame = frame[y_offset:y_offset+new_h, :]
            elif cam_ratio > self.target_ratio:
                new_w = int(cam_h * self.target_ratio)
                x_offset = (cam_w - new_w) // 2
                frame = frame[:, x_offset:x_offset+new_w]
            
            # --- Processing ---
            flow = self.flow_engine.update(frame)
            pose = self.pose_processor.process(frame)
            mask = self.bg_processor.get_mask(frame, flow, pose_mask=self.pose_processor.current_mask)
                        
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
            out_h, out_w = display_frame.shape[:2]
            
            # Margem global de 2% para os overlays
            margin_x = int(out_w * 0.02)
            margin_y = int(out_h * 0.02)

            # --- Overlays de Informação e Branding ---
            
            # 1. Equação no Top Right 
            eq_sprite = self.equation_sprites.get(current_effect.name)
            if eq_sprite is not None:
                eq_h, eq_w = eq_sprite.shape[:2]
                eq_x = out_w - eq_w - margin_x
                eq_y = margin_y
                self._overlay_transparent(display_frame, eq_sprite, eq_x, eq_y)

            # 2. QR Code no Bottom Right
            if self.qr_code is not None:
                qr_h, qr_w = self.qr_code.shape[:2]
                qr_x = out_w - qr_w - margin_x
                qr_y = out_h - qr_h - margin_y
                self._overlay_transparent(display_frame, self.qr_code, qr_x, qr_y)

            # 3. Logo no Top Left
            if self.logo is not None:
                self._overlay_transparent(display_frame, self.logo, margin_x, margin_y)

            # --- HUD & Status Logic ---
            method_label = f"MODO: {self.mask_modes_names[self.bg_processor.mode]}"
            status_parts = []
            
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
        elif key == ord('m'): self.bg_processor.mode = (self.bg_processor.mode + 1) % 3   
        elif key == ord('b'): self.bg_processor.capture_static_model(self.cap)
        elif key == ord('d'): self.hud.toggle()
        elif key == ord('r'): self.effects[self.current_idx].reset()
        return False
    
    def cleanup(self):
        print("Finalizando e resetando configurações...")
        if os.name == 'posix':
            try:
                os.system("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=3") 
                os.system("v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=1")
            except Exception:
                pass

        if self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        
    def _overlay_transparent(self, background, overlay, x, y):
        if overlay is None:
            return
        
        h, w = background.shape[:2]
        sh, sw = overlay.shape[:2]

        y1, y2 = max(0, y), min(h, y + sh)
        x1, x2 = max(0, x), min(w, x + sw)

        oy1, oy2 = max(0, -y), sh - max(0, (y + sh) - h)
        ox1, ox2 = max(0, -x), sw - max(0, (x + sw) - w)

        if y1 >= y2 or x1 >= x2:
            return

        if overlay.shape[2] == 4:
            alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0
            # Adicionado .astype(np.uint8) para proteção contra crash de broadcasting de memória
            blended = (alpha * overlay[oy1:oy2, ox1:ox2, :3] + (1.0 - alpha) * background[y1:y2, x1:x2])
            background[y1:y2, x1:x2] = blended.astype(np.uint8) 
        else:
            background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]

if __name__ == "__main__":
    app = ExhibitionApp()
    try:
        app.run()
    finally:
        app.cleanup()