import cv2
import numpy as np
import random
from .baseEffect import BaseEffect

class FluidPaintEffect(BaseEffect):
    def __init__(self, decay=0.985, advect_gain=2.0):
        super().__init__("FLUID_PAINT_BG")
        self.decay = decay
        self.advect_gain = advect_gain
        self.canvas = None

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        
        # Initialize canvas if needed
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.float32)

        # 1. Prepare Masks
        fg_float = mask.astype(np.float32) / 255.0
        fg_3 = fg_float[..., None]
        bg_3 = 1.0 - fg_3

        # 2. Advection: Move the 'ink' along the flow
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x - self.advect_gain * flow[..., 0]
        map_y = grid_y - self.advect_gain * flow[..., 1]
        
        advected = cv2.remap(
            self.canvas, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        allowed_zone = cv2.dilate(mask, kernel_zone, iterations=1)
        allowed_float = allowed_zone.astype(np.float32) / 255.0

        # 3. Injection: Add color where motion happens (Background only)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_inject = np.zeros((h, w, 3), dtype=np.uint8)
        hsv_inject[..., 0] = ang * 180 / np.pi / 2
        hsv_inject[..., 1] = 255
        hsv_inject[..., 2] = 255
        
        color_inject = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Only inject color in motion areas that are NOT the person
        motion_mask = (mag > 2.0).astype(np.float32)
        inject_weight = cv2.GaussianBlur(motion_mask, (9, 9), 0)[..., None] * bg_3
        
        # Update Canvas
        self.canvas = (advected * self.decay) + (color_inject * inject_weight * 0.3)
        self.canvas *= bg_3 # Zero out the person's area to prevent 'ghosting' behind them

        # 4. Composite: Person + Fluid Background
        out = (frame.astype(np.float32) * fg_3) + self.canvas
        return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self):
        self.canvas = None

class WaveEquationEffect(BaseEffect):
    """
    Simula a Equação da Onda 2D em tempo real com atenuação espacial (Viscosidade).
    Referência Matemática:
    $$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t}$$
    """
    def __init__(self, damping=0.94, resolution_scale=0.5):
        super().__init__("WAVE_EQUATION")
        self.damping = damping
        self.scale = resolution_scale
        self.u_prev = None
        self.u_curr = None
        self.kernel = np.array([[0, 0.5, 0], 
                                [0.5, 0, 0.5], 
                                [0, 0.5, 0]], dtype=np.float32)

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        small_w = int(w * self.scale)
        small_h = int(h * self.scale)

        if self.u_prev is None or self.u_prev.shape != (small_h, small_w):
            self.u_prev = np.zeros((small_h, small_w), dtype=np.float32)
            self.u_curr = np.zeros((small_h, small_w), dtype=np.float32)

        flow_small = cv2.resize(flow, (small_w, small_h))
        mag, _ = cv2.cartToPolar(flow_small[..., 0], flow_small[..., 1])
        
        # Aumentei o limiar para 3.0 (gera menos ondas com micromovimentos)
        energy_mask = mag > 3.0
        
        # 1. REDUÇÃO DE FORÇA: Multiplicador caiu de 8.0 para 1.5
        self.u_curr[energy_mask] += mag[energy_mask] * 2.5

        # Resolução com Diferenças Finitas
        u_next = cv2.filter2D(self.u_curr, -1, self.kernel) - self.u_prev
        u_next *= self.damping
        
        # 2. TETO DE AMPLITUDE: Impede que a onda exploda (o valor 50 é o limite de altura)
        u_next = np.clip(u_next, -70.0, 70.0)
        
        u_next = cv2.GaussianBlur(u_next, (3, 3), 0)

        self.u_prev = self.u_curr
        self.u_curr = u_next

        u_full = cv2.resize(self.u_curr, (w, h), interpolation=cv2.INTER_LINEAR)
        grad_x = cv2.Sobel(u_full, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(u_full, cv2.CV_32F, 0, 1, ksize=3)

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # 3. REDUÇÃO DA DISTORÇÃO VISUAL: Multiplicador caiu de 1.5 para 0.6
        map_x = grid_x + (grad_x * 0.7)
        map_y = grid_y + (grad_y * 0.7)

        water_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return water_frame

    def reset(self):
        self.u_prev = None
        self.u_curr = None

class KineticParticleEffect(BaseEffect):
    """
    Sistema de partículas advectado diretamente pelo Campo Vetorial do usuário.
    Referência Matemática:
    A cinemática de fluidos define a taxa de variação da posição de uma partícula $\vec{p}$
    imersa em um campo de velocidade $\vec{v}$ como:
    $$\frac{d\vec{p}}{dt} = \vec{v}(\vec{p}, t)$$
    """
    def __init__(self, num_particles=2000):
        super().__init__("KINETIC_PARTICLES")
        # Estado: [x, y, vx, vy, life]
        self.particles = np.zeros((num_particles, 5), dtype=np.float32)
        self.num_particles = num_particles

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)

        # 1. Encontra pixels válidos na pessoa para spawnar partículas
        y_idx, x_idx = np.where(mask > 127)
        
        # 2. Revive partículas mortas em posições aleatórias sobre o corpo da pessoa
        if len(x_idx) > 0:
            dead_idx = np.where(self.particles[:, 4] <= 0)[0]
            spawn_count = min(len(dead_idx), 50) # Taxa de emissão
            
            if spawn_count > 0:
                random_choices = np.random.choice(len(x_idx), spawn_count)
                for i, choice in enumerate(random_choices):
                    idx = dead_idx[i]
                    self.particles[idx, 0] = x_idx[choice]
                    self.particles[idx, 1] = y_idx[choice]
                    self.particles[idx, 2] = 0.0 # vx inicial
                    self.particles[idx, 3] = 0.0 # vy inicial
                    self.particles[idx, 4] = random.uniform(20, 60) # frames de vida

        # 3. Atualização Física Guiada pelo Optical Flow
        active = self.particles[:, 4] > 0
        if np.any(active):
            px = np.clip(self.particles[active, 0].astype(int), 0, w - 1)
            py = np.clip(self.particles[active, 1].astype(int), 0, h - 1)

            # Lê a força do movimento exata onde a partícula está
            flow_vectors = flow[py, px]
            
            # Atualiza velocidade (Inércia + Força do Flow)
            self.particles[active, 2] = (self.particles[active, 2] * 0.85) + (flow_vectors[:, 0] * 1.5)
            self.particles[active, 3] = (self.particles[active, 3] * 0.85) + (flow_vectors[:, 1] * 1.5)

            # Adiciona gravidade leve para baixo
            self.particles[active, 3] += 0.5 

            # Atualiza posição
            self.particles[active, 0] += self.particles[active, 2]
            self.particles[active, 1] += self.particles[active, 3]

            # Reduz a vida
            self.particles[active, 4] -= 1

            # 4. Renderização
            render_x = self.particles[active, 0]
            render_y = self.particles[active, 1]
            
            pts = np.vstack((render_x, render_y)).T.astype(np.int32)
            for pt in pts:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(canvas, tuple(pt), 2, (150, 200, 255), -1)

        # Escurece o usuário para destacar as partículas e soma as imagens
        frame_dark = (frame.astype(np.float32) * 0.9).astype(np.uint8)
        out = cv2.add(frame_dark, canvas)

        return out

    def reset(self):
        self.particles[:, 4] = 0