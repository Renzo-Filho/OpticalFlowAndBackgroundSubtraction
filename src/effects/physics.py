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
        
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.float32)

        fg_float = mask.astype(np.float32) / 255.0
        fg_3 = fg_float[..., None]
        bg_3 = 1.0 - fg_3

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x - self.advect_gain * flow[..., 0]
        map_y = grid_y - self.advect_gain * flow[..., 1]
        
        advected = cv2.remap(
            self.canvas, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Cria uma margem de segurança ao redor da pessoa
        kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        allowed_zone = cv2.dilate(mask, kernel_zone, iterations=1)
        allowed_float = allowed_zone.astype(np.float32) / 255.0

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv_inject = np.zeros((h, w, 3), dtype=np.uint8)
        hsv_inject[..., 0] = ang * 180 / np.pi / 2
        hsv_inject[..., 1] = 255
        hsv_inject[..., 2] = 255
        
        color_inject = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)
        
        motion_mask = (mag > 2.0).astype(np.float32)
        
        # A MÁGICA AQUI: O allowed_float impede que sombras na parede pintem o fundo!
        inject_weight = cv2.GaussianBlur(motion_mask, (9, 9), 0)[..., None] * bg_3 * allowed_float[..., None]
        
        self.canvas = (advected * self.decay) + (color_inject * inject_weight * 0.3)
        self.canvas *= bg_3 

        out = (frame.astype(np.float32) * fg_3) + self.canvas
        return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self):
        self.canvas = None

class NavierStokesFluidEffect(BaseEffect):
    def __init__(self, sim_scale=0.35, fluid_decay=0.985, vel_decay=0.90):
        super().__init__("SMOOTH_FLUID_SIM")
        self.sim_scale = sim_scale
        self.fluid_decay = fluid_decay
        self.vel_decay = vel_decay
        self.vel_x = None
        self.vel_y = None
        self.dye = None

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        sim_h = int(h * self.sim_scale)
        sim_w = int(w * self.sim_scale)

        if self.vel_x is None or self.vel_x.shape != (sim_h, sim_w):
            self.vel_x = np.zeros((sim_h, sim_w), dtype=np.float32)
            self.vel_y = np.zeros((sim_h, sim_w), dtype=np.float32)
            self.dye = np.zeros((sim_h, sim_w, 3), dtype=np.float32)

        flow_sim = cv2.resize(flow, (sim_w, sim_h))
        flow_smooth = cv2.GaussianBlur(flow_sim, (21, 21), 0)
        mag, ang = cv2.cartToPolar(flow_smooth[..., 0], flow_smooth[..., 1])

        # Cria a zona de permissão baseada na máscara
        mask_sim = cv2.resize(mask, (sim_w, sim_h))
        kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        allowed_zone = cv2.dilate(mask_sim, kernel_zone, iterations=1)
        allowed_float = (allowed_zone > 127).astype(np.float32)

        # Injeta velocidade (vento) apenas perto da pessoa
        self.vel_x += flow_smooth[..., 0] * 0.8 * allowed_float
        self.vel_y += flow_smooth[..., 1] * 0.8 * allowed_float

        hsv_inject = np.zeros((sim_h, sim_w, 3), dtype=np.uint8)
        hsv_inject[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv_inject[..., 1] = 255 
        hsv_inject[..., 2] = np.clip(mag * 35.0, 0, 255).astype(np.uint8)
        
        # Injeta tinta apenas perto da pessoa
        new_dye = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)
        new_dye *= allowed_float[..., None]
        
        self.dye += new_dye * 0.5 

        grid_y, grid_x = np.mgrid[0:sim_h, 0:sim_w].astype(np.float32)
        map_x = grid_x - self.vel_x
        map_y = grid_y - self.vel_y

        self.vel_x = cv2.remap(self.vel_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.vel_y = cv2.remap(self.vel_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.dye = cv2.remap(self.dye, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        self.vel_x = cv2.GaussianBlur(self.vel_x, (9, 9), 0)
        self.vel_y = cv2.GaussianBlur(self.vel_y, (9, 9), 0)
        self.dye = cv2.GaussianBlur(self.dye, (5, 5), 0)

        self.vel_x *= self.vel_decay
        self.vel_y *= self.vel_decay
        self.dye *= self.fluid_decay

        dye_full = cv2.resize(self.dye, (w, h), interpolation=cv2.INTER_LINEAR)
        dye_full = np.clip(dye_full, 0, 255).astype(np.uint8)

        mask_soft = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
        mask_soft = mask_soft[..., None]

        frame_dark = frame.astype(np.float32) * 0.4
        out = (dye_full.astype(np.float32) * (1.0 - mask_soft * 0.4)) + (frame_dark * mask_soft)

        return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self):
        self.vel_x = None
        self.vel_y = None
        self.dye = None

class NavierStokesRealityEffect(BaseEffect):
    def __init__(self, sim_scale=0.35, fluid_decay=0.96, vel_decay=0.90):
        super().__init__("FLUID_REALITY")
        self.sim_scale = sim_scale
        self.fluid_decay = fluid_decay
        self.vel_decay = vel_decay
        self.vel_x = None
        self.vel_y = None
        self.dye = None

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        sim_h = int(h * self.sim_scale)
        sim_w = int(w * self.sim_scale)

        if self.vel_x is None or self.vel_x.shape != (sim_h, sim_w):
            self.vel_x = np.zeros((sim_h, sim_w), dtype=np.float32)
            self.vel_y = np.zeros((sim_h, sim_w), dtype=np.float32)
            self.dye = np.zeros((sim_h, sim_w, 3), dtype=np.float32)

        flow_sim = cv2.resize(flow, (sim_w, sim_h))
        flow_smooth = cv2.GaussianBlur(flow_sim, (21, 21), 0)
        mag, ang = cv2.cartToPolar(flow_smooth[..., 0], flow_smooth[..., 1])

        # Cria a zona de permissão baseada na máscara
        mask_sim = cv2.resize(mask, (sim_w, sim_h))
        kernel_zone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        allowed_zone = cv2.dilate(mask_sim, kernel_zone, iterations=1)
        allowed_float = (allowed_zone > 127).astype(np.float32)

        self.vel_x += flow_smooth[..., 0] * 0.8 * allowed_float
        self.vel_y += flow_smooth[..., 1] * 0.8 * allowed_float

        hsv_inject = np.zeros((sim_h, sim_w, 3), dtype=np.uint8)
        hsv_inject[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv_inject[..., 1] = 255
        hsv_inject[..., 2] = np.clip(mag * 35.0, 0, 255).astype(np.uint8)
        
        new_dye = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)
        new_dye *= allowed_float[..., None]
        
        self.dye += new_dye * 0.5 

        grid_y, grid_x = np.mgrid[0:sim_h, 0:sim_w].astype(np.float32)
        map_x = grid_x - self.vel_x
        map_y = grid_y - self.vel_y

        self.vel_x = cv2.remap(self.vel_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.vel_y = cv2.remap(self.vel_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.dye = cv2.remap(self.dye, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        self.vel_x = cv2.GaussianBlur(self.vel_x, (9, 9), 0)
        self.vel_y = cv2.GaussianBlur(self.vel_y, (9, 9), 0)
        self.dye = cv2.GaussianBlur(self.dye, (5, 5), 0)

        self.vel_x *= self.vel_decay
        self.vel_y *= self.vel_decay
        self.dye *= self.fluid_decay

        dye_full = cv2.resize(self.dye, (w, h), interpolation=cv2.INTER_LINEAR)
        dye_full = np.clip(dye_full, 0, 255).astype(np.uint8)

        frame_dimmed = (frame.astype(np.float32) * 0.8).astype(np.uint8)
        out = cv2.add(frame_dimmed, dye_full)

        return out

    def reset(self):
        self.vel_x = None
        self.vel_y = None
        self.dye = None

class WaveEquationEffect(BaseEffect):
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
        
        # Filtra a injeção de energia APENAS para perto da pessoa
        mask_small = cv2.resize(mask, (small_w, small_h))
        allowed_zone = cv2.dilate(mask_small, np.ones((7, 7), np.uint8), iterations=1)
        
        energy_mask = (mag > 3.0) & (allowed_zone > 127)
        
        self.u_curr[energy_mask] += mag[energy_mask] * 2.5

        u_next = cv2.filter2D(self.u_curr, -1, self.kernel) - self.u_prev
        u_next *= self.damping
        u_next = np.clip(u_next, -70.0, 70.0)
        u_next = cv2.GaussianBlur(u_next, (3, 3), 0)

        self.u_prev = self.u_curr
        self.u_curr = u_next

        u_full = cv2.resize(self.u_curr, (w, h), interpolation=cv2.INTER_LINEAR)
        grad_x = cv2.Sobel(u_full, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(u_full, cv2.CV_32F, 0, 1, ksize=3)

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x + (grad_x * 0.7)
        map_y = grid_y + (grad_y * 0.7)

        water_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return water_frame

    def reset(self):
        self.u_prev = None
        self.u_curr = None

class GlowingWaveEffect(BaseEffect):
    def __init__(self, damping=0.95, resolution_scale=0.25, glow_color=(255, 200, 50)):
        super().__init__("NEON_WAVE")
        self.damping = damping
        self.scale = resolution_scale
        self.glow_color = np.array(glow_color, dtype=np.float32) 
        
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
        
        # Restringe detecção de movimento à silhueta
        mask_small = cv2.resize(mask, (small_w, small_h))
        allowed_zone = cv2.dilate(mask_small, np.ones((7, 7), np.uint8), iterations=1)
        
        motion_mask = ((mag > 2.0) & (allowed_zone > 127)).astype(np.uint8)
        
        kernel_edges = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_edges = cv2.morphologyEx(motion_mask, cv2.MORPH_GRADIENT, kernel_edges)
        
        active_edges = motion_edges > 0
        self.u_curr[active_edges] += mag[active_edges] * 5.0

        u_next = cv2.filter2D(self.u_curr, -1, self.kernel) - self.u_prev
        u_next *= self.damping
        u_next = np.clip(u_next, -60.0, 60.0) 
        u_next = cv2.GaussianBlur(u_next, (3, 3), 0)
        u_next *= 1.07

        self.u_prev = self.u_curr
        self.u_curr = u_next

        u_full = cv2.resize(self.u_curr, (w, h), interpolation=cv2.INTER_LINEAR)
        base_frame = cv2.addWeighted(frame, 0.8, np.zeros_like(frame), 0, 0)

        grad_x = cv2.Sobel(u_full, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(u_full, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(grad_x, grad_y)

        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x + (grad_x * 0.9)
        map_y = grid_y + (grad_y * 0.9)

        warped_frame = cv2.remap(base_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        glow_mask = np.clip(grad_mag * 0.8, 0, 255) / 255.0 
        glow_layer = (glow_mask[..., None] * self.glow_color).astype(np.uint8)

        return cv2.add(warped_frame, glow_layer)

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
