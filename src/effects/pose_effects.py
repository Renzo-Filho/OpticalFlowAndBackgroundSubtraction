import math
import cv2
import numpy as np
import random
import time
import os
import pygame
from .baseEffect import BaseEffect
from collections import deque

class FlowBenderEffect(BaseEffect):
    def __init__(self, num_particles=500): 
        super().__init__("FLOW_BENDER")
        self.particles = np.zeros((num_particles, 6), dtype=np.float32) 

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)
        pose = kwargs.get('pose', None)

        # Margem de segurança na borda inferior (15% da altura da tela)
        margin_bottom = int(h * 0.85)

        # 1. ENCONTRA AS MÃOS VALIDANDO COM A MÁSCARA E MARGEM
        hands_coords = []
        if pose is not None:
            for idx in (18,17,19,20):
                landmark = pose[idx]
                if landmark.presence > 0.5: 
                    hx, hy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Trava de segurança para não acessar um pixel fora do array
                    hx = np.clip(hx, 0, w - 1)
                    hy = np.clip(hy, 0, h - 1)
                    
                    # Condição 1: Longe da borda inferior
                    # Condição 2: O pixel correspondente à mão deve estar dentro da máscara da pessoa ativa
                    if hy < margin_bottom and mask[hy, hx] > 127:
                        hands_coords.append((hx, hy))

        # 2. GERA PARTÍCULAS "GORDAS" NA ÁREA DAS MÃOS
        if hands_coords:
            dead_idx = np.where(self.particles[:, 4] <= 0)[0]
            spawn_count = min(len(dead_idx), 25) 
            
            for i in range(spawn_count):
                idx = dead_idx[i]
                hand_x, hand_y = random.choice(hands_coords)
                
                self.particles[idx, 0] = hand_x + random.uniform(-30, 30)
                self.particles[idx, 1] = hand_y + random.uniform(-30, 30)
                self.particles[idx, 2] = random.uniform(-3, 3)
                self.particles[idx, 3] = random.uniform(-3, 3)
                
                life = random.uniform(25, 45)
                self.particles[idx, 4] = life
                self.particles[idx, 5] = life 

        # 3. MOVE AS PARTÍCULAS USANDO O OPTICAL FLOW
        active = self.particles[:, 4] > 0
        if np.any(active):
            px = np.clip(self.particles[active, 0].astype(int), 0, w - 1)
            py = np.clip(self.particles[active, 1].astype(int), 0, h - 1)

            flow_vectors = flow[py, px]
            
            self.particles[active, 2] = (self.particles[active, 2] * 0.85) + (flow_vectors[:, 0] * 1.5)
            self.particles[active, 3] = (self.particles[active, 3] * 0.85) + (flow_vectors[:, 1] * 1.5)

            self.particles[active, 0] += self.particles[active, 2]
            self.particles[active, 1] += self.particles[active, 3] - 2.5 
            self.particles[active, 4] -= 1

            # 4. RENDERIZAÇÃO
            render_data = self.particles[active]
            
            for pt_data in render_data:
                x, y = int(pt_data[0]), int(pt_data[1])
                life, max_life = pt_data[4], pt_data[5]
                
                if 0 <= x < w and 0 <= y < h:
                    scale = life / max_life
                    radius = int(35 * scale) 
                    
                    if radius > 0:
                        cv2.circle(canvas, (x, y), radius, (0, 70, 200), -1)
                        cv2.circle(canvas, (x, y), int(radius * 0.4), (150, 255, 255), -1)

        canvas = cv2.GaussianBlur(canvas, (15, 15), 0)
        return cv2.add(frame, canvas)

    def reset(self):
        self.particles[:, 4] = 0
        
class NeonSkeletonEffect(BaseEffect):
    """
    Rastreia a estrutura óssea do usuário e desenha um esqueleto de neon.
    Usa um acumulador temporal para deixar um rastro fotográfico (Light Painting).
    """
    def __init__(self, color=(255, 50, 255), decay=0.75): # Rosa Neon por padrão
        super().__init__("NEON_SKELETON")
        self.color = color
        self.decay = decay
        self.canvas = None
        
        # Mapeamento manual das conexões essenciais do MediaPipe Pose
        # Evita a captura do rosto e foca no corpo (Ombros, Braços, Tronco e Pernas)
        self.connections = [
            (6, 4), (1, 3),
            (10, 9),
            (11, 12), # Ombros
            (11, 13), (13, 15), # Braço esquerdo
            (12, 14), (14, 16), # Braço direito
            (11, 23), (12, 24), (23, 24), # Tronco
            (23, 25), (25, 27), # Perna esquerda
            (24, 26), (26, 28)  # Perna direita
        ]

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        pose = kwargs.get('pose', None)

        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros_like(frame)

        # Decaimento temporal para apagar o rastro aos poucos
        self.canvas = (self.canvas.astype(np.float32) * self.decay).astype(np.uint8)

        if pose is not None:
            # 1. Filtra os pontos visíveis na tela
            pts = {}
            for idx, landmark in enumerate(pose):
                # presence garante que a IA encontrou o ponto; visibility garante que não está escondido atrás do corpo
                if landmark.presence > 0.5 and landmark.visibility > 0.5:
                    pts[idx] = (int(landmark.x * w), int(landmark.y * h))

            # 2. Desenha os "ossos" (Linhas)
            for p1, p2 in self.connections:
                if p1 in pts and p2 in pts:
                    # Linha grossa e colorida (Aura externa)
                    cv2.line(self.canvas, pts[p1], pts[p2], self.color, 6, cv2.LINE_AA)
                    # Linha fina e branca (Núcleo de luz)
                    cv2.line(self.canvas, pts[p1], pts[p2], (255, 255, 255), 2, cv2.LINE_AA)

            # 3. Desenha as "juntas" (Círculos) nos nós da estrutura
            for idx, pt in pts.items():
                if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    cv2.circle(self.canvas, pt, 8, self.color, -1)
                    cv2.circle(self.canvas, pt, 4, (255, 255, 255), -1)

        # Escurece a câmera (deixa em 30% de brilho) para o Neon "pular" na tela
        frame_dark = (frame.astype(np.float32) * 0.3).astype(np.uint8)
        return cv2.add(frame_dark, self.canvas)

    def reset(self):
        self.canvas = None

class PopArtEchoesEffect(BaseEffect):
    """
    Deixa um rastro de clones estáticos preenchidos com cores sólidas e vibrantes,
    criando uma estética de arte moderna / Andy Warhol.
    """
    def __init__(self, max_clones=6, frame_delay=7):
        super().__init__("POP_ART_ECHOES")
        self.max_clones = max_clones
        self.frame_delay = frame_delay
        
        # Buffer armazena apenas as máscaras do passado para economizar memória
        self.buffer = deque(maxlen=max_clones * frame_delay + 1)
        
        # Paleta de cores vibrantes (Ciano, Magenta, Amarelo, Verde Neon, Laranja) em BGR
        self.colors = [
            (255, 255, 0),   # Ciano
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Amarelo
            (0, 255, 0),     # Verde Neon
            (0, 165, 255)    # Laranja
        ]
        self.color_idx = 0

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        
        # 1. Guarda a máscara atual na história
        self.buffer.append(mask.copy())
        
        # 2. Cria uma tela escura para as cores saltarem aos olhos
        canvas = np.zeros_like(frame)

        # 3. Desenha os clones do mais antigo (profundo) para o mais novo
        for i in range(self.max_clones, 0, -1):
            idx = -(i * self.frame_delay)
            
            if abs(idx) < len(self.buffer):
                past_mask = self.buffer[idx]
                
                # Escolhe uma cor diferente para cada clone na fila
                color = self.colors[(self.color_idx + i) % len(self.colors)]
                
                # Cria uma tela preenchida 100% com essa cor
                solid_color = np.zeros_like(frame)
                solid_color[:] = color
                
                # Recorta a tela colorida no formato da silhueta do passado e cola
                cv2.copyTo(solid_color, past_mask, canvas)

        # 4. Desenha o usuário atual (real) por cima de todas as silhuetas
        cv2.copyTo(frame, mask, canvas)
        
        # 5. Gira a paleta de cores lentamente para as cores "viajarem" pelo rastro
        if len(self.buffer) % 3 == 0:
            self.color_idx = (self.color_idx + 1) % len(self.colors)
            
        return canvas

    def reset(self):
        self.buffer.clear()

class MysticTrianglesEffect(BaseEffect):
    """
    Rastreia os pulsos e desenha mandalas feitas de triângulos giratórios.
    Usa ondas senoidais para fazer as formas encolherem e expandirem organicamente.
    """
    def __init__(self, base_radius=60, pulse_amplitude=35, pulse_speed=5.0, rotation_speed=2.0):
        super().__init__("MYSTIC_TRIANGLES")
        self.base_radius = base_radius
        self.pulse_amplitude = pulse_amplitude  # O quanto o triângulo cresce/encolhe
        self.pulse_speed = pulse_speed          # A velocidade da "respiração"
        self.rotation_speed = rotation_speed    # A velocidade do giro

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        
        # Escurecemos um pouco o ambiente para a magia brilhar
        frame_dark = (frame.astype(np.float32) * 0.6).astype(np.uint8)
        out = frame_dark.copy()
        
        pose = kwargs.get('pose', None)

        # O Tempo contínuo é o motor da animação
        t = time.time()
        
        # Matemática da Pulsação: math.sin() varia suavemente entre -1 e 1
        # Isso faz o raio oscilar constantemente (Ex: de 20px a 100px)
        radius = self.base_radius + self.pulse_amplitude * math.sin(t * self.pulse_speed)
        
        # Matemática da Rotação: cresce infinitamente com o tempo
        angle_offset = t * self.rotation_speed

        if pose is not None:
            # 15 = Pulso Esquerdo | 16 = Pulso Direito
            for idx in (15, 16):
                landmark = pose[idx]
                
                # Só desenha se a mão estiver visível e confiável na tela
                if landmark.presence > 0.5 and landmark.visibility > 0.5:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Cores diferentes: Mão esquerda Ciano, Mão direita Dourada
                    color = (255, 255, 0) if idx == 15 else (0, 200, 255)

                    # Desenha 2 triângulos invertidos (formando uma estrela/mandala)
                    self._draw_triangle(out, cx, cy, radius, angle_offset, color)
                    self._draw_triangle(out, cx, cy, radius, angle_offset + math.pi, color)
                    
                    # Desenha um triângulo central menor que gira no sentido oposto e pulsa invertido!
                    inner_radius = self.base_radius - self.pulse_amplitude * math.sin(t * self.pulse_speed)
                    inner_radius = max(5, inner_radius * 0.5) # Garante que não fique negativo
                    self._draw_triangle(out, cx, cy, inner_radius, -angle_offset * 1.5, (255, 255, 255))

                    # Ponto de energia fixo no centro do pulso
                    cv2.circle(out, (cx, cy), 8, (255, 255, 255), -1)

        # Restaura a imagem original nos pixels do usuário (Opcional, mas apara que as runas fiquem só no fundo/frente)
        # cv2.copyTo(frame, mask, out) # Descomente se quiser a pessoa por cima dos triângulos
        
        return out

    def _draw_triangle(self, img, cx, cy, radius, angle_offset, color):
        """Função auxiliar para calcular vértices de um triângulo equilátero e desenhá-lo."""
        pts = []
        # Um círculo tem 2*pi radianos. Dividindo por 3, temos as pontas do triângulo (120 graus)
        for i in range(3):
            angle = angle_offset + i * (2 * math.pi / 3)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            pts.append([int(x), int(y)])
        
        # Converte para o formato exigido pelo OpenCV e desenha as linhas fechadas
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)

    def reset(self):
        pass

class KamehamehaEffect(BaseEffect):
    def __init__(self, charge_rate=3, max_charge=100):
        super().__init__("KAMEHAMEHA")
        self.charge_level = 0
        self.charge_rate = charge_rate
        self.max_charge = max_charge
        self.is_firing = False
        
        self.color_core = (255, 255, 255) 
        self.color_aura = (255, 200, 50)  
        self.hand_centers = [] 
        self.blast_angle = -math.pi / 2 # Padrão: apontado para cima (90 graus negativos)
        
        # --- SETUP DE ÁUDIO ---
        pygame.mixer.init()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        charge_path = os.path.join(base_dir, 'assets', 'audio', 'charge.mp3')
        fire_path = os.path.join(base_dir, 'assets', 'audio', 'haaa.mp3')
        
        self.snd_charge = pygame.mixer.Sound(charge_path) if os.path.exists(charge_path) else None
        self.snd_fire = pygame.mixer.Sound(fire_path) if os.path.exists(fire_path) else None
        
        self.is_playing_charge = False
        self.charge_channel = None

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)
        pose = kwargs.get('pose', None)

        # Lógica de estados e gatilhos
        # A detecção agora engloba tudo em um loop só
        hands_detected = False
        if pose is not None:
            l_hand = pose[19] 
            r_hand = pose[20] 

            if l_hand.presence > 0.5 and r_hand.presence > 0.5:
                hands_detected = True
                lx, ly = int(l_hand.x * w), int(l_hand.y * h)
                rx, ry = int(r_hand.x * w), int(r_hand.y * h)
                
                dist = math.hypot(lx - rx, ly - ry)
                mid_x, mid_y = (lx + rx) // 2, (ly + ry) // 2
                
                self.hand_centers.append((mid_x, mid_y))
                if len(self.hand_centers) > 5:
                    self.hand_centers.pop(0)

                speed = 0
                old_x, old_y = mid_x, mid_y
                if len(self.hand_centers) == 5:
                    old_x, old_y = self.hand_centers[0]
                    speed = math.hypot(mid_x - old_x, mid_y - old_y)

                charge_threshold = w * 0.10 
                fire_speed_threshold = h * 0.05 

                # --- MÁQUINA DE ESTADOS ---
                if not self.is_firing:
                    if dist < charge_threshold:
                        self.charge_level = min(self.charge_level + self.charge_rate, self.max_charge)

                        # GATILHO DE DISPARO
                        if self.charge_level == self.max_charge and speed > fire_speed_threshold:
                            self.is_firing = True
                            
                            # --- LEITURA DIRECCIONAL VIA FLUXO ÓPTICO ---
                            # Recorta uma região de interesse (ROI) ampla ao redor das mãos para ler o vento
                            y1, y2 = max(0, mid_y - 80), min(h, mid_y + 80)
                            x1, x2 = max(0, mid_x - 80), min(w, mid_x + 80)
                            roi_flow = flow[y1:y2, x1:x2]
                            
                            if roi_flow.size > 0:
                                # Tira a média dos vetores de movimento dessa região
                                mean_dx = np.mean(roi_flow[..., 0])
                                mean_dy = np.mean(roi_flow[..., 1])
                                
                                # Misturamos levemente com a cinemática das mãos para blindar contra ruídos da câmera
                                kin_dx = mid_x - old_x
                                kin_dy = mid_y - old_y
                                
                                final_dx = (mean_dx * 0.6) + (kin_dx * 0.4)
                                final_dy = (mean_dy * 0.6) + (kin_dy * 0.4)
                                
                                self.blast_angle = math.atan2(final_dy, final_dx)
                            else:
                                self.blast_angle = -math.pi / 2 # Backup: Cima
                                
                            self.hand_centers.clear()
                            
                            self._stop_charge_audio()
                            if self.snd_fire:
                                self.snd_fire.play()
                    else:
                        self.charge_level = max(self.charge_level - (self.charge_rate * 2), 0)
                else:
                    self.charge_level -= (self.charge_rate * 1.5)
                    if self.charge_level <= 0:
                        self.is_firing = False
                        self.charge_level = 0

        # Tratamento de perdas (Histerese)
        if not hands_detected:
            self.charge_level = max(self.charge_level - (self.charge_rate * 2), 0)
            if self.charge_level <= 0:
                self.is_firing = False
                self.hand_centers.clear()

        # Áudio
        if self.charge_level > 0 and not self.is_firing:
            if self.snd_charge and not self.is_playing_charge:
                self.charge_channel = self.snd_charge.play(loops=-1)
                self.is_playing_charge = True
        elif self.charge_level == 0:
            self._stop_charge_audio()

        # --- RENDERIZAÇÃO ---
        if self.charge_level > 0 and not self.is_firing and hands_detected:
            radius = int((self.charge_level / self.max_charge) * 90)
            noise = random.randint(-8, 8)
            cv2.circle(canvas, (mid_x, mid_y), radius + 20 + noise, self.color_aura, -1)
            cv2.circle(canvas, (mid_x, mid_y), int(radius * 0.6), self.color_core, -1)
        
        elif self.is_firing and hands_detected:
            beam_width = int((self.charge_level / self.max_charge) * 120) + random.randint(10, 30)
            
            # Comprimento infinito garantido (cruza a tela inteira em qualquer ângulo)
            L = max(h, w) * 1.5 
            
            cos_A = math.cos(self.blast_angle)
            sin_A = math.sin(self.blast_angle)
            
            # Vetor Perpendicular (Normal) para a espessura do raio
            nx = -sin_A
            ny = cos_A
            
            # Calcula os 4 pontos do polígono do feixe baseado no ângulo de disparo
            p1 = [mid_x - nx * beam_width, mid_y - ny * beam_width]
            p2 = [mid_x + nx * beam_width, mid_y + ny * beam_width]
            p3 = [mid_x + cos_A * L + nx * (beam_width * 3), mid_y + sin_A * L + ny * (beam_width * 3)]
            p4 = [mid_x + cos_A * L - nx * (beam_width * 3), mid_y + sin_A * L - ny * (beam_width * 3)]
            
            pts = np.array([p1, p2, p3, p4], np.int32)
            
            cv2.fillPoly(canvas, [pts], self.color_aura)
            cv2.fillPoly(canvas, [pts], self.color_core)
            cv2.circle(canvas, (mid_x, mid_y), beam_width, self.color_aura, -1)
            cv2.circle(canvas, (mid_x, mid_y), int(beam_width * 0.6), self.color_core, -1)

        # Luz e Sombras (Bloom & Darkening)
        glow = cv2.GaussianBlur(canvas, (31, 31), 0)
        canvas_bloomed = cv2.addWeighted(canvas, 1.0, glow, 0.9, 0)

        if self.is_firing:
            frame_dark = (frame.astype(np.float32) * 0.4).astype(np.uint8)
            return cv2.add(frame_dark, canvas_bloomed)
        elif self.charge_level > 20:
            darkness = 1.0 - (self.charge_level / self.max_charge) * 0.5
            frame_dark = (frame.astype(np.float32) * darkness).astype(np.uint8)
            return cv2.add(frame_dark, canvas_bloomed)

        return cv2.add(frame, canvas_bloomed)

    def _stop_charge_audio(self):
        if self.charge_channel and self.is_playing_charge:
            self.charge_channel.fadeout(300) 
            self.is_playing_charge = False

    def reset(self):
        self.charge_level = 0
        self.is_firing = False
        self.hand_centers.clear()
        self.blast_angle = -math.pi / 2
        self._stop_charge_audio()

class KamehamehaEffect2(BaseEffect):
    def __init__(self, charge_rate=3, max_charge=100):
        super().__init__("KAMEHAMEHA")
        self.charge_level = 0
        self.charge_rate = charge_rate
        self.max_charge = max_charge
        self.is_firing = False
        
        self.color_core = (255, 255, 255) 
        self.color_aura = (255, 200, 50)  
        
        # Histórico para calcular a velocidade do movimento das mãos
        self.hand_centers = [] 

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)
        pose = kwargs.get('pose', None)

        margin_bottom = int(h * 0.9)

        if pose is not None:
            l_hand = pose[19] 
            r_hand = pose[20] 

            if l_hand.presence > 0.5 and r_hand.presence > 0.5:
                lx, ly = int(l_hand.x * w), int(l_hand.y * h)
                rx, ry = int(r_hand.x * w), int(r_hand.y * h)

                lx = np.clip(lx, 0, w - 1)
                ly = np.clip(ly, 0, h - 1)
                rx = np.clip(rx, 0, w - 1)
                ry = np.clip(ry, 0, h - 1)

                valid_l = ly < margin_bottom and mask[ly, lx] > 127
                valid_r = ry < margin_bottom and mask[ry, rx] > 127

                if valid_l and valid_r:
                
                    dist = math.hypot(lx - rx, ly - ry)
                    mid_x, mid_y = (lx + rx) // 2, (ly + ry) // 2
                    
                    # 1. Rastreia o centro das mãos em uma janela de 5 frames
                    self.hand_centers.append((mid_x, mid_y))
                    if len(self.hand_centers) > 5:
                        self.hand_centers.pop(0)

                    # 2. Calcula a velocidade das mãos (Distância entre o frame atual e o de 5 frames atrás)
                    speed = 0
                    if len(self.hand_centers) == 5:
                        old_x, old_y = self.hand_centers[0]
                        speed = math.hypot(mid_x - old_x, mid_y - old_y)

                    # Limiares de Ação
                    charge_threshold = w * 0.25  # Bem mais flexível para carregar (30% da tela)
                    fire_speed_threshold = h * 0.08  # Precisa mover as mãos rápido (8% da altura da tela de uma vez)

                    # --- MÁQUINA DE ESTADOS ---
                    if not self.is_firing:
                        if dist < charge_threshold:
                            # CARREGANDO
                            self.charge_level = min(self.charge_level + self.charge_rate, self.max_charge)
                            
                            # GATILHO DE DISPARO: Carga cheia + Movimento brusco das mãos juntas
                            if self.charge_level == self.max_charge and speed > fire_speed_threshold:
                                self.is_firing = True
                                self.hand_centers.clear() # Limpa o histórico para evitar tiros duplos
                        else:
                            # Se afastar as mãos, a energia se desfaz devagar
                            self.charge_level = max(self.charge_level - (self.charge_rate * 2), 0)
                    else:
                        # ATIRANDO (Drena a carga)
                        self.charge_level -= (self.charge_rate * 1.5)
                        if self.charge_level <= 0:
                            self.is_firing = False
                            self.charge_level = 0

                    # --- RENDERIZAÇÃO ---
                    if self.charge_level > 0 and not self.is_firing:
                        # Esfera carregando
                        radius = int((self.charge_level / self.max_charge) * 90)
                        noise = random.randint(-8, 8)
                        cv2.circle(canvas, (mid_x, mid_y), radius + 20 + noise, self.color_aura, -1)
                        cv2.circle(canvas, (mid_x, mid_y), int(radius * 0.6), self.color_core, -1)
                    
                    elif self.is_firing:
                        # Feixe sustentado
                        beam_width = int((self.charge_level / self.max_charge) * 120) + random.randint(10, 30)
                        
                        pts = np.array([
                            [mid_x - beam_width, mid_y],
                            [mid_x + beam_width, mid_y],
                            [mid_x + beam_width*3, 0],
                            [mid_x - beam_width*3, 0]
                        ], np.int32)
                        
                        cv2.fillPoly(canvas, [pts], self.color_aura)
                        cv2.fillPoly(canvas, [pts], self.color_core)
                        
                        cv2.circle(canvas, (mid_x, mid_y), beam_width, self.color_aura, -1)
                        cv2.circle(canvas, (mid_x, mid_y), int(beam_width * 0.6), self.color_core, -1)

            else:
                # Perdeu as mãos da tela
                self.charge_level = max(self.charge_level - 10, 0)
                self.is_firing = False
                self.hand_centers.clear()

        # Efeitos de luz
        glow = cv2.GaussianBlur(canvas, (31, 31), 0)
        canvas_bloomed = cv2.addWeighted(canvas, 1.0, glow, 0.9, 0)

        if self.is_firing:
            frame_dark = (frame.astype(np.float32) * 0.4).astype(np.uint8)
            return cv2.add(frame_dark, canvas_bloomed)
        elif self.charge_level > 20:
            darkness = 1.0 - (self.charge_level / self.max_charge) * 0.5
            frame_dark = (frame.astype(np.float32) * darkness).astype(np.uint8)
            return cv2.add(frame_dark, canvas_bloomed)

        return cv2.add(frame, canvas_bloomed)

    def reset(self):
        self.charge_level = 0
        self.is_firing = False
        self.hand_centers.clear()

        