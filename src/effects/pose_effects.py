import cv2
import numpy as np
import random
from .baseEffect import BaseEffect

class FlowBenderEffect(BaseEffect):
    def __init__(self, num_particles=500): # Aumentei um pouco o limite
        super().__init__("FLOW_BENDER")
        # Estado: [x, y, vx, vy, life, max_life] <-- Adicionamos max_life para o cálculo de escala
        self.particles = np.zeros((num_particles, 6), dtype=np.float32) 

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        canvas = np.zeros_like(frame)

        # Resgata o pose do kwargs (já que agora usamos **kwargs em tudo)
        pose = kwargs.get('pose', None)

        # 1. ENCONTRA AS MÃOS USANDO MEDIA PIPE POSE
        hands_coords = []
        if pose is not None:
            # Pontos 15 e 16 são os pulsos. (Opcional: 19 e 20 são as pontas dos indicadores)
            for idx in (18,17,19,20):#(15, 16): 
                landmark = pose[idx]
                if landmark.presence > 0.5: 
                    hx, hy = int(landmark.x * w), int(landmark.y * h)
                    hands_coords.append((hx, hy))

        # 2. GERA PARTÍCULAS "GORDAS" NA ÁREA DAS MÃOS
        if hands_coords:
            dead_idx = np.where(self.particles[:, 4] <= 0)[0]
            spawn_count = min(len(dead_idx), 25) # Quantidade de emissão por frame
            
            for i in range(spawn_count):
                idx = dead_idx[i]
                hand_x, hand_y = random.choice(hands_coords)
                
                # Espalhamento maior para preencher a mão inteira (raio de ~30 pixels)
                self.particles[idx, 0] = hand_x + random.uniform(-30, 30)
                self.particles[idx, 1] = hand_y + random.uniform(-30, 30)
                
                # Explosão inicial leve para os lados
                self.particles[idx, 2] = random.uniform(-3, 3)
                self.particles[idx, 3] = random.uniform(-3, 3)
                
                # Vida variável
                life = random.uniform(15, 45)
                self.particles[idx, 4] = life
                self.particles[idx, 5] = life # Salva a vida inicial máxima

        # 3. MOVE AS PARTÍCULAS USANDO O OPTICAL FLOW
        active = self.particles[:, 4] > 0
        if np.any(active):
            px = np.clip(self.particles[active, 0].astype(int), 0, w - 1)
            py = np.clip(self.particles[active, 1].astype(int), 0, h - 1)

            # O Vento: Lê a direção do Optical Flow
            flow_vectors = flow[py, px]
            
            # Adiciona a força do Flow e aplica atrito (0.85 para ser mais fluido)
            self.particles[active, 2] = (self.particles[active, 2] * 0.85) + (flow_vectors[:, 0] * 1.5)
            self.particles[active, 3] = (self.particles[active, 3] * 0.85) + (flow_vectors[:, 1] * 1.5)

            # Move e envelhece
            self.particles[active, 0] += self.particles[active, 2]
            self.particles[active, 1] += self.particles[active, 3] - 2.5 # Sobe um pouco mais rápido como fogo
            self.particles[active, 4] -= 1

            # 4. RENDERIZAÇÃO GORDA E EM CAMADAS
            render_data = self.particles[active]
            
            for pt_data in render_data:
                x, y = int(pt_data[0]), int(pt_data[1])
                life, max_life = pt_data[4], pt_data[5]
                
                if 0 <= x < w and 0 <= y < h:
                    # Calcula o tamanho (Nasce grande, morre pequena)
                    scale = life / max_life
                    # Raio base enorme (até 35 pixels)
                    radius = int(35 * scale) 
                    
                    if radius > 0:
                        # Camada 1: Aura expansiva (Vermelho/Laranja escuro)
                        cv2.circle(canvas, (x, y), radius, (0, 70, 200), -1)
                        # Camada 2: Miolo de energia (Amarelo/Branco brilhante)
                        cv2.circle(canvas, (x, y), int(radius * 0.4), (150, 255, 255), -1)

        # 5. EFEITO DE BLOOM (Brilho suave ao redor das partículas combinadas)
        # O desfoque funde as esferas, parecendo um fluido de energia contínuo
        canvas = cv2.GaussianBlur(canvas, (15, 15), 0)

        return cv2.add(frame, canvas)

    def reset(self):
        self.particles[:, 4] = 0

import cv2
import numpy as np
from .baseEffect import BaseEffect

class NeonSkeletonEffect(BaseEffect):
    """
    Rastreia a estrutura óssea do usuário e desenha um esqueleto de neon.
    Usa um acumulador temporal para deixar um rastro fotográfico (Light Painting).
    """
    def __init__(self, color=(255, 50, 255), decay=0.85): # Rosa Neon por padrão
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
