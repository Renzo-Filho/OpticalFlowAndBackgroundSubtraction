import cv2
import numpy as np
from .baseEffect import BaseEffect

class CyberGlitchEffect(BaseEffect):
    def __init__(self, intensity_multiplier=1.5):
        """
        Efeito Cyberpunk/Glitch. O canal Vermelho e Azul se separam 
        horizontalmente baseados na velocidade do movimento do usuário.
        """
        super().__init__("CYBER_GLITCH")
        self.intensity = intensity_multiplier

    def apply(self, frame, flow, mask):
        h, w = frame.shape[:2]
        out = frame.copy()

        # Calcula a magnitude do movimento apenas na área da pessoa (Foreground)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active_flow = mag[mask > 127]
        mean_motion = np.mean(active_flow) if len(active_flow) > 0 else 0

        # Só aplica o glitch se houver movimento relevante
        if mean_motion > 0.5:
            # Limita o deslocamento máximo para não quebrar a imagem
            shift = int(min(mean_motion * self.intensity, 60))

            # Separa os canais de cor (BGR)
            B, G, R = cv2.split(frame)

            # Desloca o canal Vermelho para a direita
            M_R = np.float32([[1, 0, shift], [0, 1, 0]])
            R_shifted = cv2.warpAffine(R, M_R, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Desloca o canal Azul para a esquerda
            M_B = np.float32([[1, 0, -shift], [0, 1, 0]])
            B_shifted = cv2.warpAffine(B, M_B, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Mescla de volta
            glitched_frame = cv2.merge([B_shifted, G, R_shifted])

            # Cria linhas de falha aleatórias (Scanlines) se o movimento for muito forte
            if mean_motion > 3.0:
                num_lines = np.random.randint(2, 8)
                for _ in range(num_lines):
                    y = np.random.randint(0, h)
                    thickness = np.random.randint(1, 8)
                    # Inverte a cor na linha gerada para dar um aspecto digital quebrado
                    glitched_frame[y:y+thickness, :] = 255 - glitched_frame[y:y+thickness, :]

            # Aplica o efeito APENAS na pessoa, mantendo o fundo normal
            mask_3 = (mask.astype(np.float32) / 255.0)[..., None]
            out_float = (glitched_frame.astype(np.float32) * mask_3) + (frame.astype(np.float32) * (1.0 - mask_3))
            out = np.clip(out_float, 0, 255).astype(np.uint8)

        return out

    def reset(self):
        pass


class NeonSilhouetteEffect(BaseEffect):
    def __init__(self, color=(255, 255, 0)): 
        """
        Extrai o contorno da máscara e cria um avatar de neon (Padrão: Ciano).
        O fundo fica completamente preto.
        """
        super().__init__("NEON_SILHOUETTE")
        self.color = color # Formato BGR
        self.kernel = np.ones((5, 5), np.uint8)

    def apply(self, frame, flow, mask):
        # Encontra as bordas usando gradiente morfológico (Diferença entre dilatação e erosão)
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self.kernel)

        # Cria uma tela preta
        canvas = np.zeros_like(frame)

        # Pinta as bordas com a cor de neon escolhida
        canvas[edges > 0] = self.color

        # Aplica desfoques gaussianos sucessivos para criar o efeito de brilho (Bloom)
        glow1 = cv2.GaussianBlur(canvas, (11, 11), 0)
        glow2 = cv2.GaussianBlur(canvas, (25, 25), 0)

        # Soma as camadas: Contorno sólido + Brilho 1 + Brilho 2
        out = cv2.addWeighted(canvas, 1.0, glow1, 0.8, 0)
        out = cv2.addWeighted(out, 1.0, glow2, 0.5, 0)

        return out

    def reset(self):
        pass
