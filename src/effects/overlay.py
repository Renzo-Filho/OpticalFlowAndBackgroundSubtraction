import cv2
import numpy as np
import io
import os
from .baseEffect import BaseEffect

class ExhibitionSlideEffect(BaseEffect):
    """
    Exibe um slide estático sobrepondo toda a tela.
    Ideal para introduções ou respiros durante a exposição.
    """
    def __init__(self, image_filename="slide.png"):
        super().__init__("PROJETO")
        
        # Constrói o caminho absoluto para a pasta assets
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        image_path = os.path.join(base_dir, 'assets/icons', image_filename)
        
        self.slide = cv2.imread(image_path)
        if self.slide is None:
            print(f"[AVISO] Slide não encontrado em: {image_path}")

    def apply(self, frame, flow, mask, **kwargs):
        # Se a imagem não for encontrada, retorna a câmera normal
        if self.slide is None:
            return frame
        
        #h, w = frame.shape[:2]
        
        # Redimensiona o slide para cobrir toda a tela perfeitamente
        #slide_resized = cv2.resize(self.slide, (w, h), interpolation=cv2.INTER_AREA)
        #return slide_resized
        return self.slide.copy()

    def reset(self):
        pass

class MathChromaKeyEffect(BaseEffect):
    """
    Renderiza Fórmulas LaTeX em alta qualidade e cria um Chroma Key animado em ziguezague.
    """
    def __init__(self):
        super().__init__("MATH_CHROMA_KEY")
        self.time_counter = 0
        self.sprites = []
        
        self._generate_latex_sprites()

    def _generate_latex_sprites(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[ERRO] Instale o matplotlib (pip install matplotlib) para renderizar LaTeX.")
            return

        formulas = [
            r"\nabla I \cdot \mathbf{v} + \frac{\partial I}{\partial t} = 0",                     
            r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t}", 
            r"\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = \nu \nabla^2 \mathbf{v}", 
            r"C_{out} = \alpha C_{src} + (1 - \alpha) C_{dst}",                                   
            r"V - E + F = 2",                                                                     
            r"\mathbf{p}(t) = \mathbf{p}_0 + \mathbf{v}_0 t + \frac{1}{2}\mathbf{a}t^2"           
        ]

        fig = plt.figure(figsize=(8, 2), dpi=120) 
        
        print("[INFO] Renderizando e recortando equações matemáticas...")
        for tex in formulas:
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

                alpha_channel = sprite[:, :, 3] # Pega o canal de transparência
                y_idx, x_idx = np.where(alpha_channel > 0) # Acha onde tem tinta
                
                if len(y_idx) > 0 and len(x_idx) > 0:
                    y_min, y_max = np.min(y_idx), np.max(y_idx)
                    x_min, x_max = np.min(x_idx), np.max(x_idx)
                    
                    # Corta a imagem no limite exato do texto
                    sprite = sprite[y_min:y_max+1, x_min:x_max+1]
                
                self.sprites.append(sprite)
            else:
                print(f"[AVISO] Falha ao decodificar a equação: {tex}")
        
        plt.close(fig)
        print("[INFO] Matriz de Equações pronta e otimizada!")

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        
        background = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.sprites:
            self.time_counter += 1
            
            speed = 4.0 
            y_spacing = max(h // 2, 200) 
            total_scroll_height = max(h + 300, len(self.sprites) * y_spacing)

            for i, sprite in enumerate(self.sprites):
                sh, sw = sprite.shape[:2]
                
                margin = int(w * 0.05) # Margem dinâmica: sempre 5% da largura da tela

                pos_type = i % 3
                if pos_type == 0:
                    x_pos = margin                   # Esquerda com margem de 5%
                elif pos_type == 1:
                    x_pos = (w - sw) // 2            # Exatamente no centro
                else:
                    x_pos = w - sw - margin          # Direita com margem de 5%
                
                # Trava de segurança
                x_pos = min(max(int(x_pos), 0), w - sw)
                progress = (self.time_counter * speed + i * y_spacing) % total_scroll_height
                y_pos = (h + 50) - progress
                
                self._overlay_transparent(background, sprite, x_pos, int(y_pos))
                
        else:
            cv2.putText(background, "Erro: Equacoes ausentes", (50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # CHROMA KEY CORE
        mask_float = mask.astype(np.float32) / 255.0
        mask_3d = mask_float[..., None]
        inv_mask_3d = 1.0 - mask_3d

        foreground = frame.astype(np.float32) * mask_3d
        bg_filtered = background.astype(np.float32) * inv_mask_3d

        output = cv2.add(foreground, bg_filtered)
        return np.clip(output, 0, 255).astype(np.uint8)

    def _overlay_transparent(self, background, overlay, x, y):
        h, w = background.shape[:2]
        sh, sw = overlay.shape[:2]

        y1, y2 = max(0, y), min(h, y + sh)
        x1, x2 = max(0, x), min(w, x + sw)

        oy1, oy2 = max(0, -y), sh - max(0, (y + sh) - h)
        ox1, ox2 = max(0, -x), sw - max(0, (x + sw) - w)

        if y1 >= y2 or x1 >= x2:
            return

        alpha = overlay[oy1:oy2, ox1:ox2, 3:4] / 255.0

        background[y1:y2, x1:x2] = (alpha * overlay[oy1:oy2, ox1:ox2, :3] + 
                                    (1.0 - alpha) * background[y1:y2, x1:x2])

    def reset(self):
        self.time_counter = 0