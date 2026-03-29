import cv2
import numpy as np
from .baseEffect import BaseEffect

class ArrowEffect(BaseEffect):
    def __init__(self, step=30, threshold=2.0, color=(0, 255, 0)):
        super().__init__("ARROWS")
        self.step = step
        self.threshold = threshold
        self.color = color

    def apply(self, frame, flow, mask=None, **kwargs):
        out = frame.copy()
        h, w = frame.shape[:2]

        # Filtra o ruído do fundo apenas para as setas
        if mask is not None:
            flow_filtered = flow.copy()
            flow_filtered[mask == 0] = 0
        else:
            flow_filtered = flow

        for y in range(self.step // 2, h, self.step):
            for x in range(self.step // 2, w, self.step):
                fx, fy = flow_filtered[y, x]

                # Ignore noise
                if (fx**2 + fy**2) < (self.threshold**2):
                    continue

                cv2.arrowedLine(
                    out, (x, y), (int(x + fx * 4), int(y + fy * 4)),
                    self.color, 1, tipLength=0.3
                )
        return out

    def reset(self):
        pass


class GridWarpEffect(BaseEffect):
    def __init__(self, step=40, amplitude=3.0, color=(0, 255, 255)):
        """
        Deforms a virtual wireframe grid based on optical flow.
        :param step: Distance between grid lines (density).
        :param amplitude: How much the motion 'pulls' the grid.
        :param color: BGR color of the grid lines.
        """
        super().__init__("GRID_WARP")
        self.step = step
        self.amplitude = amplitude
        self.color = color

    def apply(self, frame, flow, mask=None, **kwargs):
        """
        Draws the warped grid. Note: It does not use the mask, 
        as the effect covers the whole screen.
        """
        h, w = frame.shape[:2]
        out = np.zeros_like(frame)

        # A MÁGICA AQUI: Filtra o ruído do fundo apenas para o grid
        if mask is not None:
            flow_filtered = flow.copy()
            flow_filtered[mask == 0] = 0
        else:
            flow_filtered = flow

        # 1. Draw Vertical Lines (warped by flow)
        for x in range(0, w, self.step):
            pts = []
            for y in range(0, h, 10): 
                dx, dy = flow_filtered[y, min(x, w-1)]
                pts.append([x + dx * self.amplitude, y + dy * self.amplitude])
            
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts_arr], False, self.color, 1, cv2.LINE_AA)

        # 2. Draw Horizontal Lines (warped by flow)
        for y in range(0, h, self.step):
            pts = []
            for x in range(0, w, 10):
                dx, dy = flow_filtered[min(y, h-1), x]
                pts.append([x + dx * self.amplitude, y + dy * self.amplitude])
            
            pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts_arr], False, self.color, 1, cv2.LINE_AA)

        return out

    def reset(self):
        pass

class DelaunayConstellationEffect(BaseEffect):
    """
    Constrói uma malha geométrica viva conectada ao corpo com persistência temporal.
    Referência Matemática: Triangulação de Delaunay
    """
    def __init__(self, max_points=300, line_color=(0, 255, 255), decay=0.85):
        super().__init__("DELAUNAY_CONSTELLATION")
        self.max_points = max_points
        self.color = line_color
        self.decay = decay
        self.canvas = None

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]
        
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros_like(frame)

        # Apaga o canvas lentamente para criar o rastro do movimento
        self.canvas = (self.canvas.astype(np.float32) * self.decay).astype(np.uint8)

        # minDistance menor e maxCorners maior geram uma malha muito mais densa e rica
        corners = cv2.goodFeaturesToTrack(mask, maxCorners=self.max_points, qualityLevel=0.005, minDistance=10)

        if corners is not None and len(corners) > 3:
            rect = (0, 0, w, h)
            subdiv = cv2.Subdiv2D(rect)

            for p in corners:
                x, y = p[0]
                if 0 <= x < w and 0 <= y < h:
                    subdiv.insert((int(x), int(y)))

            triangles = subdiv.getTriangleList()

            for t in triangles:
                pt1 = (int(t[0]), int(t[1]))
                pt2 = (int(t[2]), int(t[3]))
                pt3 = (int(t[4]), int(t[5]))

                if self._is_valid(pt1, pt2, pt3, w, h, mask):
                    cv2.line(self.canvas, pt1, pt2, self.color, 1, cv2.LINE_AA)
                    cv2.line(self.canvas, pt2, pt3, self.color, 1, cv2.LINE_AA)
                    cv2.line(self.canvas, pt3, pt1, self.color, 1, cv2.LINE_AA)
                    cv2.circle(self.canvas, pt1, 2, (255, 255, 255), -1)

        # Escurece um pouco o frame original para a malha neon se destacar mais
        frame_dark = (frame.astype(np.float32) * 0.9).astype(np.uint8)
        out = cv2.add(frame_dark, self.canvas)

        return out

    def _is_valid(self, p1, p2, p3, w, h, mask):
        for x, y in [p1, p2, p3]:
            if not (0 <= x < w and 0 <= y < h): return False
            if mask[y, x] == 0: return False
        return True

    def reset(self):
        self.canvas = None

import cv2
import numpy as np
from .baseEffect import BaseEffect

class ShatteredGlassEffect(BaseEffect):
    """
    Simula uma tela de vidro estilhaçada. 
    Usa o Diagrama de Voronoi para criar os cacos e aplica refração e espelhamento aleatórios em cada um.
    """
    def __init__(self, num_shards=18, max_offset=20):
        super().__init__("SHATTERED_GLASS")
        self.num_shards = num_shards
        self.max_offset = max_offset
        self.map_x = None
        self.map_y = None
        self.cracks_canvas = None

    def _generate_shards(self, h, w):
        """Gera os cacos de vidro e os cálculos de distorção apenas uma vez para economizar CPU."""
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        self.map_x = grid_x.astype(np.float32)
        self.map_y = grid_y.astype(np.float32)

        # 1. Sorteia os "centros" de cada caco de vidro
        pts = np.random.randint(0, [w, h], size=(self.num_shards, 2))

        # 2. Calcula as regiões de Voronoi (Qual pixel pertence a qual caco)
        min_dist = np.full((h, w), np.inf, dtype=np.float32)
        regions = np.zeros((h, w), dtype=np.uint8)

        for i, (px, py) in enumerate(pts):
            dist = (self.map_x - px)**2 + (self.map_y - py)**2
            mask = dist < min_dist
            min_dist[mask] = dist[mask]
            regions[mask] = i

        # 3. Aplica espelhamento e deslocamento DENTRO de cada caco
        for i, (px, py) in enumerate(pts):
            shard_mask = (regions == i)

            # Refração: deslocamento aleatório
            dx = np.random.uniform(-self.max_offset, self.max_offset)
            dy = np.random.uniform(-self.max_offset, self.max_offset)

            # Espelhamento: 35% de chance de espelhar horizontal ou verticalmente
            flip_x = -1 if np.random.random() > 0.65 else 1
            flip_y = -1 if np.random.random() > 0.65 else 1

            self.map_x[shard_mask] = px + (self.map_x[shard_mask] - px) * flip_x + dx
            self.map_y[shard_mask] = py + (self.map_y[shard_mask] - py) * flip_y + dy

        # 4. Desenha as "rachaduras" do vidro usando gradiente morfológico
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(regions, cv2.MORPH_GRADIENT, kernel)
        
        # Pinta as bordas das rachaduras de branco/cinza claro
        cracks_mask = np.where(edges > 0, 200, 0).astype(np.uint8)
        self.cracks_canvas = cv2.cvtColor(cracks_mask, cv2.COLOR_GRAY2BGR)

    def apply(self, frame, flow, mask, **kwargs):
        h, w = frame.shape[:2]

        # Gera o vidro quebrado no primeiro frame ou se o usuário apertar 'r' (reset)
        if self.map_x is None or self.map_x.shape[:2] != (h, w):
            self._generate_shards(h, w)

        # Destaca a pessoa escurecendo o fundo (para manter o foco no usuário)
        bg_mask = cv2.bitwise_not(mask)
        frame_darkened = frame.copy()
        frame_darkened[bg_mask == 255] = (frame_darkened[bg_mask == 255] * 0.3).astype(np.uint8)

        # Quebra a imagem de acordo com o mapa de distorção gerado
        shattered_frame = cv2.remap(
            frame_darkened, 
            self.map_x, 
            self.map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
        )

        # Sobrepõe as rachaduras brilhantes por cima da imagem quebrada
        result = cv2.add(shattered_frame, self.cracks_canvas)
        return result

    def reset(self):
        # Limpar os mapas faz com que o vidro "quebre" em um padrão totalmente novo!
        self.map_x = None
        self.map_y = None
        self.cracks_canvas = None