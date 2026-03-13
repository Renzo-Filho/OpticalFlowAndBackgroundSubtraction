import cv2
import numpy as np
from .baseEffect import BaseEffect

class ArrowEffect(BaseEffect):
    def __init__(self, step=30, threshold=2.0, color=(0, 255, 0)):
        super().__init__("ARROWS")
        self.step = step
        self.threshold = threshold
        self.color = color

    def apply(self, frame, flow, mask=None):
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

    def apply(self, frame, flow, mask=None):
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