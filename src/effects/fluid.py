import cv2
import numpy as np
from .baseEffect import BaseEffect

class FluidPaintEffect(BaseEffect):
    def __init__(self, decay=0.985, advect_gain=2.0):
        super().__init__("FLUID_PAINT_BG")
        self.decay = decay
        self.advect_gain = advect_gain
        self.canvas = None

    def apply(self, frame, flow, mask):
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