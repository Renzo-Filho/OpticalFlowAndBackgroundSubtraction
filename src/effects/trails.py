import cv2
import numpy as np
from .baseEffect import BaseEffect

class GhostTrailEffect(BaseEffect):
    def __init__(self, alpha=0.90):
        super().__init__("GHOST_TRAILS")
        self.alpha = alpha
        self.acc = None

    def apply(self, frame, flow, mask, **kwargs):
        if self.acc is None: self.acc = frame.astype(np.float32)
        
        # The logic from your script
        frame_f = frame.astype(np.float32)
        self.acc = cv2.addWeighted(self.acc, self.alpha, frame_f, (1.0 - self.alpha), 0)
        
        # Apply mask so trails only appear on the person/motion
        mask_3 = (mask.astype(np.float32) / 255.0)[..., None]
        out = (self.acc * mask_3) + (frame_f * (1.0 - mask_3))
        return np.clip(out, 0, 255).astype(np.uint8)


class MotionTrailEffect(BaseEffect):
    def __init__(self, trail_length=0.92, thick_mode=False):
        """
        Creates a 'ghosting' trail based on movement.
        :param trail_length: 0.0 to 1.0 (Higher = longer, more persistent trails).
        :param thick_mode: If True, uses dilation for a bolder, solid rastro.
        """
        super().__init__("MOTION_TRAIL")
        self.trail_length = trail_length
        self.thick_mode = thick_mode
        self.acc = None

    def apply(self, frame, flow, mask, **kwargs):
        """
        Blends the current frame with the history (accumulator) 
        specifically in moving areas defined by the mask.
        """
        h, w = frame.shape[:2]
        frame_f = frame.astype(np.float32)

        # 1. Initialize or resize accumulator
        if self.acc is None or self.acc.shape[:2] != (h, w):
            self.acc = frame_f.copy()

        # 2. Refine the Mask (Internal Morphology)
        # We use a copy of the mask to avoid affecting other effects in the loop
        processed_mask = mask.astype(np.float32) / 255.0
        
        if self.thick_mode:
            # Replicates your 'effect_long_trail' logic
            kernel = np.ones((15, 15), np.uint8)
            processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
            processed_mask = cv2.GaussianBlur(processed_mask, (5, 5), 0)

        # Ensure mask is 3-channel for math
        mask_3 = processed_mask[..., None]

        # 3. Blending Logic (The 'Ghosting' math)
        # acc_next = (current * mask) + (history * decay * (1 - mask))
        # This keeps the background sharp but lets the person leave a trail
        trail_blend = cv2.addWeighted(self.acc, self.trail_length, frame_f, (1.0 - self.trail_length), 0)
        
        self.acc = (trail_blend * mask_3) + (frame_f * (1.0 - mask_3))

        return np.clip(self.acc, 0, 255).astype(np.uint8)

    def reset(self):
        """Clears the history so the next user starts fresh."""
        self.acc = None