import cv2
import numpy as np
from .baseEffect import BaseEffect

class ShowMaskEffect(BaseEffect):
    def __init__(self):
        """
        Visualizes the binary foreground mask.
        Useful for academic comparison between Static and Flow methods.
        """
        super().__init__("SHOW_MASK")

    def apply(self, frame, flow, mask, **kwargs):
        """
        Converts the 1-channel binary mask into a 3-channel BGR image
        so it can be displayed in the main window.
        """
        if mask is None:
            # If no mask exists yet (e.g., no background captured)
            h, w = frame.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert the 0-255 grayscale mask to BGR for consistent display
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def reset(self):
        """No persistent state to reset."""
        pass