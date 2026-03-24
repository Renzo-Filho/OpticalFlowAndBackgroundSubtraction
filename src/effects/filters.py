import cv2
import numpy as np
from .baseEffect import BaseEffect

class NegativeEffect(BaseEffect):
    def __init__(self):
        super().__init__("NEGATIVE_FILTER")

    def apply(self, frame, flow, mask, pose_results=None, **kwargs):
        # Simply invert the bits: 255 - pixel_value
        return cv2.bitwise_not(frame)

    def reset(self):
        pass

class CartoonEffect(BaseEffect):
    def __init__(self):
        super().__init__("CARTOON_FILTER")

    def apply(self, frame, flow, mask, pose_results=None, **kwargs):
        # 1. Reduce noise while preserving edges
        color = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 2. Detect edges and thicken them
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 2)
        
        # 3. Combine color and edges
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_bgr)

    def reset(self):
        pass

class HeatmapEffect(BaseEffect):
    def __init__(self):
        super().__init__("COLORFUL_HEAT")

    def apply(self, frame, flow, mask, pose_results=None, **kwargs):
        # Convert to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a 'COLORMAP' for a scientific heatmap look
        # Try cv2.COLORMAP_JET, COLORMAP_HOT, or COLORMAP_RAINBOW
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET  )

    def reset(self):
        pass