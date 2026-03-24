import cv2
import numpy as np
from collections import deque
from .baseEffect import BaseEffect

class SolidCloneEffect(BaseEffect):
    def __init__(self, max_clones=5, frame_delay=10):
        """
        :param max_clones: Number of solid duplicates to show.
        :param frame_delay: Number of frames between each clone (controls 'distance' in time).
        """
        super().__init__("SOLID_CLONES")
        self.max_clones = max_clones
        self.frame_delay = frame_delay
        
        # Buffer to store (frame, mask) tuples
        self.buffer = deque(maxlen=max_clones * frame_delay + 1)

    def apply(self, frame, flow, mask, pose_results=None, **kwargs):
        # 1. Store current state in history
        # We store copies to prevent issues as the main loop updates frames
        self.buffer.append((frame.copy(), mask.copy()))
        
        # Start with a copy of the current frame as the base (the background)
        output = frame.copy()

        # 2. Iterate through history to draw clones
        # We skip the very last frame (current) and look back at specific intervals
        for i in range(1, self.max_clones + 1):
            idx = -(i * self.frame_delay)
            
            # Check if we have enough history to reach this clone
            if abs(idx) < len(self.buffer):
                past_frame, past_mask = self.buffer[idx]
                
                # Use the past mask to "cut" the past person out
                # and paste them onto the current output
                # We use the mask as a target so the clone is SOLID
                cv2.copyTo(past_frame, past_mask, output)

        # 3. Final Layer: Draw the current person on top of all clones
        # This ensures the 'real' person is never covered by a duplicate
        cv2.copyTo(frame, mask, output)
        
        return output

    def reset(self):
        self.buffer.clear()

