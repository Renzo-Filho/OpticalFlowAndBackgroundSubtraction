from abc import ABC, abstractmethod
import numpy as np

class BaseEffect(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, frame, flow, mask, pose=None):
        """
        Each effect must implement this.
        Returns: the processed BGR image.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Clears internal buffers/canvases.
        """
        pass