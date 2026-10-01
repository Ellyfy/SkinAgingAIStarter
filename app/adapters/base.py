from abc import ABC, abstractmethod
import numpy as np

class ExpressionAdapter(ABC):
    @abstractmethod
    def analyze(self, face_bgr: np.ndarray) -> dict:
        raise NotImplementedError

class FaceParsingAdapter(ABC):
    @abstractmethod
    def parse(self, face_bgr: np.ndarray) -> dict:
        raise NotImplementedError

class WrinkleSegmentationAdapter(ABC):
    @abstractmethod
    def predict_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError
