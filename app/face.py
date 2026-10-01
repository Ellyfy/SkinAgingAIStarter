import cv2
import numpy as np
from typing import Optional, Tuple

class HaarFaceDetector:
    """Lightweight Phase-1 detector; later replaced with landmark alignment."""

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(path)

    def detect(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
        )
        if len(faces) == 0:
            return None
        return tuple(max(faces, key=lambda b: int(b[2]) * int(b[3])))
