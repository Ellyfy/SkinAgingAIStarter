import numpy as np
from .base import ExpressionAdapter

class SmirkAdapter(ExpressionAdapter):
    def __init__(self, enabled=False):
        self.enabled = enabled

    def analyze(self, face_bgr: np.ndarray) -> dict:
        if not self.enabled:
            return {"enabled": False, "status": "SMIRK adapter not installed yet."}
        raise NotImplementedError("Connect SMIRK encoders in Phase 2.")
