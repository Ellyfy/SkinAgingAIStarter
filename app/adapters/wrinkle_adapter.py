import numpy as np
from .base import WrinkleSegmentationAdapter
from app.scoring import wrinkle_proxy_mask

class BaselineWrinkleAdapter(WrinkleSegmentationAdapter):
    def predict_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        return wrinkle_proxy_mask(face_bgr)

class LearnedWrinkleAdapter(WrinkleSegmentationAdapter):
    """
    Planned practical model: rmsandu/FFHQ-detect-face-wrinkles.
    Research comparison: labhai/ffhq-wrinkle-dataset.
    """
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def predict_mask(self, face_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Load the trained U-Net here in Phase 2.")
