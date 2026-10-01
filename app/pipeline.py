import numpy as np

from .quality import QualityChecker
from .face import HaarFaceDetector
from .regions import approximate_regions
from .scoring import score_regions, overall_score

from .adapters.wrinkle_adapter import BaselineWrinkleAdapter
from .adapters.smirk_adapter import SmirkAdapter
from .adapters.parsing_adapter import BiSeNetParsingAdapter


class SkinAgingPipeline:
    def __init__(self):
        self.quality = QualityChecker()
        self.face_detector = HaarFaceDetector()
        self.wrinkle_model = BaselineWrinkleAdapter()

        self.smirk = SmirkAdapter(enabled=False)

        # BiSeNet is now enabled
        self.face_parser = BiSeNetParsingAdapter(enabled=True)

    def analyze(self, image_bgr: np.ndarray) -> dict:
        # Check image quality
        quality = self.quality.check(image_bgr)

        # Detect face
        face_box = self.face_detector.detect(image_bgr)

        if face_box is None:
            return {
                "quality": quality,
                "face": None,
                "face_parsing": None,
                "overall_wrinkle_proxy": 0.0,
                "regions": {},
                "notes": [
                    "No frontal face was detected.",
                    "Phase 1 currently uses a simple frontal-face detector.",
                ],
            }

        # Crop detected face
        x, y, w, h = face_box
        face = image_bgr[y:y + h, x:x + w]

        # Run BiSeNet face parsing
        parsing_result = self.face_parser.parse(face)

        # Existing Phase-1 wrinkle analysis
        regions = approximate_regions(face_box)
        wrinkle_mask = self.wrinkle_model.predict_mask(face)
        scores = score_regions(wrinkle_mask, regions)

        return {
            "quality": quality,

            "face": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },

            "face_parsing": {
                "status": parsing_result["status"],
                "mask_shape": parsing_result.get("mask_shape"),
                "class_names_found": parsing_result.get("class_names_found"),
            },

            "overall_wrinkle_proxy": overall_score(scores),

            "regions": scores,

            "notes": [
                "Phase-1 proxy only; not a clinical measurement.",
                self.smirk.analyze(face)["status"],
                parsing_result["status"],
            ],
        }
