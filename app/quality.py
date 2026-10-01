import cv2
import numpy as np

class QualityChecker:
    def __init__(
        self,
        min_width=480,
        min_height=480,
        min_brightness=45.0,
        max_brightness=220.0,
        min_blur_variance=45.0,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_variance = min_blur_variance

    def check(self, image_bgr: np.ndarray) -> dict:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        warnings = []
        if w < self.min_width or h < self.min_height:
            warnings.append("Image resolution is lower than recommended.")
        if brightness < self.min_brightness:
            warnings.append("Image appears too dark.")
        if brightness > self.max_brightness:
            warnings.append("Image appears overexposed.")
        if blur_variance < self.min_blur_variance:
            warnings.append("Image may be blurry.")

        return {
            "width": w,
            "height": h,
            "brightness": round(brightness, 2),
            "blur_variance": round(blur_variance, 2),
            "passed": not warnings,
            "warnings": warnings,
        }
