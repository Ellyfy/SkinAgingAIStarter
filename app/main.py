import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from .pipeline import SkinAgingPipeline
from .schemas import AnalysisResponse

app = FastAPI(
    title="Expression-Aware Skin Aging AI",
    version="0.1.0",
    description="Research prototype; not a medical diagnostic system.",
)
pipeline = SkinAgingPipeline()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase": 1,
        "models": {
            "face_detector": "OpenCV Haar baseline",
            "wrinkle": "Gabor baseline",
            "smirk": "planned",
            "face_parsing": "planned",
        },
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return pipeline.analyze(image)
