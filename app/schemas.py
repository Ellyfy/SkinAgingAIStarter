from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ImageQuality(BaseModel):
    width: int
    height: int
    brightness: float
    blur_variance: float
    passed: bool
    warnings: List[str] = Field(default_factory=list)


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class RegionScore(BaseModel):
    wrinkle_density: float
    relative_area: float


class FaceParsingResult(BaseModel):
    status: str
    mask_shape: Optional[List[int]] = None
    class_names_found: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    quality: ImageQuality
    face: Optional[FaceBox]
    face_parsing: Optional[FaceParsingResult] = None
    overall_wrinkle_proxy: float
    regions: Dict[str, RegionScore]
    notes: List[str] = Field(default_factory=list)