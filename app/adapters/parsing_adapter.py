import sys
from pathlib import Path

import numpy as np
import torch

from PIL import Image
import torchvision.transforms as transforms

from .base import FaceParsingAdapter


FACE_CLASSES = {
    0: "background",
    1: "skin",
    2: "left eyebrow",
    3: "right eyebrow",
    4: "left eye",
    5: "right eye",
    6: "eyeglasses",
    7: "left ear",
    8: "right ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper lip",
    13: "lower lip",
    14: "neck",
    15: "neck accessory",
    16: "clothing",
    17: "hair",
    18: "hat",
}


class BiSeNetParsingAdapter(FaceParsingAdapter):
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.model = None

        if not self.enabled:
            return

        # Project root:
        # C:\Dev\SkinAgingAIStarter
        project_root = Path(__file__).resolve().parents[2]

        # External BiSeNet repository
        face_parsing_root = project_root / "external" / "face-parsing"

        # Pretrained weights
        weight_path = face_parsing_root / "weights" / "resnet18.pt"

        # Allow imports from external/face-parsing
        if str(face_parsing_root) not in sys.path:
            sys.path.insert(0, str(face_parsing_root))

        from models.bisenet import BiSeNet

        # GPU if available, otherwise CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create 19-class BiSeNet face parsing model
        self.model = BiSeNet(
            num_classes=19,
            backbone_name="resnet18",
        )

        # Load pretrained weights
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"BiSeNet loaded on: {self.device}")

    @torch.no_grad()
    def parse(self, face_bgr: np.ndarray) -> dict:
        if not self.enabled:
            return {
                "enabled": False,
                "status": "BiSeNet face parsing is disabled.",
            }

        # OpenCV image is BGR.
        # BiSeNet expects RGB.
        face_rgb = face_bgr[:, :, ::-1]

        # Convert NumPy image to PIL
        image = Image.fromarray(face_rgb)

        # Save original face size
        original_size = image.size

        # Resize for BiSeNet
        resized = image.resize(
            (512, 512),
            resample=Image.BILINEAR,
        )

        # Same preprocessing used by inference.py
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

        image_tensor = (
            transform(resized)
            .unsqueeze(0)
            .to(self.device)
        )

        # Run BiSeNet
        output = self.model(image_tensor)[0]

        # Choose the most likely class for each pixel
        mask = (
            output
            .squeeze(0)
            .cpu()
            .numpy()
            .argmax(0)
        )

        # Resize mask back to original face size
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize(
            original_size,
            resample=Image.NEAREST,
        )

        mask = np.array(mask_pil)

        class_ids = np.unique(mask).tolist()
        class_names = [
            FACE_CLASSES[class_id]
            for class_id in class_ids
        ]

        return {
            "enabled": True,
            "status": "BiSeNet face parsing completed.",
            "mask": mask,
            "mask_shape": list(mask.shape),
            "classes_found": class_ids,
            "class_names_found": class_names,
        }