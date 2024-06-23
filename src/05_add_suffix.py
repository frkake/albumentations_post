from typing import List, Tuple
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import rootutils
import importlib

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.utils import make_grid_image
Crop = importlib.import_module("src.02_dualtransform").Crop

class CropAndAddSuffix(Crop):
    def apply_to_str(self, string: str, **params) -> str:
        return string + "/cropped"
    
    @property
    def targets(self) -> List[str]:
        return {
            "image": self.apply,
            "mask": self.apply,
            "image_name": self.apply_to_str,
        }


transform = CropAndAddSuffix(100, 100, 300, 300, p=1)

image_path = Path('data/images/dog_and_cat.png')
image = cv2.imread(str(image_path))
transformed = transform(image=image, image_name=image_path.stem)
print(f"transformed.keys(): {transformed.keys()}")
print(f"transformed['image_name']: {transformed['image_name']}")
