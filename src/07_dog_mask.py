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

class CropDogArea(A.DualTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)
    
    def apply(
        self, 
        image: np.ndarray,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        **params,
    ) -> np.ndarray:
        return image[y_min:y_max, x_min:x_max]
    
    def get_params_dependent_on_targets(self, params):
        mask = params["mask_dog"]
        indices = np.where(mask > 0)
        y_min, y_max = indices[0].min(), indices[0].max()
        x_min, x_max = indices[1].min(), indices[1].max()
        
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    @property
    def targets_as_params(self):
        return ["mask_dog"]

transform = A.Compose([
    CropDogArea(p=1), 
    A.PadIfNeeded(
        min_height=210,
        min_width=140,
        value=(128, 128, 128),
        mask_value=128,
    )], 
    additional_targets={"mask_dog": "mask", "mask_cat": "mask"}
)

image = cv2.imread('data/images/dog_and_cat.png')
mask_dog = cv2.imread('data/masks/dog.png', cv2.IMREAD_GRAYSCALE)
mask_cat = cv2.imread('data/masks/cat.png', cv2.IMREAD_GRAYSCALE)
transformed = transform(image=image, mask_dog=mask_dog, mask_cat=mask_cat)
grid_image = make_grid_image([
    transformed["image"], 
    transformed["mask_dog"], 
    transformed["mask_cat"]], n_cols=3)
cv2.imwrite('data/results/crop_dog_area.png', grid_image)
