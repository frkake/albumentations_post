from typing import List, Tuple, Dict, Any
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import rootutils
import importlib

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

from src.utils import make_grid_image
Crop = importlib.import_module("src.02_dualtransform").Crop

class MergeMasks(A.DualTransform):
    def __init__(
        self, 
        object_value: Tuple[int, int, int] = (255, 0, 0),
        bg_value: Tuple[int, int, int] = (0, 255, 0),
        always_apply=False,
        p=1,
    ):
        super().__init__(always_apply, p)
        self.object_value = object_value
        self.bg_value = bg_value
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img
    
    def apply_with_params(
        self, 
        params: Dict[str, Any], 
        *args: Any, 
        **kwargs: Any,
    ) -> Dict[str, Any]:
        res = super().apply_with_params(params, *args, **kwargs)
        
        H, W = res["mask_dog"].shape
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        mask_object = np.logical_or(res["mask_dog"], res["mask_cat"])
        mask_bg = np.logical_or(res["mask_grass"], res["mask_tree"])
        union = np.logical_or(mask_object, mask_bg)
        complement = np.logical_not(union)
        
        canvas[mask_object] = self.object_value
        canvas[mask_bg] = self.bg_value
        res["mask_merged"] = canvas
        res["image_overlay"] = cv2.addWeighted(res["image"], 0.5, canvas, 0.5, 0)
        
        return res
    
    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("object_value", "bg_value")

transform = A.Compose([
    A.Rotate(limit=10, border_mode=0, p=1), 
    MergeMasks(object_value=(0, 170, 246), bg_value=(255, 90, 0), p=1),
], additional_targets={
    "mask_dog": "mask", 
    "mask_cat": "mask",
    "mask_grass": "mask",
    "mask_tree": "mask",
})

image = cv2.imread('data/images/dog_and_cat.png')
mask_dog = cv2.imread('data/masks/dog.png', cv2.IMREAD_GRAYSCALE)
mask_cat = cv2.imread('data/masks/cat.png', cv2.IMREAD_GRAYSCALE)
mask_grass = cv2.imread('data/masks/grass.png', cv2.IMREAD_GRAYSCALE)
mask_tree = cv2.imread('data/masks/tree.png', cv2.IMREAD_GRAYSCALE)
transformed = transform(
    image=image, 
    mask_dog=mask_dog, 
    mask_cat=mask_cat,
    mask_grass=mask_grass,
    mask_tree=mask_tree,
)
grid_image = make_grid_image([
    transformed["image"], 
    transformed["mask_merged"], 
    transformed["image_overlay"]], n_cols=3)
cv2.imwrite('data/results/merged_mask.png', grid_image)
