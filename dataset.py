import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# ==============================================================================
# 1. Auto-Crop Fonksiyonu
# ==============================================================================
def auto_crop(image, threshold=10, padding=5):
    if isinstance(image, Image.Image):
        image = np.array(image)

    gray = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 3
        else image
    )
    mask = gray > threshold

    if not np.any(mask):
        return image

    y_coords, x_coords = np.where(mask)
    x_min, y_min = max(0, x_coords.min() - padding), max(
        0, y_coords.min() - padding
    )
    x_max, y_max = min(image.shape[1], x_coords.max() + padding + 1), min(
        image.shape[0], y_coords.max() + padding + 1
    )

    cropped = image[y_min:y_max, x_min:x_max]

    if (
        cropped.shape[0] < image.shape[0] * 0.1
        or cropped.shape[1] < image.shape[1] * 0.1
    ):
        return image

    return cropped

# ==============================================================================
# 2. CLAHE Fonksiyonu
# ==============================================================================
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return enhanced_image

# ==============================================================================
# 3. APTOSDataset Sınıfı
# ==============================================================================
class APTOSDataset(Dataset):
    def __init__(self, df, folder_prefix="", transform=None):
        self.df = df.reset_index(drop=True)
        self.folder_prefix = folder_prefix
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_file = (
            row["image_file"]
            if "image_file" in row and pd.notnull(row["image_file"])
            else f"{row['id_code']}.png"
        )
        label = (
            int(row["diagnosis"])
            if "diagnosis" in row and pd.notnull(row["diagnosis"])
            else -1
        )

        img_path = os.path.join(self.folder_prefix, img_file)

        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            img_cropped = auto_crop(img_np, threshold=10, padding=5)
            img_final = apply_clahe(img_cropped)
        except Exception:
            img_final = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=img_final)
            img_tensor = augmented["image"]
        else:
            img_tensor = (
                torch.from_numpy(img_final).permute(2, 0, 1).float() / 255.0
            )

        return img_tensor, torch.tensor(label, dtype=torch.long)
