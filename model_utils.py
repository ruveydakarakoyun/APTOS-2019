import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import numpy as np

from dataset import auto_crop, apply_clahe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/resnet50_best_model.pth"

@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)

    # Kaydedilen ağırlıkları yükle
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


# Sadece resize + tensor + normalize (auto-crop/CLAHE zaten uygulanmış görüntü için)
_final_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image):
    """
    Eğitimde kullanılan ön işleme sırasıyla aynı: Auto-Crop -> CLAHE -> Resize/Normalize.
    `image` bir PIL Image olmalı (RGB).
    """
    image_np = np.array(image.convert("RGB"))
    cropped_image = auto_crop(image_np, threshold=10, padding=5)
    clahe_image = apply_clahe(cropped_image)

    input_tensor = _final_transform(clahe_image).unsqueeze(0).to(DEVICE)
    return input_tensor


def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class, probabilities
