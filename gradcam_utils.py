import numpy as np
from PIL import Image

def generate_gradcam(model, input_tensor, target_class=None):
    # Arkadaşınız gerçek Grad-CAM kodunu verene kadar geçici mock görsel üretir
    img_np = np.random.randint(100, 250, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_np)
