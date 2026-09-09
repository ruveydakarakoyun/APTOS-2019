import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize_for_display(input_tensor):
    """
    model_utils.preprocess_image() ile normalize edilmiş tensor'u,
    Grad-CAM ısı haritasını üzerine bindirebilmek için
    0-1 aralığında RGB görüntüye geri çevirir.
    """
    img = input_tensor.squeeze(0).detach().cpu()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam(model, input_tensor, target_class=None):
    """
    app.py kullanımı:
        cam_image = generate_gradcam(model, input_tensor, target_class=predicted_class)
        st.image(cam_image, ...)

    Bu yüzden tuple değil, doğrudan gösterilebilir TEK bir görüntü (numpy array) döner.
    """
    model.eval()

    # ResNet50'nin son convolution bloğu
    target_layers = [model.layer4[-1]]

    if target_class is None:
        with torch.inference_mode():
            outputs = model(input_tensor)
            target_class = torch.argmax(outputs, dim=1).item()

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0]

    # input_tensor normalize edilmişti, görselleştirme için geri çeviriyoruz
    visual_image = _denormalize_for_display(input_tensor)

    cam_image = show_cam_on_image(
        visual_image,
        grayscale_cam,
        use_rgb=True
    )

    return cam_image
