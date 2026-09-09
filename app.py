import streamlit as st
import torch
from PIL import Image
from model_utils import load_model, preprocess_image, predict
from gradcam_utils import generate_gradcam

st.set_page_config(
    page_title="APTOS-2019 Retina Sınıflandırma",
    layout="centered"
)

st.title("👁️ Diyabetik Retinopati (APTOS 2019)")
st.write("Retina fotoğrafınızı yükleyin, model değerlendirsin.")

# Modeli yükle
model = load_model()

# Dosya yükleme bileşeni
uploaded_files = st.file_uploader(
    "Bir retina fotoğrafı seçin (.png, .jpg, .jpeg)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Analiz Et"):
        with st.spinner("Model analiz ediyor..."):
            cols = st.columns(2)

            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i % 2]:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Yüklenen Orijinal Görsel", width="stretch")

                    # Ön işleme ve tahmin
                    input_tensor = preprocess_image(image)
                    predicted_class, probabilities = predict(model, input_tensor)

                    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

                    st.success(
                        f"**Tahmin Edilen Evre:** "
                        f"{class_names[predicted_class]} "
                        f"(Sınıf {predicted_class})"
                    )

                    # Olasılık grafiği
                    st.subheader("Sınıf Olasılıkları")
                    st.bar_chart({
                        class_names[j]: float(probabilities[j])
                        for j in range(5)
                    })

                    # Grad-CAM (Mock / Gerçek)
                    st.subheader("Grad-CAM Isı Haritası")
                    cam_image = generate_gradcam(
                        model,
                        input_tensor,
                        target_class=predicted_class
                    )
                    st.image(
                        cam_image,
                        caption="Modelin Odak Noktaları",
                        width="stretch"
                    )
