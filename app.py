import streamlit as st
import torch
from PIL import Image
from model_utils import load_model, preprocess_image, predict
from gradcam_utils import generate_gradcam

st.set_page_config(
    page_title="Diyabetik Retinopati Analizi",
    page_icon="👁️",
    layout="wide"
)

# Özel UI / Kart Tasarımı İçin CSS Enjeksiyonu
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9f0fc 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #dce6f1;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        height: 48px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)

# Üst Bilgi Alanı
st.markdown("""
<div class="main-header">
    <h2>👁️ Diyabetik Retinopati Analizi</h2>
    <p style="color: #666; font-size: 16px;">Retina fotoğraflarınızı yükleyin, yapay zeka modeli görüntüleri değerlendirerek diyabetik retinopati düzeyini tahmin etsin.</p>
    <span style="background-color: #ede9fe; color: #6d28d9; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;">APTOS 2019 • AI-assisted screening</span>
</div>
""", unsafe_allow_html=True)

# Modeli Yükle
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Dosya Yükleme Alanı
uploaded_files = st.file_uploader(
    "Retina fotoğraflarınızı buraya sürükleyin veya tıklayın",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("✨ Görüntüleri Analiz Et"):
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        descriptions = [
            "Normal retina bulgusu",
            "Hafif düzey diyabetik retinopati bulguları",
            "Orta düzey diyabetik retinopati bulguları",
            "İleri düzey diyabetik retinopati bulguları",
            "Proliferatif diyabetik retinopati bulguları"
        ]

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            input_tensor = preprocess_image(image)
            predicted_class, probabilities = predict(model, input_tensor)
            confidence = float(probabilities[predicted_class]) * 100

            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Görsel Karşılaştırma Kolonları (Orijinal vs Grad-CAM)
            img_col1, img_col2 = st.columns(2)

            with img_col1:
                st.markdown("##### **Orijinal Görüntü**")
                st.caption("Yüklenen retina görüntüsü")
                st.image(image, use_container_width=True)

            with img_col2:
                st.markdown("##### **Model Attention (Grad-CAM)**")
                st.caption("Modelin odaklandığı bölgeler (açıklanabilir yapay zeka)")
                cam_image = generate_gradcam(model, input_tensor, target_class=predicted_class)
                st.image(cam_image, use_container_width=True)

            st.markdown("---")

            # Tahmin ve Olasılık Dağılımı
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("##### **MODEL TAHMİNİ**")
                st.markdown(f"<h2 style='color:#4f46e5; margin:0;'>{class_names[predicted_class]}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Güven skoru: %{confidence:.1f}**")
                st.info(descriptions[predicted_class])

            with res_col2:
                st.markdown("##### **SINIF OLASILIKLARI**")
                for j, name in enumerate(class_names):
                    prob_val = float(probabilities[j])
                    st.write(f"**{name}** ({prob_val*100:.1f}%)")
                    st.progress(prob_val)

    # Yasal Uyarı
    st.warning(
        "**Önemli Bilgilendirme**\n\n"
        "Bu uygulama eğitim ve araştırma amaçlıdır. Klinik tanı yerine geçmez. Tıbbi kararlar için mutlaka bir göz hastalıkları uzmanına danışınız."
    )
# Tasarlayanlar / Geliştiriciler Bilgisi
    st.markdown(
         "<div style='text-align: center; color: #666; font-size: 14px; margin-top: 30px; padding: 10px;'>"
         "Geliştiren ve Tasarlayan: <b>Ceren Kocabaş & Rüveyda Karakoyun</b> • APTOS 2019 Projesi"
         "</div>",
         unsafe_allow_html=True
    )
