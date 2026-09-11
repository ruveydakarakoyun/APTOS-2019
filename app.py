import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import base64

from model_utils import load_model, preprocess_image, predict
from gradcam_utils import generate_gradcam


# ==============================================================================
# SAYFA AYARLARI
# ==============================================================================
st.set_page_config(
    page_title="Diyabetik Retinopati Analizi",
    page_icon="👁️",
    layout="wide"
)


# ==============================================================================
# TASARIM
# ==============================================================================
st.markdown(
    """
<style>

/* Ana içerik alanı */
.block-container {
    max-width: 1180px;
    padding-top: 3.5rem;
    padding-bottom: 2rem;
}

/* --------------------------------------------------------------------------
   ÜST ALAN
-------------------------------------------------------------------------- */

.hero {
    background: linear-gradient(135deg, #f8faff 0%, #f1f5ff 100%);
    border: 1px solid #e2e7f2;
    border-radius: 18px;
    padding: 26px 30px 22px 30px;
    margin-bottom: 28px;
    box-shadow: 0 4px 18px rgba(40, 50, 90, 0.035);
}

.hero-top {
    display: flex;
    align-items: center;
    gap: 16px;
}

.hero-icon {
    width: 50px;
    height: 50px;
    flex-shrink: 0;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #e0e4f2;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 27px;
    box-shadow: 0 2px 8px rgba(40, 50, 90, 0.05);
}

.hero-heading {
    display: flex;
    flex-direction: column;
}

.hero-kicker {
    font-size: 11px;
    font-weight: 750;
    color: #5951d8;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.hero-title {
    font-size: 31px;
    font-weight: 760;
    color: #202638;
    line-height: 1.15;
}

.hero-description {
    border-top: 1px solid #e2e6ef;
    margin-top: 18px;
    padding-top: 15px;
    color: #6b7280;
    font-size: 14.5px;
    line-height: 1.55;
}


/* --------------------------------------------------------------------------
   BÖLÜM ETİKETİ
-------------------------------------------------------------------------- */

.section-label {
    font-size: 12px;
    font-weight: 750;
    color: #737b8c;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}


/* --------------------------------------------------------------------------
   GÖRSEL KARTLARI
-------------------------------------------------------------------------- */

.visual-card {
    border: 1px solid #e5e9f2;
    border-radius: 16px;
    background: #ffffff;
    padding: 16px;
    box-shadow: 0 3px 12px rgba(20, 30, 60, 0.04);
}

.visual-title {
    font-size: 18px;
    font-weight: 700;
    color: #292e3d;
    margin-bottom: 3px;
}

.visual-caption {
    font-size: 13px;
    color: #8a909d;
    margin-bottom: 13px;
}

.image-frame {
    width: 100%;
    height: 330px;
    background: #f7f8fb;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.image-frame img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}


/* --------------------------------------------------------------------------
   SONUÇ KARTI
-------------------------------------------------------------------------- */

.result-card {
    border: 1px solid #e3e5ff;
    border-radius: 16px;
    background: linear-gradient(135deg, #fafaff 0%, #f4f5ff 100%);
    padding: 22px;
    min-height: 245px;
}

.result-label {
    font-size: 12px;
    font-weight: 700;
    color: #7c8291;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}

.result-class {
    font-size: 28px;
    font-weight: 750;
    color: #5146e5;
    line-height: 1.2;
    margin-bottom: 16px;
}

.confidence-label {
    font-size: 13px;
    color: #777d89;
    margin-bottom: 2px;
}

.confidence-value {
    font-size: 22px;
    font-weight: 700;
    color: #292e3d;
    margin-bottom: 14px;
}

.result-description {
    font-size: 14px;
    color: #626875;
    line-height: 1.5;
}


/* --------------------------------------------------------------------------
   OLASILIKLAR
-------------------------------------------------------------------------- */

.prob-card {
    border: 1px solid #e5e9f2;
    border-radius: 16px;
    background: #ffffff;
    padding: 22px;
    min-height: 245px;
}

.prob-title {
    font-size: 17px;
    font-weight: 700;
    color: #292e3d;
    margin-bottom: 17px;
}

.prob-row {
    margin-bottom: 12px;
}

.prob-head {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-size: 13px;
}

.prob-name {
    font-weight: 600;
    color: #3f4451;
}

.prob-value {
    color: #727887;
    font-weight: 600;
}

.prob-track {
    width: 100%;
    height: 7px;
    background: #edf0f5;
    border-radius: 999px;
    overflow: hidden;
}

.prob-fill {
    height: 7px;
    border-radius: 999px;
    background: #b8c0ce;
}

.prob-fill.active {
    background: linear-gradient(90deg, #5146e5, #7167ef);
}


/* --------------------------------------------------------------------------
   STREAMLIT ELEMANLARI
-------------------------------------------------------------------------- */

.stButton > button {
    width: 100%;
    background: #5146e5;
    color: white;
    border: none;
    border-radius: 10px;
    height: 46px;
    font-weight: 650;
}

.stButton > button:hover {
    background: #4338ca;
    color: white;
}

[data-testid="stFileUploader"] {
    border-radius: 14px;
}

.result-divider {
    border-top: 1px solid #e9ecf2;
    margin: 24px 0;
}


/* --------------------------------------------------------------------------
   BİLGİLENDİRME
-------------------------------------------------------------------------- */

.disclaimer {
    margin-top: 24px;
    padding: 13px 16px;
    border-radius: 12px;
    background: #fffbeb;
    border: 1px solid #f3e6ad;
    color: #6c5a1b;
    font-size: 13px;
    line-height: 1.5;
}

.footer {
    margin-top: 20px;
    text-align: center;
    color: #969ba6;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True
)


# ==============================================================================
# YARDIMCI FONKSİYON
# ==============================================================================
def image_to_base64(image):
    image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


# ==============================================================================
# ÜST ALAN
# ==============================================================================
st.markdown(
    '<div class="hero">'
        '<div class="hero-top">'
            '<div class="hero-icon">👁️</div>'
            '<div class="hero-heading">'
                '<div class="hero-kicker">'
                    'APTOS 2019 · Derin Öğrenme Tabanlı Analiz'
                '</div>'
                '<div class="hero-title">'
                    'Diyabetik Retinopati Analizi'
                '</div>'
            '</div>'
        '</div>'
        '<div class="hero-description">'
            'Retina görüntülerinden diyabetik retinopati seviyesini analiz edin '
            've modelin karar verirken odaklandığı bölgeleri Grad-CAM ile inceleyin.'
        '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ==============================================================================
# MODEL
# ==============================================================================
@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# ==============================================================================
# DOSYA YÜKLEME
# ==============================================================================
st.markdown(
    '<div class="section-label">Görüntü Yükleme</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Retina görüntülerinizi seçin",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ==============================================================================
# ANALİZ
# ==============================================================================
if uploaded_files:

    if st.button(
        "✨ Görüntüleri Analiz Et",
        use_container_width=True
    ):

        class_names = [
            "No DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferative DR"
        ]

        descriptions = [
            "Normal retina bulgusu",
            "Hafif düzey diyabetik retinopati bulguları",
            "Orta düzey diyabetik retinopati bulguları",
            "İleri düzey diyabetik retinopati bulguları",
            "Proliferatif diyabetik retinopati bulguları"
        ]

        for uploaded_file in uploaded_files:

            # ------------------------------------------------------------------
            # MODEL İŞLEMLERİ
            # ------------------------------------------------------------------
            image = Image.open(uploaded_file).convert("RGB")

            input_tensor = preprocess_image(image)

            predicted_class, probabilities = predict(
                model,
                input_tensor
            )

            confidence = float(
                probabilities[predicted_class]
            ) * 100

            cam_image = generate_gradcam(
                model,
                input_tensor,
                target_class=predicted_class
            )

            if not isinstance(cam_image, Image.Image):
                cam_image = Image.fromarray(cam_image)

            # ------------------------------------------------------------------
            # DOSYA ADI
            # ------------------------------------------------------------------
            st.markdown(
                f'<div style="font-size:14px;color:#737887;'
                f'margin-top:22px;margin-bottom:12px;">'
                f'📄 <b>{uploaded_file.name}</b>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------------------------
            # GÖRSELLER
            # ------------------------------------------------------------------
            original_b64 = image_to_base64(image)
            cam_b64 = image_to_base64(cam_image)

            img_col1, img_col2 = st.columns(
                2,
                gap="large"
            )

            with img_col1:
                st.markdown(
                    f'<div class="visual-card">'
                    f'<div class="visual-title">Orijinal Görüntü</div>'
                    f'<div class="visual-caption">'
                    f'Yüklenen retina görüntüsü'
                    f'</div>'
                    f'<div class="image-frame">'
                    f'<img src="data:image/jpeg;base64,{original_b64}">'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with img_col2:
                st.markdown(
                    f'<div class="visual-card">'
                    f'<div class="visual-title">'
                    f'Model Attention (Grad-CAM)'
                    f'</div>'
                    f'<div class="visual-caption">'
                    f'Modelin tahmin sırasında odaklandığı bölgeler'
                    f'</div>'
                    f'<div class="image-frame">'
                    f'<img src="data:image/jpeg;base64,{cam_b64}">'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="result-divider"></div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------------------------------
            # OLASILIK HTML
            # ------------------------------------------------------------------
            probability_html = ""

            for j, name in enumerate(class_names):

                prob = float(probabilities[j])
                percent = prob * 100

                active_class = (
                    "active"
                    if j == predicted_class
                    else ""
                )

                probability_html += (
                    f'<div class="prob-row">'
                    f'<div class="prob-head">'
                    f'<span class="prob-name">{name}</span>'
                    f'<span class="prob-value">%{percent:.1f}</span>'
                    f'</div>'
                    f'<div class="prob-track">'
                    f'<div class="prob-fill {active_class}" '
                    f'style="width:{percent:.1f}%;"></div>'
                    f'</div>'
                    f'</div>'
                )

            # ------------------------------------------------------------------
            # SONUÇLAR
            # ------------------------------------------------------------------
            result_col1, result_col2 = st.columns(
                [0.9, 1.1],
                gap="large"
            )

            with result_col1:
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-label">'
                    f'MODEL TAHMİNİ'
                    f'</div>'
                    f'<div class="result-class">'
                    f'{class_names[predicted_class]}'
                    f'</div>'
                    f'<div class="confidence-label">'
                    f'Güven skoru'
                    f'</div>'
                    f'<div class="confidence-value">'
                    f'%{confidence:.1f}'
                    f'</div>'
                    f'<div class="result-description">'
                    f'{descriptions[predicted_class]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with result_col2:
                st.markdown(
                    f'<div class="prob-card">'
                    f'<div class="prob-title">'
                    f'Sınıf Olasılıkları'
                    f'</div>'
                    f'{probability_html}'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ==============================================================================
# BİLGİLENDİRME
# ==============================================================================
st.markdown(
    '<div class="disclaimer">'
    '<b>Bilgilendirme</b><br>'
    'Bu uygulama eğitim ve araştırma amaçlıdır ve klinik tanı yerine geçmez. '
    'Tıbbi değerlendirme ve kararlar için göz hastalıkları uzmanına danışılmalıdır.'
    '</div>',
    unsafe_allow_html=True
)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    '<div class="footer">'
    'Geliştiren ve Tasarlayan: '
    '<b>Ceren Kocabaş & Rüveyda Karakoyun</b> '
    '• APTOS 2019 Projesi'
    '</div>',
    unsafe_allow_html=True
)
