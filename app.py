import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import waste_logic as wl

st.set_page_config(page_title='Smart Waste Classifier', page_icon='♻️', layout='centered')
st.title('♻️ Smart Waste Classification & Disposal Recommendation')
st.write('Upload a photo of a waste item to get its category, a confidence score, disposal guidance and a visual explanation (Grad-CAM).')


@st.cache_resource
def load_model():
    path = os.path.join(wl.APP_DIR, 'best_model.keras')
    return tf.keras.models.load_model(path, safe_mode=False)


model = load_model()

uploaded = st.file_uploader('Upload waste image', type=['jpg', 'jpeg', 'png'])
residue = st.radio('Does the item have food, grease or liquid residue on it?', ['No / not sure', 'Yes'], horizontal=True)

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    category, conf, probs, arr = wl.classify(model, img)

    col1, col2 = st.columns(2)
    col1.image(img, caption='Uploaded image', use_container_width=True)
    cam = wl.make_gradcam(model, arr)
    col2.image(wl.overlay_gradcam(img, cam), caption='Grad-CAM: regions that influenced the prediction', use_container_width=True)

    st.subheader(f'Prediction: {category.upper()}')
    st.progress(min(conf, 1.0), text=f'Confidence: {conf * 100:.1f}%')

    rec = wl.get_recommendation(category, conf, contaminated=(residue == 'Yes'))
    if rec['status'] == 'uncertain':
        st.warning(rec['message'])
    else:
        if rec['warning']:
            st.error(rec['warning'])
        st.success(f"**Bin:** {rec['bin']}")
        st.write(f"**Preparation:** {rec['prep']}")
        st.info(f"**Environmental note:** {rec['env_note']}")

    with st.expander('All class probabilities'):
        for c, p in zip(wl.CLASS_NAMES, probs):
            st.write(f'{c}: {p * 100:.1f}%')
