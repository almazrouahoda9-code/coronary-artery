import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("❤️ Coronary Artery Disease Detection")

model = load_model("model.h5")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image)

    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.error("Abnormal")
    else:
        st.success("Normal")
