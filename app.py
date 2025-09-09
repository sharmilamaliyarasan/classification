import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Digit Classification App", page_icon="🖋️")

st.title("🖋️ Digit Classification App")
st.write("Predict handwritten digits (0–9) using a trained ML model.")

MODEL_PATH = "best_model.pkl" 

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Pickle file not found at '{MODEL_PATH}'. Make sure it exists in your repo.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

input_method = st.sidebar.radio("Select Input Method:", ("Manual Input (64 features)", "Upload Image"))

if input_method == "Manual Input (64 features)":
    st.sidebar.header("Enter 64 Pixel Values")
    st.sidebar.write("Each pixel value: 0–16 grayscale")

    features = []
    for i in range(64):
        value = st.sidebar.number_input(
            f"Feature {i+1}", min_value=0.0, max_value=16.0, step=1.0, value=0.0
        )
        features.append(value)

    if st.button("Predict Digit (Manual Input)"):
        features_array = np.array(features).reshape(1, -1)
        try:
            prediction = model.predict(features_array)[0]
            st.success(f"✅ Predicted Digit: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

else:
    uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  
        st.image(image, caption="Uploaded Digit", width=150)

        image_resized = image.resize((8, 8))
        data = np.array(image_resized)
        
        features_array = 16 - (data / 255.0 * 16)
        features_array = features_array.flatten().reshape(1, -1)

        if st.button("Predict Digit (Image)"):
            try:
                prediction = model.predict(features_array)[0]
                st.success(f"✅ Predicted Digit: **{prediction}**")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
