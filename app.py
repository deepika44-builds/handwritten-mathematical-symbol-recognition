
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import json
import random
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Handwritten Math Recognition",
    page_icon="✍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_math_model.keras")

model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "handwritten_math_model.keras"))

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_labels.json"), "r") as f:
    classes = json.load(f)

data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.npz"))
X_test = data["X_test"]
y_test = data["y_test"]

st.title("Handwritten Mathematical Symbol Recognition")
st.write("An AI-based system that recognises handwritten digits and mathematical symbols using CNN.")

st.divider()

st.subheader("Test the Recognition Model")

if st.button("Generate Random Test"):
    st.session_state.sample_index = random.randrange(len(X_test))

if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0

index = st.session_state.sample_index

image = X_test[index]
actual = classes[int(y_test[index])]

prediction = model.predict(image.reshape(1, 28, 28, 1), verbose=0)
predicted_index = int(np.argmax(prediction))
predicted = classes[predicted_index]
confidence = float(np.max(prediction) * 100)

col1, col2 = st.columns(2)

with col1:
    st.image(image.reshape(28, 28), caption="Test Image", width=200)

with col2:
    st.metric("Actual Symbol", actual)
    st.metric("Predicted Symbol", predicted)
    st.metric("Confidence", f"{confidence:.2f}%")

    if actual == predicted:
        st.success("Correct Prediction!")
    else:
        st.error("Incorrect Prediction")

st.divider()

st.subheader("Model Information")
st.write("Model: Convolutional Neural Network (CNN)")
st.write("Number of Classes:", len(classes))
st.write("Image Size: 28 × 28 pixels")
st.write("Dataset Test Images:", len(X_test))

st.caption("Handwritten Mathematical Symbol Recognition Project")
