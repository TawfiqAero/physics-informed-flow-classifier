import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model, predict_regime

st.set_page_config(page_title="Flow Regime Classifier", layout="wide")

st.title("Hybrid Physics-Informed ML for Flow Regime Classification")

st.markdown("""
Upload a CFD velocity field (.npy file) to classify the flow regime.
Supported regimes:
- Laminar
- Vortex Shedding
- Transitional/Turbulent
""")

# Upload file
uploaded_file = st.file_uploader("Upload Velocity Field (.npy)", type=["npy"])

if uploaded_file is not None:
    data = np.load(uploaded_file)

    st.subheader("Velocity Field Preview")
    fig, ax = plt.subplots()
    im = ax.imshow(data[0], origin="lower")
    plt.colorbar(im)
    st.pyplot(fig)

    if st.button("Predict Flow Regime"):
        model = load_model("model/model.pth")
        prediction, confidence = predict_regime(model, data)

        regime_names = {
            0: "Laminar",
            1: "Vortex Shedding",
            2: "Transitional / Turbulent"
        }

        st.success(f"Predicted Regime: {regime_names[prediction]}")
        st.info(f"Confidence: {confidence:.2f}%")
