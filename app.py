import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("flow_model.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Airfoil Flow Classifier", layout="wide")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #2E86C1;
    }
    .sub-text {
        font-size: 18px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown('<div class="main-title">✈️ Airfoil Flow Regime Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict Laminar vs Turbulent Flow (NACA 0012)</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([1, 2])

# -------------------------------
# INPUT
# -------------------------------
with col1:
    st.subheader("⚙️ Input Parameters")

    Re = st.number_input("Reynolds Number", 10000, 600000, 100000)
    AoA = st.slider("Angle of Attack (°)", -5, 15, 5)

    # Compute Cl & Cd
    AoA_rad = np.radians(AoA)
    Cl = 2 * np.pi * AoA_rad
    Cd = 0.01 + (Cl**2)/(np.pi * 0.9 * 4)

    st.write(f"Estimated Cl: {Cl:.3f}")
    st.write(f"Estimated Cd: {Cd:.4f}")

    predict_btn = st.button("🚀 Predict")

# -------------------------------
# OUTPUT
# -------------------------------
with col2:
    st.subheader("📊 Prediction & Visualization")

    if predict_btn:

        # -------------------------------
        # Prediction
        # -------------------------------
        prediction = model.predict([[Re, AoA, Cl, Cd]])
        prob = model.predict_proba([[Re, AoA, Cl, Cd]])

        if prediction[0] == 0:
            st.success("🌊 Laminar Flow")
        else:
            st.error("🌪️ Turbulent Flow")

        st.write("### Confidence")
        st.progress(float(prob[0][1]))

        st.write(f"Laminar: {prob[0][0]*100:.2f}%")
        st.write(f"Turbulent: {prob[0][1]*100:.2f}%")

        # -------------------------------
        # FLOW REGIME MAP (FAST VERSION)
        # -------------------------------
        st.write("### Flow Regime Map")

        Re_range = np.linspace(10000, 600000, 40)
        AoA_range = np.linspace(-5, 15, 40)

        xx, yy = np.meshgrid(Re_range, AoA_range)

        AoA_rad_grid = np.radians(yy)
        Cl_grid = 2 * np.pi * AoA_rad_grid
        Cd_grid = 0.01 + (Cl_grid**2)/(np.pi * 0.9 * 4)

        inputs = np.c_[xx.ravel(), yy.ravel(), Cl_grid.ravel(), Cd_grid.ravel()]
        Z = model.predict(inputs).reshape(xx.shape)

        fig1, ax1 = plt.subplots()

        ax1.contourf(xx, yy, Z, alpha=0.3)
        ax1.scatter(Re, AoA, color='red', s=100, label="Your Input")

        ax1.set_xlabel("Reynolds Number")
        ax1.set_ylabel("Angle of Attack")
        ax1.set_title("Laminar vs Turbulent Regions")

        ax1.legend()

        st.pyplot(fig1)

        # -------------------------------
        # AIRFLOW VISUALIZATION
        # -------------------------------
        st.write("### 🌬️ Airflow Visualization")

        # NACA 0012
        def naca0012(x):
            t = 0.12
            return 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

        x = np.linspace(0, 1, 300)
        y_upper = naca0012(x)
        y_lower = -y_upper

        # Rotate airfoil
        theta = np.radians(AoA)

        def rotate(x, y, theta):
            return x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta)

        x_u, y_u = rotate(x, y_upper, theta)
        x_l, y_l = rotate(x, y_lower, theta)

        # Reynolds normalization
        Re_norm = (Re - 10000) / (600000 - 10000)

        # Flow field
        X, Y = np.meshgrid(np.linspace(-1, 2, 120),
                           np.linspace(-1.5, 1.5, 120))

        speed = 0.5 + 1.5 * Re_norm

        U = speed * np.cos(theta) * np.ones_like(X)
        V = speed * np.sin(theta) * np.ones_like(Y)

        disturbance = 1.5 * (1 - Re_norm) + 0.3

        # Vectorized disturbance (FAST)
        dx = X - 0.5
        dy = Y
        r2 = dx**2 + dy**2 + 0.02

        U -= disturbance * dx / r2
        V -= disturbance * dy / r2

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        ax2.streamplot(X, Y, U, V, density=2)
        ax2.fill(x_u, y_u, 'black')
        ax2.fill(x_l, y_l, 'black')

        ax2.set_title(f"Airflow (AoA={AoA}°, Re={Re})")
        ax2.set_xlim(-1, 2)
        ax2.set_ylim(-1.5, 1.5)

        st.pyplot(fig2)

# -------------------------------
# Footer
# -------------------------------
st.divider()

st.markdown("""
### 📘 About This Project
This system uses Machine Learning to predict airflow behavior over a **NACA 0012 airfoil**.

- Model: Random Forest  
- Inputs: Reynolds Number, Angle of Attack  
- Output: Flow Regime Classification  
""")
