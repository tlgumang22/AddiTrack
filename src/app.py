import streamlit as st
import pandas as pd
import subprocess
import os
import base64
from streamlit_lottie import st_lottie
import requests
import os

from Training import (
    train_or_load_model_and_noise,
    simulate_height_series,
)

# ======================
# Helper Functions
# ======================
def load_lottie(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ======================
# Load default model
# ======================
model, noise_stats = train_or_load_model_and_noise()

# ======================
# Page Config
# ======================
st.set_page_config(page_title="Deposition Simulator", page_icon="⚙️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {
        background-color: #2c3e50;
        color:white;
        border-radius:12px;
        padding:8px 16px;
    }
    .stButton>button:hover {
        background-color: #1abc9c;
        color:white;
    }
    .block-container {
        padding-top:2rem;
        padding-bottom:2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Header Section
# ======================
col1, col2 = st.columns([2,1])
with col1:
    st.title("⚙️ Deposition Process Simulator")
    st.write("A B.Tech project on **Additive Manufacturing** – simulating layer deposition with AI + MATLAB integration.")
with col2:
    lottie_robot = load_lottie("https://assets10.lottiefiles.com/packages/lf20_zrqthn6o.json")
    if lottie_robot:
        st_lottie(lottie_robot, height=200, key="robot")

st.markdown("---")

# ======================
# Tabs for Main Sections
# ======================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Simulation", "🔄 Training", "📖 Read About Project", "👨‍💻 About Developer"
])

# ----------------------
# Simulation Tab
# ----------------------
with tab1:
    st.header("📊 Generate Simulation Data")

    # New Animation (Process Animation)
    lottie_process = load_lottie("https://assets2.lottiefiles.com/packages/lf20_cg3c4z.json")
    if lottie_process:
        st_lottie(lottie_process, height=180, key="sim-process")

    st.sidebar.header("Input Parameters")
    V = st.sidebar.number_input("⚡ Voltage (V)", min_value=0.0, value=7.0, step=0.1)
    I = st.sidebar.number_input("🔌 Current (A)", min_value=0.0, value=14.5, step=0.1)
    F = st.sidebar.number_input("🚀 Feedrate (mm/min)", min_value=0.0, value=48.0, step=1.0)
    T = st.sidebar.number_input("⏱ Time (s)", min_value=1, value=100, step=1)

    if st.sidebar.button("Generate Output"):
        table = simulate_height_series(
            model, V, I, F, T, noise_stats, seed=42,
            enforce_nonnegative=True, monotonic_soft=True
        )
        st.success("✅ Simulation complete!")

        # Show data
        st.subheader("Generated Output Table")
        st.dataframe(table)

        # Save Excel
        out_file = f"C:\\Users\\dosiu\\OneDrive\\Desktop\\python_vscode\\BTP\data\\simulated_series_V{V}_I{I}_F{F}_T{T}.xlsx"
        table.to_excel(out_file, index=False, engine="openpyxl")

        # Download link
        with open(out_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{out_file}">📥 Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.markdown("### 🎥 MATLAB 3D Simulation")
    if st.button("Run MATLAB Simulation"):
        try:
            matlab_script = r"C:\Users\dosiu\OneDrive\Desktop\python_vscode\BTP\scripts\depositionAnimation.m"
            subprocess.run(["matlab", "-batch", f"run('{matlab_script}')"], check=True)

        except Exception as e:
            st.error(f"MATLAB run failed: {e}")

# ----------------------
# Training Tab
# ----------------------
with tab2:
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("🔄 Retrain Model with New Dataset")
        st.write("Upload a new dataset to retrain the AI model for deposition simulation.")
    with col2:
        lottie_train = load_lottie("https://assets4.lottiefiles.com/packages/lf20_j1adxtyb.json")
        if lottie_train:
            st_lottie(lottie_train, height=180, key="train-anim")

    uploaded = st.file_uploader("Upload new dataset (Excel)", type=["xlsx"])
    if uploaded is not None:
        new_path = r"BTP\data\uploaded_dataset.xlsx"
        with open(new_path, "wb") as f:
            f.write(uploaded.read())

        if st.button("Retrain Model"):
            model, noise_stats = train_or_load_model_and_noise(new_path)
            st.success("✅ Model retrained successfully!")

# ----------------------
# About Project Tab
# ----------------------
with tab3:
    col1, col2 = st.columns([3,1])
    with col1:
        st.header("📖 Read About Project")
        st.write("""
        This project focuses on **Additive Manufacturing (AM)**, also known as **3D Printing**.  
        It integrates **Machine Learning** with **MATLAB-based 3D simulation** to:
        - Predict deposition **height profiles** from process parameters (Voltage, Current, Feedrate, Time).
        - Generate **synthetic datasets** for process planning.
        - Visualize deposition in **3D animations** for better understanding.
        
        The ML model is trained on experimental data to capture process behavior,  
        and then extended with noise modeling for realistic simulation.
        """)
    with col2:
        lottie_project = load_lottie("https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json")
        if lottie_project:
            st_lottie(lottie_project, height=180, key="proj-anim")

    # New Animation (Research/Project Animation)


# ----------------------
# About Developer Tab
# ----------------------
with tab4:
    st.header("👨‍💻 About Developer")

    col1, col2 = st.columns([1,3])
    with col1:
        st.image(os.path.join("extra", "your_photo.jpg"), caption="Umang Dosi", width=160)
    with col2:
        st.write("""
        Hi, I am **Umang Dosi**, a 4th-year Mechanical Engineering student at IIT Indore.  
        My interests include:
        - 🧠 Machine Learning & AI for engineering applications  
        - ⚙️ Additive Manufacturing and Digital Twins  
        - 📸 Photography and Visual Storytelling  

        This project combines my interest in **AI + Manufacturing**,  
        showcasing how intelligent models can assist in real-world process optimization.
        """)
    st.write("")
    st.write("")
    col1, col2 = st.columns([1,3])
    with col1:
        # st.image(r"C:\Users\dosiu\OneDrive\Desktop\python_vscode\BTP\extra\your_photo.jpg", caption="Umang Dosi", width=160)
        st.image(os.path.join("extra", "your_photo.jpg"), caption="Umang Dosi", width=160)

    with col2:
        st.write("""
        Hi, I am **Umang Dosi**, a 4th-year Mechanical Engineering student at IIT Indore.  
        My interests include:
        - 🧠 Machine Learning & AI for engineering applications  
        - ⚙️ Additive Manufacturing and Digital Twins  
        - 📸 Photography and Visual Storytelling  

        This project combines my interest in **AI + Manufacturing**,  
        showcasing how intelligent models can assist in real-world process optimization.
        """)
