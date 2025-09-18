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
from ConstantHeight import train_or_load_model, predict_constant_height
from AvgHeight import compute_avgheight

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
tab1, tab2, tab3, tab4,tab5, tab6 = st.tabs([
    "📊 Simulation", "🔄 Training", "📐 Average Height Profile", "📏 Constant Height Control", "📖 Read About Project", "👨‍💻 About Developer"
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
# AvgHeight Tab
# ----------------------
with tab3:
    st.header("📐 Average Height Profile")

    # Unique Animation
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("This section computes **average height** from a simulated series and adjusts V–I values.")
    with col2:    
        lottie_avg = load_lottie("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")
        if lottie_avg:
            st_lottie(lottie_avg, height=180, key="avg-anim")

    if st.button("Generate Average Height Profile"):
        try:
            # Use the same file that was generated in Simulation tab
            sim_file = f"C:\\Users\\dosiu\\OneDrive\\Desktop\\python_vscode\\BTP\\data\\simulated_series_V{V}_I{I}_F{F}_T{T}.xlsx"
            df_avg, avg_h = compute_avgheight(sim_file)
            st.success("✅ Average height profile generated successfully!")

            st.write(f"**Average Height:** {avg_h:.3f} mm")
            st.dataframe(df_avg)

            # Download link
            with open("data/constant_height_profile.xlsx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="constant_height_profile.xlsx">📥 Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Failed to run AvgHeight: {e}")

# ----------------------
# ConstantHeight Tab
# ----------------------
with tab4:
    st.header("📏 Constant Height Control")

    col1, col2 = st.columns([2,1])
    with col1:
        st.write("This section allows training the inverse model and predicting V–I for a **desired constant height**.")
    with col2:    
        lottie_const = load_lottie("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")
        if lottie_const:
            st_lottie(lottie_const, height=180, key="const-anim")

    if "const_model" not in st.session_state:
        st.session_state.const_model = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train / Load Constant Height Model"):
            try:
                st.session_state.const_model = train_or_load_model()
                st.success("✅ Model trained/loaded successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    with col2:
        desired_h = st.number_input("Enter Desired Height (mm)", min_value=0.0, value=5.0, step=0.1)
        if st.button("Generate Constant Height Profile"):
            if st.session_state.const_model is None:
                st.warning("⚠️ Please train/load the model first.")
            else:
                try:
                    df_const = predict_constant_height(st.session_state.const_model, desired_h)
                    st.dataframe(df_const)

                    # Download link
                    const_file = r"data\predicted_iv.xlsx"
                    if os.path.exists(const_file):
                        with open(const_file, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                            href = f'<a href="data:application/octet-stream;base64,{b64}" download="predicted_iv.xlsx">📥 Download Excel</a>'
                            st.markdown(href, unsafe_allow_html=True)

                    st.success(f"✅ Predictions generated for constant height {desired_h} mm")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ----------------------
# About Project Tab
# ----------------------
with tab5:
    col1, col2 = st.columns([2,1])
    with col1:
        st.header("📖 Read About Project")
        st.write("""
        ### 🔹 Overview  
        This project is built around **Additive Manufacturing (AM)**, often called **3D Printing**.  
        It focuses on simulating the **layer-by-layer deposition process** with the help of **AI models** and **MATLAB-based visualization**.

        ---

        """)

    with col2:
        lottie_project = load_lottie("https://assets1.lottiefiles.com/packages/lf20_w51pcehl.json")
        if lottie_project:
            st_lottie(lottie_project, height=180, key="proj-anim")

    st.write("""
    ### 🔹 What It Does  
    - 🧩 **Process Simulation** → Predicts **layer height growth** using input parameters (Voltage, Current, Feedrate, Time).  
    - 📊 **Synthetic Dataset Generation** → Creates data for training and testing when experiments are limited.  
    - 🤖 **Machine Learning Models** → Trains models to capture the true deposition behavior, including variability and noise.  
    - ⚡ **Dynamic Control Features**:  
    - **Average Height Mode** → Keeps track of mean height and adjusts process parameters accordingly.  
    - **Constant Height Mode** → Uses an inverse model (Neural Network) to maintain a desired fixed height profile.  
    - 🎥 **3D MATLAB Animations** → Visualizes the deposition process in a more intuitive, interactive way.  

    ---

    ### 🔹 Why It’s Useful  
    - Helps engineers and researchers **understand process–parameter interactions**.  
    - Acts as a **Digital Twin** for deposition → enabling better monitoring, planning, and optimization.  
    - Reduces the dependency on expensive and time-consuming experimental trials.  

    ---

    ### 🔹 Key Technologies  
    - **Python + Streamlit** → Interactive Web UI  
    - **Scikit-learn & PyTorch** → Machine Learning & Neural Networks  
    - **MATLAB** → 3D visualization and simulation of deposition  
    - **Pandas & Excel Integration** → Data handling and export  

    ---

    ✨ In short, this project blends **Artificial Intelligence**, **Data Science**, and **Manufacturing Technology** to provide a smart, easy-to-use tool for simulating and controlling additive manufacturing processes.
    """)


# ----------------------
# About Developer Tab
# ----------------------
with tab6:
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

