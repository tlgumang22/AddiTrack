# ⚙️ AddiTrack  
**AI-driven framework for Additive Manufacturing (AM)** that extracts **layer height** from deposition video frames, builds a dataset with **voltage, current, feed rate, and time**, and trains a **RandomForest model** to simulate **time vs. height growth curves** with **noise modeling**.  
This enables **digital twin insights** for process optimization in 3D printing.  

---

## 📌 Overview  
This project establishes a **digital twin framework** for additive manufacturing by modeling the relationship between **layer height** and **process parameters**.  
It extracts height data from deposition videos, builds a structured dataset, and trains a machine learning model to **predict and simulate layer growth** for unseen parameter combinations.  

**Key Highlights:**  
- Extracts frames from deposition videos at 1 fps.  
- Processes frames to detect **layer height** using OpenCV (color masking + contour analysis).  
- Builds datasets combining **process parameters (V, I, Feedrate, Time)** with extracted height.  
- Trains **RandomForestRegressor** (scikit-learn) with R² > 0.9 on experimental data.  
- Adds **noise modeling** to mimic real-world deposition variation.  
- Integrates with **MATLAB 3D animation** for visualization of the deposition process.  

---

## 🔹 Workflow  
1. **🎥 Video Frame Extraction** – Extract one frame per second using OpenCV.  
2. **🖼️ Image Processing** – Detect annotated layer heights via color masking + contour analysis.  
3. **📊 Dataset Generation** – Build dataset with (Voltage, Current, Feedrate, Time, Height).  
4. **🤖 Model Training** – Train RandomForest + fit growth curves (scikit-learn + SciPy).  
5. **📈 Simulation** – Generate noisy **Time vs. Height** curves for digital twin insights.  
6. **🧩 Visualization** – Run MATLAB 3D simulation for process visualization.  

---

## 🔹 Tech Stack  
- **Python 3.x**  
- **OpenCV** – frame extraction & image processing  
- **NumPy / Pandas** – numerical computation & data handling  
- **SciPy** – curve fitting for growth modeling  
- **scikit-learn** – RandomForest regression & model evaluation  
- **Joblib / Pickle** – model persistence  
- **Streamlit** – interactive web app UI  
- **MATLAB** – 3D deposition process visualization  
- **Excel/CSV** – dataset storage & simulation output  

---

## 🔹 Repository Structure  

```bash
BTP/
│── README.md                 # Project documentation
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignore venv, data, outputs
│
├── src/                       # Source code
│   ├── app.py                 # Streamlit app (main entry point)
│   ├── Training.py            # Model training & simulation
│   ├── VideoToFrame.py        # Extract frames from deposition videos
│   ├── TimeVsHeight.py        # Frame → height dataset builder
│   ├── Graph.py               # Plotting utilities
│   ├── ContantHeight.py       # (To be renamed → ConstantHeight.py)
│   ├── temp.py       # (To be renamed → ConstantHeight.py)
│
├── data/                      # Raw & processed datasets
│   ├── all_folders_results.xlsx
│   ├── simulated_series_*.xlsx
│   ├── constant_height_profile.xlsx
│
├── models/                    # Saved models & noise stats
│   ├── curve_param_model.pkl
│   ├── noise_stats.json
│
├── extra/                    # Static assets for app
│   ├── your_photo.jpg
│
├── matlab/                    # MATLAB scripts
    ├── depositionAnimation.m