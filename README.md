# AddiTrack
AI-driven framework for additive manufacturing that extracts layer height from video frames, builds a dataset with voltage, current, and feed rate, and trains a RandomForest model to simulate time vs. height growth curves with noise modeling. Enables digital twin insights for process optimization.
# 📌 AI-Driven Process–Height Correlation in Additive Manufacturing  

## 🔹 Overview  
This project establishes a **digital twin framework** for additive manufacturing by modeling the relationship between **layer height** and process parameters (**voltage, current, feed rate**).  
It extracts height data from deposition videos, builds a structured dataset, and trains a machine learning model to **predict and simulate layer growth** for unseen parameter combinations.  

---

## 🔹 Workflow  
1. **Video Frame Extraction** – Extract one frame per second from deposition videos using **OpenCV**.  
2. **Image Processing** – Detect annotated layer heights via **color masking and contour analysis**.  
3. **Dataset Generation** – Combine heights with process parameters (V, I, Feedrate, Time).  
4. **Model Training** – Fit growth curves and train a **RandomForestRegressor** in **scikit-learn** (R² > 0.9).  
5. **Simulation** – Generate realistic **Time vs Height** curves with noise modeling for digital twin applications.  

---

## 🔹 Tech Stack  
- **Python 3.x**  
- **OpenCV** – frame extraction & image processing  
- **NumPy / Pandas** – numerical computation & data handling  
- **SciPy** – curve fitting for growth modeling  
- **scikit-learn** – RandomForest regression & model evaluation  
- **Joblib** – model persistence  
- **Excel/CSV** – dataset storage & simulation output  

---

## 🔹 Repository Structure  
├── VideoToFrame.py # Extracts frames from deposition videos (1 fps)
├── TimeVsHeight.py # Processes frames → detects height → builds dataset
├── Training.py # Fits growth curves, trains RandomForest, simulates noisy height data
├── all_folders_results.xlsx # Example dataset (not uploaded here due to size)
└── README.md # Project documentation
