# 🌾 Explainable AI for Smart Irrigation
### Using Soil Moisture and Climate Data | Tamil Nadu, India

> **Published at:** 7th International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV 2026)  
> **Authors:** Vignesh R, Janani C, Dhyanesh M — SRM Institute of Science and Technology

---

## 📌 Overview

This project presents an **Explainable AI (XAI) based Smart Irrigation Prediction System** that combines satellite climate data with soil properties to predict daily irrigation water requirements. The system uses **XGBoost** as the best-performing model and **SHAP** for explainability, deployed via a **Streamlit** web application.

---

## 🏗️ Project Structure

```
smart-irrigation-xai/
├── app/
│   └── app.py                    # Streamlit web application (3 tabs)
├── data/
│   └── irrigation_dataset.csv    # Spatial climate + soil dataset
├── models/
│   └── best_model.pkl            # Trained XGBoost model
├── notebooks/
│   ├── 01_spatial_grid.ipynb
│   ├── 02_climate_data_collection.ipynb
│   ├── 03_soil_assignment_fao.ipynb
│   ├── 04_build_final_dataset.ipynb
│   ├── 05_model_training_spatial.ipynb
│   ├── 06_explainability_analysis.ipynb
│   ├── train_advanced_models.ipynb   # Extended model comparison
│   └── lstm_irrigation.ipynb         # LSTM deep learning comparison
├── requirements.txt
└── README.md
```

---

## ✨ Features

- 🛰️ **NASA POWER API** — Real satellite climate data (temperature, humidity, rainfall)
- 🤖 **8 ML Models Compared** — Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVR, MLP
- 🧠 **LSTM Deep Learning** — Time-series comparison with XGBoost
- 📊 **SHAP Explainability** — Feature contribution analysis for every prediction
- 📍 **Live Location Mode** — Enter any city/village to get real-time climate-based predictions
- 📈 **Real-Time Simulation** — Hourly sensor simulation with 4-panel dashboard
- 🗂️ **3-Tab Streamlit App** — Prediction, Simulation, Model Comparison

---

## 📊 Model Performance

| Model | R² Score | CV R² | MAE | RMSE |
|-------|----------|-------|-----|------|
| Linear Regression | 0.7400 | 0.7200 | 1820 | 2341 |
| Ridge Regression | 0.7612 | 0.7401 | 1743 | 2198 |
| Decision Tree | 0.9995 | 0.9947 | 48 | 61 |
| Random Forest | 0.9999 | 0.9951 | 22 | 29 |
| Gradient Boosting | 0.9996 | 0.9962 | 38 | 48 |
| **XGBoost ✅** | **0.9998** | **0.9973** | **31** | **40** |
| SVR (RBF) | 0.9201 | 0.9134 | 812 | 1054 |
| MLP Regressor | 0.9843 | 0.9801 | 189 | 241 |

> XGBoost selected as best model based on CV R² and overall robustness.

---

## 🧠 LSTM vs XGBoost

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| XGBoost | 0.9998 | 31 | 40 |
| LSTM (1 Layer) | 0.9712 | 312 | 418 |
| LSTM (2 Layers) | 0.9801 | 241 | 321 |
| Bi-LSTM | 0.9834 | 198 | 274 |

> XGBoost outperforms LSTM on this structured tabular dataset. LSTM may improve with dense IoT sensor streams.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Vignesh-Rathnakumar/smart-irrigation-xai.git
cd smart-irrigation-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app/app.py
```

---

## 📦 Requirements

```
streamlit
xgboost
scikit-learn
shap
pandas
numpy
matplotlib
seaborn
joblib
geopy
requests
lightgbm
tensorflow
```

> Full pinned versions in `requirements.txt`

---

## 🗂️ Dataset

- **Source:** NASA POWER API (satellite-based climate data)
- **Coverage:** 30 spatial grid points across Tamil Nadu, India
- **Period:** 2020–2024 (daily resolution)
- **Features:** Temperature, Humidity, Rainfall, Latitude, Longitude, Season, Soil Type
- **Soil Types:** Sandy, Red, Alluvial, Black, Clay (FAO classification)
- **Target:** Irrigation water requirement (litres/hectare)

---

## 🔬 Methodology

```
NASA POWER Climate Data  ──►  Spatial Grid (30 locations)
         +
FAO Soil Classification  ──►  Soil Retention Factors
         │
         ▼
ET₀ = 0.5T − 0.3(RH/100)       ← Evapotranspiration formula
Water_mm = max(0, ET₀ − Rainfall)
Water_litres = Water_mm × SoilFactor × 10,000
         │
         ▼
One-Hot Encoding (Soil Type + Season)
         │
         ▼
8 ML Models + LSTM  ──►  XGBoost (Best)
         │
         ▼
SHAP Explainability  ──►  Feature Contributions
```

---

## 🖥️ App Tabs

| Tab | Description |
|-----|-------------|
| 🔍 Prediction & Explainability | Manual or live location input, irrigation prediction, SHAP chart |
| 📈 Real-Time Simulation | Simulated hourly sensor data with 4-panel dashboard |
| 📊 Model Comparison | All model metrics, LSTM results, encoding strategy tables |

---

## 📄 Citation

If you use this work, please cite:

```
R.Vignesh, C. Janani, and M. Dhyanesh, "Explainable AI for Smart 
Irrigation Using Soil Moisture and Climate Data," in Proceedings of the 
7th International Conference on Intelligent Communication Technologies 
and Virtual Mobile Networks (ICICV 2026), May 2026.
```

---

## 📬 Contact

| Name | Email |
|------|-------|
| Vignesh R | vr8677@srmist.edu.in |
| Janani C | jc2433@srmist.edu.in |
| Dhyanesh M | dm1963@srmist.edu.in |

---

## 📝 License

This project is for academic research purposes. Please contact the authors before reuse.
