# 🌾 Smart Crop Recommendation & Yield Prediction System

An AI-powered agriculture decision support system that combines **XGBoost ensemble learning** with **real-time weather data** to recommend the best crops for a given field and predict their expected yield and estimated profit.

---

## 📌 Overview

Farmers and agronomists often make crop decisions based on intuition or tradition — ignoring soil chemistry, climate, and market potential. This system addresses that gap by combining machine learning with live weather integration to deliver:

- **Top-3 crop recommendations** ranked by suitability probability
- **Yield predictions** for each recommended crop
- **Profit estimates** based on predicted yield
- **Downloadable reports** in CSV format

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌱 Crop Recommendation | XGBoost classifier predicts the top-N best-fit crops for given soil and climate conditions |
| 🌾 Yield Prediction | XGBoost regressor estimates expected yield (log-transformed target, inverse-transformed for output) |
| 🌦️ Live Weather | Auto-fills temperature and humidity via OpenWeatherMap API based on user's city |
| 💰 Profit Estimation | Simple profitability estimate derived from predicted yield |
| 📊 Confidence Chart | Bar chart showing model confidence for each recommended crop |
| 📥 Report Download | One-click CSV export of predictions |
| 🗺️ State & Season Aware | Supports 29 Indian states and Kharif / Rabi / Summer seasons |

---

## 🛠️ Tech Stack

- **ML Framework:** XGBoost (`XGBClassifier`, `XGBRegressor`)
- **Web App:** Streamlit
- **Data Processing:** NumPy, Pandas, Scikit-learn (LabelEncoder, joblib)
- **Weather API:** OpenWeatherMap
- **Language:** Python 3.x

---

## 📁 Project Structure

```
├── app.py                  # Streamlit web application
├── data/                   # Raw and processed datasets
├── models/
│   ├── crop_model.json     # Trained XGBoost crop classifier
│   ├── yield_model.json    # Trained XGBoost yield regressor
│   ├── label_encoder.pkl   # Crop label encoder
│   ├── crop_feature_columns.pkl
│   └── yield_feature_columns.pkl
├── notebooks/              # Jupyter notebooks for EDA, training, and evaluation
├── report.txt              # (Reserved for model reports)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhi24112006/crop-recommendation_and_yield_prediction_system_using_ensemble_learning_XGBoost.git
cd crop-recommendation_and_yield_prediction_system_using_ensemble_learning_XGBoost
```

### 2. Install dependencies

```bash
pip install streamlit xgboost scikit-learn pandas numpy requests joblib
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🚀 How to Use

1. **Select your state and season** from the sidebar dropdowns.
2. **Enter your city** and enable *Auto-fill weather* to pull live temperature and humidity — or set them manually with sliders.
3. **Adjust soil parameters:** Nitrogen (N), Phosphorus (P), Potassium (K), pH, and Rainfall.
4. **Set the farm area** in hectares.
5. Click **🚀 Predict** to see:
   - Top 3 recommended crops with suitability scores
   - Estimated yield and profit for each
   - A confidence bar chart
6. **Download the report** as a CSV with one click.

---

## 📥 Input Parameters

| Parameter | Range | Description |
|---|---|---|
| State | 29 Indian states | Used for one-hot encoding in yield model |
| Season | Kharif / Rabi / Summer | Cropping season |
| City | Any city name | Used to fetch live weather data |
| Nitrogen (N) | 0 – 140 | Soil nitrogen content (kg/ha) |
| Phosphorus (P) | 0 – 140 | Soil phosphorus content (kg/ha) |
| Potassium (K) | 0 – 140 | Soil potassium content (kg/ha) |
| pH | 4.0 – 9.0 | Soil pH level |
| Rainfall | 0 – 300 mm | Expected rainfall |
| Area | 1 – 10 ha | Cultivated farm area |
| Temperature | 10 – 40 °C | Auto-filled or manual |
| Humidity | 20 – 100 % | Auto-filled or manual |

> ⚠️ A warning is shown if pH falls outside the optimal range (4.5–8.5), as predictions may be less reliable.

---

## 🧠 Model Details

### Crop Recommendation Model
- **Type:** `XGBClassifier`
- **Input:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Output:** Probability distribution over crop classes; top-3 crops returned
- **Encoding:** Crops encoded with `LabelEncoder`, probabilities extracted via `predict_proba`

### Yield Prediction Model
- **Type:** `XGBRegressor`
- **Input:** Area, Annual Rainfall, Fertilizer, Pesticide, Crop Year, one-hot encoded State, Season, and Crop
- **Output:** Predicted yield (log-transformed during training, converted back via `np.expm1`)

### Training Notebooks
Refer to the `notebooks/` directory for exploratory data analysis, feature engineering, model training, and evaluation.

---

## 🌍 Supported States

The app supports all 29 Indian states including Tamil Nadu, Karnataka, Maharashtra, Uttar Pradesh, Punjab, and more — used as one-hot features in the yield model.

---

## 📊 Output Example

```
Top Recommendations:
┌─────────────┬──────────────┬────────────┬──────────────┐
│ Crop        │ Suitability  │ Yield      │ Profit (₹)   │
├─────────────┼──────────────┼────────────┼──────────────┤
│ Rice        │ 87.34%       │ 3245.12    │ 64,902.40    │
│ Maize       │ 9.21%        │ 2108.55    │ 42,171.00    │
│ Wheat       │ 3.45%        │ 1987.30    │ 39,746.00    │
└─────────────┴──────────────┴────────────┴──────────────┘
💡 Insight: Rice gives the highest yield under current conditions.
```

---

## 🔑 Weather API

This project uses the [OpenWeatherMap API](https://openweathermap.org/api). The API key is hardcoded in `app.py` for convenience. For production use, move it to an environment variable:

```python
import os
api_key = os.environ.get("OPENWEATHER_API_KEY")
```

---

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

- Add more crops and regional datasets
- Improve profit estimation with real market price APIs
- Add SHAP-based feature importance explanations
- Support multilingual UI (Hindi, Tamil, etc.)
- Deploy to Streamlit Cloud or Hugging Face Spaces

To contribute: fork the repo, make your changes on a new branch, and open a pull request.

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute it with attribution.

---

## 👤 Author

**Abhi** — [GitHub @abhi24112006](https://github.com/abhi24112006)

---

*Built with ❤️ using XGBoost Ensemble Learning for smarter, data-driven agriculture.*
