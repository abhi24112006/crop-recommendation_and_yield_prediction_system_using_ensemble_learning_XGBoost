# Crop Recommendation & Yield Prediction System
## Overview

This project presents an AI-driven agricultural decision support system that recommends the most suitable crops and predicts expected yield based on soil and climatic conditions. The system leverages ensemble learning techniques, primarily XGBoost, to achieve high accuracy and robust performance.

The model not only predicts a single crop but provides a Top-N recommendation, making it more practical for real-world farming scenarios.

## Features
**Crop Recommendation (Top-N)**
Suggests the top 3 most suitable crops based on input conditions.

**Yield Prediction**
Predicts expected crop yield using regression modeling.

**Ensemble Learning (XGBoost)**
High-performance model for both classification and regression.

**Weather Integration (Optional)**
Auto-fetch temperature and humidity using API.

**Interactive Dashboard (Streamlit)**
User-friendly interface for real-time predictions.

**Ablation Study & Explainability**
Analyzes feature importance and model behavior.

## Tech Stack
1. Language: Python
2. Libraries:
pandas, numpy
scikit-learn
xgboost
matplotlib, seaborn
3. Frontend: Streamlit
4. Model Storage: joblib / XGBoost JSON

## Dataset

Two datasets were used:

**1. Crop Recommendation Dataset**

Features:

N, P, K (soil nutrients)
Temperature
Humidity
pH
Rainfall
Crop Label

**2. Crop Yield Dataset**

Features:

Crop, State, Season
Area, Production
Rainfall, Fertilizer, Pesticide
Yield (target variable)
## Installation
1. git clone https://github.com/your-username/crop-capstone-project.git
2. cd crop-capstone-project
3. pip install -r requirements.txt

**Run the Application:**
streamlit run app.py
## Sample Input
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 26,
  "humidity": 80,
  "ph": 6.5,
  "rainfall": 200
}
## Sample Output
Top 3 Recommended Crops:

1. rice → 95% suitability
2. jute → 3% suitability
3. coffee → 1% suitability

Predicted Yield: ~0.95 tons/hectare

 **Model Performance**
1. Classification (Crop Recommendation)
2. Accuracy: 99.4% (XGBoost)
3. F1 Score: 0.994
4. Regression (Yield Prediction)
5. R² Score: 0.957
6. RMSE: 0.234
## Key Insights
Climate factors like humidity and rainfall strongly influence crop prediction.

Crop type is the most important feature in yield prediction.

Ensemble models significantly outperform traditional models.

## Limitations
Dataset imbalance affects prediction of certain crops.

Some crops (e.g., sugarcane) are not included in dataset.

Overlapping conditions for fruits may reduce prediction confidence.

## Future Work
Add more crop varieties and real-time soil data

Integrate IoT sensors for live monitoring

Implement SHAP for explainability

Add profit optimization module