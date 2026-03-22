import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Smart Crop System", layout="wide")

st.title("🌾 Smart Crop Recommendation & Yield Prediction")
st.markdown("### AI-powered agriculture decision system")

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    crop_model = XGBClassifier()
    crop_model.load_model("models/crop_model.json")

    yield_model = XGBRegressor()
    yield_model.load_model("models/yield_model.json")

    label_encoder = joblib.load("models/label_encoder.pkl")
    crop_columns = joblib.load("models/crop_feature_columns.pkl")
    yield_columns = joblib.load("models/yield_feature_columns.pkl")

    return crop_model, yield_model, label_encoder, crop_columns, yield_columns


crop_model, yield_model, label_encoder, crop_columns, yield_columns = load_models()

# ---------------------------
# WEATHER API
# ---------------------------
def get_weather(city):
    api_key = "324632143b2966f76a00fe77a0b3938d"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if data["cod"] != 200:
            st.warning(f"⚠️ Weather API error: {data.get('message', 'Unknown error')}")
            return 25, 70

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        return temp, humidity

    except Exception as e:
        st.error("Weather fetch failed")
        return 25, 70

# ---------------------------
# SIDEBAR INPUT
# ---------------------------
st.sidebar.header("🌱 Input Data")

# ALL STATES
states = [
    "Arunachal_Pradesh","Assam","Bihar","Chhattisgarh","Delhi","Goa",
    "Gujarat","Haryana","Himachal_Pradesh","Jammu_and_Kashmir",
    "Jharkhand","Karnataka","Kerala","Madhya_Pradesh","Maharashtra",
    "Manipur","Meghalaya","Mizoram","Nagaland","Odisha",
    "Puducherry","Punjab","Sikkim","Tamil_Nadu","Telangana",
    "Tripura","Uttar_Pradesh","Uttarakhand","West_Bengal"
]

state = st.sidebar.selectbox("Select State", states)
season = st.sidebar.selectbox("Select Season", ["Kharif", "Rabi", "Summer"])

# WEATHER
st.sidebar.markdown("### 🌦️ Climate")

city = st.sidebar.text_input("City", "Chennai")
use_weather = st.sidebar.checkbox("Auto-fill weather", True)

if use_weather:
    temperature, humidity = get_weather(city)

    st.sidebar.success("Auto-filled")
    st.sidebar.write(f"🌡️ Temp: {temperature:.2f} °C")
    st.sidebar.write(f"💧 Humidity: {humidity:.2f} %")
else:
    temperature = st.sidebar.slider("Temperature", 10.0, 40.0, 25.0)
    humidity = st.sidebar.slider("Humidity", 20.0, 100.0, 70.0)

# OTHER INPUTS
N = st.sidebar.slider("Nitrogen", 0, 140, 90)
P = st.sidebar.slider("Phosphorus", 0, 140, 40)
K = st.sidebar.slider("Potassium", 0, 140, 40)

ph = st.sidebar.slider("pH", 4.0, 9.0, 6.5)
rainfall = st.sidebar.slider("Rainfall", 0.0, 300.0, 200.0)
area = st.sidebar.slider("Area (hectare)", 1.0, 10.0, 2.0)

# VALIDATION
if ph < 4.5 or ph > 8.5:
    st.warning("⚠️ pH outside optimal range. Results may be unreliable.")

predict = st.sidebar.button("🚀 Predict")

# ---------------------------
# PREDICTION
# ---------------------------
def predict_all():

    crop_input = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    crop_input = crop_input[crop_columns]

    probs = crop_model.predict_proba(crop_input)
    top_idx = np.argsort(probs[0])[-3:][::-1]

    crops = label_encoder.inverse_transform(top_idx)
    probs = probs[0][top_idx]

    results = []

    for crop, prob in zip(crops, probs):

        row = pd.DataFrame(columns=yield_columns)
        row.loc[0] = 0

        # numeric
        row["Area"] = area
        row["Annual_Rainfall"] = rainfall
        row["Fertilizer"] = 120
        row["Pesticide"] = 30
        row["Crop_Year"] = 2020

        # state
        state_col = f"State_{state}"
        if state_col in row.columns:
            row[state_col] = 1

        # season
        season_col = f"Season_{season}"
        if season_col in row.columns:
            row[season_col] = 1

        # crop encoding (robust)
        for col in row.columns:
            if col.startswith("Crop_") and crop.lower() in col.lower():
                row[col] = 1

        log_y = yield_model.predict(row)
        y = np.expm1(log_y)[0]

        # profit (simple assumption)
        price = 20
        profit = y * price

        results.append((crop, prob, y, profit))

    return results

# ---------------------------
# OUTPUT
# ---------------------------
if predict:

    results = predict_all()

    st.subheader("🌟 Top Recommendations")

    cols = st.columns(3)

    for i, (crop, prob, y, profit) in enumerate(results):

        with cols[i]:
            st.markdown(f"""
            <div style="background:#1f2937;padding:20px;border-radius:15px;color:white;text-align:center">
                <h3 style="color:#22c55e">{crop}</h3>
                <p>🌱 Suitability: {prob*100:.2f}%</p>
                <p>🌾 Yield: {y:.2f}</p>
                <p>💰 Profit: {profit:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    # CONFIDENCE GRAPH
    st.subheader("📊 Model Confidence")

    chart_df = pd.DataFrame({
        "Crop": [c for c,_,_,_ in results],
        "Confidence": [p for _,p,_,_ in results]
    })

    st.bar_chart(chart_df.set_index("Crop"))

    # INSIGHT
    best = max(results, key=lambda x: x[2])
    st.info(f"💡 Insight: {best[0]} gives highest yield under current conditions.")

    # DOWNLOAD
    df = pd.DataFrame(results, columns=["Crop", "Probability", "Yield", "Profit"])

    st.download_button(
        "📥 Download Report",
        df.to_csv(index=False),
        "crop_report.csv",
        "text/csv"
    )

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Built using XGBoost Ensemble Learning 🌾")