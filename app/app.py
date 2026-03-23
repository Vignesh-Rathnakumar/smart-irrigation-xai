import streamlit as st
import joblib
import pandas as pd
import os
import shap
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

# ----------------------------
# Load Trained Model
# ----------------------------
model_path = os.path.join(os.path.dirname(__file__), "../models/best_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Smart Irrigation System", layout="centered")

st.title("🌾 Explainable Smart Irrigation System")
st.write("AI-based daily irrigation prediction using satellite climate data.")

mode = st.radio(
    "Select Input Mode",
    ["Manual Input", "Live Location-Based Prediction"]
)

# Initialize variables
temp = None
humidity = None
rainfall = None

# =====================================================
# MANUAL MODE
# =====================================================
if mode == "Manual Input":

    temp = st.number_input("Temperature (°C)", value=30.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.0)

# =====================================================
# LIVE LOCATION MODE
# =====================================================
else:

    location_input = st.text_input("Enter Village / Town / City (e.g., Madurai, Chennai)")

    if location_input:

        try:
            geolocator = Nominatim(user_agent="irrigation_app", timeout=10)
            location = geolocator.geocode(location_input + ", India")

            if location:

                latitude = location.latitude
                longitude = location.longitude

                st.success(f"📍 Location Found: {location.address}")
                st.write(f"Coordinates: {latitude:.4f}, {longitude:.4f}")

                # NASA daily data can lag → fetch last 7 days
                end_date = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

                url = (
                    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
                    f"parameters=T2M,RH2M,PRECTOTCORR"
                    f"&community=AG"
                    f"&longitude={longitude}"
                    f"&latitude={latitude}"
                    f"&start={start_date}"
                    f"&end={end_date}"
                    f"&format=JSON"
                )

                response = requests.get(url, timeout=10)

                if response.status_code == 200:

                    data = response.json()
                    climate_data = data["properties"]["parameter"]

                    df_live = pd.DataFrame({
                        "Temperature": climate_data["T2M"],
                        "Humidity": climate_data["RH2M"],
                        "Rainfall": climate_data["PRECTOTCORR"]
                    })

                    # Remove invalid values
                    df_live = df_live.replace(-999, pd.NA).dropna()

                    if df_live.empty:
                        st.error("No recent valid climate data available.")
                        st.stop()

                    # Take most recent valid record
                    latest_row = df_live.iloc[-1]

                    temp = latest_row["Temperature"]
                    humidity = latest_row["Humidity"]
                    rainfall = latest_row["Rainfall"]

                    st.write(f"🌡 Temperature: {temp} °C")
                    st.write(f"💧 Humidity: {humidity} %")
                    st.write(f"🌧 Rainfall: {rainfall} mm")

                else:
                    st.error("NASA API request failed.")
                    st.stop()

            else:
                st.error("Location not found. Please try another place.")
                st.stop()

        except Exception:
            st.error("Error fetching location or climate data.")
            st.stop()

# =====================================================
# PREDICTION
# =====================================================
if st.button("Predict Irrigation Requirement"):

    if temp is None or humidity is None or rainfall is None:
        st.warning("Please provide valid climate inputs first.")
        st.stop()

    # Automatic season detection
    month = datetime.now().month
    if month in [1, 2]:
        season = "Winter"
    elif month in [3, 4, 5]:
        season = "Summer"
    elif month in [6, 7, 8, 9]:
        season = "SW_Monsoon"
    else:
        season = "NE_Monsoon"

    # Prepare input dataframe
    expected_features = model.get_booster().feature_names
    input_df = pd.DataFrame(columns=expected_features)
    input_df.loc[0] = 0

    input_df.loc[0, "Temperature"] = temp
    input_df.loc[0, "Humidity"] = humidity
    input_df.loc[0, "Rainfall"] = rainfall

    season_column = f"Season_{season}"
    if season_column in input_df.columns:
        input_df.loc[0, season_column] = 1

    # Prediction
    prediction = model.predict(input_df)

    st.success(f"💧 Recommended Water: {prediction[0]:.2f} Liters per hectare")

    # -------------------------
    # SIMPLE EXPLAINABILITY
    # -------------------------
    st.subheader("📊 Irrigation Impact Analysis")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    contributions = dict(zip(input_df.columns, shap_values[0]))

    for feature, value in contributions.items():
        if feature in ["Temperature", "Humidity", "Rainfall"]:
            if value > 0:
                st.write(f"🔺 {feature} increased water requirement by {abs(value):.2f} liters")
            elif value < 0:
                st.write(f"🔻 {feature} reduced water requirement by {abs(value):.2f} liters")

    st.info("Seasonal influence is automatically included in prediction.")