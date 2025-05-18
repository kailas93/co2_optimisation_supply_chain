import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os

st.set_page_config(page_title="Green Logistics Optimizer", layout="wide")

st.title("🚛 Carbon Footprint Optimizer with Route Optimization")

OPENWEATHER_API_KEY = "df013fcd7e7e4aee7ff6a234c6c81621"
GOOGLE_MAPS_API_KEY = "AIzaSyB3i8YnGjpF6O0Wt5N5HQ2bTAf_f6TluRI"

@st.cache_resource
def load_model():
    return joblib.load("carbon_rf_model.pkl")

def extract_distance(value):
    try:
        return float(''.join(filter(str.isdigit, str(value))))
    except:
        return np.nan

def clean_mpg(value):
    try:
        return float(value)
    except:
        return np.nan

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        return {
            "Temperature": data["main"]["temp"],
            "Weather": data["weather"][0]["description"],
            "Wind Speed": data["wind"]["speed"]
        }
    return {}
#api_key = GOOGLE_MAPS_API_KEY
# def get_route_alternatives(origin, destination, api_key):
#     url = (
#         f"https://maps.googleapis.com/maps/api/directions/json"
#         f"?origin={origin}&destination={destination}&alternatives=true&key={api_key}"
#     )
#     r = requests.get(url)
#     routes = []
#     if r.status_code == 200:
#         data = r.json()
#         for route in data.get("routes", []):
#             leg = route["legs"][0]
#             routes.append({
#                 "summary": route.get("summary", "Unnamed"),
#                 "distance_km": leg["distance"]["value"] / 1000,
#                 "duration_min": leg["duration"]["value"] / 60,
#                 "start_location": leg["start_location"],
#                 "end_location": leg["end_location"]
#             })
#     return routes
api_key = "AIzaSyB3i8YnGjpF6O0Wt5N5HQ2bTAf_f6TluRI"
def get_route_alternatives(origin, destination,api_key):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={origin}&destination={destination}&alternatives=true&key={api_key}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error fetching routes: HTTP {r.status_code}")
        return []

    data = r.json()

    if data.get("status") != "OK":
        st.error(f"Google Maps Error: {data.get('status')} - {data.get('error_message', '')}")
        return []

    # proceed if OK
    routes = []
    for route in data.get("routes", []):
        leg = route["legs"][0]
        routes.append({
            "summary": route.get("summary", "Unnamed"),
            "distance_km": leg["distance"]["value"] / 1000,
            "duration_min": leg["duration"]["value"] / 60,
            "start_location": leg["start_location"],
            "end_location": leg["end_location"]
        })
    return routes


def estimate_emissions_for_routes(routes, avg_mpg, avg_type_code, model):
    results = []
    for r in routes:
        miles = r["distance_km"] / 1.60934
        est_fuel = miles / avg_mpg
        X_input = pd.DataFrame([{
            "Fuel": est_fuel,
            "Distance": r["distance_km"] * 1000,
            "MPG": avg_mpg,
            "Type_encoded": avg_type_code
        }])
        r["predicted_emissions"] = model.predict(X_input)[0]
        results.append(r)
    return sorted(results, key=lambda x: x["predicted_emissions"])

def predict_emissions(df, model):
    df['Distance'] = df['Dist.Run'].apply(extract_distance)
    df['MPG'] = df['MPG'].apply(clean_mpg)
    df.dropna(subset=['Fuel', 'Distance', 'MPG'], inplace=True)
    df['Type_encoded'] = df['Type'].astype('category').cat.codes
    df['CO2_Emissions'] = df['Fuel'] * 2.68
    X = df[['Fuel', 'Distance', 'MPG', 'Type_encoded']]
    df['Predicted_Emissions'] = model.predict(X)
    return df

uploaded_file = st.file_uploader("📂 Upload Fleet CSV", type="csv")

with st.sidebar:
    st.header("📍 Route Inputs")
    origin = st.text_input("Origin", value="Leeds, UK")
    destination = st.text_input("Destination", value="Manchester, UK")

    if st.button("🚦 Optimize Route"):
        model = load_model()
        routes = get_route_alternatives(origin, destination, GOOGLE_MAPS_API_KEY)

        if routes:
            avg_mpg = 10
            avg_type_code = 0
            optimized = estimate_emissions_for_routes(routes, avg_mpg, avg_type_code, model)

            st.subheader("📊 Route Comparison")
            st.dataframe(pd.DataFrame(optimized)[["summary", "distance_km", "duration_min", "predicted_emissions"]])

            best = optimized[0]
            st.success(f"✅ Best Route: {best['summary']} — {best['predicted_emissions']:.2f} kg CO₂")

            weather_origin = get_weather(best['start_location']['lat'], best['start_location']['lng'])
            weather_dest = get_weather(best['end_location']['lat'], best['end_location']['lng'])

            st.subheader("🌦 Weather at Origin")
            st.write(weather_origin)
            st.subheader("🌦 Weather at Destination")
            st.write(weather_dest)
        else:
            st.error("No route found.")

if uploaded_file:
    model = load_model()
    df = pd.read_csv(uploaded_file)
    result = predict_emissions(df, model)

    st.subheader("📈 Fleet Emissions Prediction")
    st.dataframe(result[['Fuel', 'Distance', 'MPG', 'Type', 'Predicted_Emissions']].head(10))
    st.download_button("📥 Download Results", result.to_csv(index=False), "emissions_predictions.csv")
