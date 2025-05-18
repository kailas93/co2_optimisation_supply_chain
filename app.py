# Updated app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

st.set_page_config(page_title="Green Logistics Optimizer", layout="wide")

st.title("\U0001F69B Carbon Footprint Optimizer with Route Optimization")

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

def get_emission_factor(vehicle):
    vehicle = vehicle.upper()
    emission_factors = {
        'SMALL VAN': 2.6,
        'LARGE VAN': 2.8,
        'CAR': 2.3,
        'MINIBUS': 3.0,
        'HGV': 3.2,
        'TRACTOR': 3.5,
        'SWEEPER': 3.3,
        'TIPPER': 3.4,
        'HOOKLOADER': 3.6,
    }
    for key in emission_factors:
        if key in vehicle:
            return emission_factors[key]
    return 2.68

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

def get_route_alternatives(origin, destination, api_key):
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

def estimate_emissions_for_routes(routes, avg_mpg, vehicle_type, model):
    emission_factor = get_emission_factor(vehicle_type)
    results = []
    for r in routes:
        miles = r["distance_km"] / 1.60934
        est_fuel = miles / avg_mpg
        X_input = pd.DataFrame([{
            "Fuel": est_fuel,
            "Distance": r["distance_km"] * 1000,
            "MPG": avg_mpg,
            "Type_encoded": 0
        }])
        r["predicted_emissions"] = model.predict(X_input)[0]
        r["adjusted_emissions"] = est_fuel * emission_factor
        results.append(r)
    return sorted(results, key=lambda x: x["adjusted_emissions"])

def predict_emissions(df, model):
    df['Distance'] = df['Dist.Run'].apply(extract_distance)
    df['MPG'] = df['MPG'].apply(clean_mpg)
    df.dropna(subset=['Fuel', 'Distance', 'MPG'], inplace=True)
    df['Type_encoded'] = df['Type'].astype('category').cat.codes
    df['Emission_Factor'] = df['Vehicle'].astype(str).str.strip().str.upper().apply(get_emission_factor)
    df['CO2_Emissions'] = df['Fuel'] * df['Emission_Factor']
    X = df[['Fuel', 'Distance', 'MPG', 'Type_encoded']]
    df['Predicted_Emissions'] = model.predict(X)
    return df

uploaded_file = st.file_uploader("\U0001F4C2 Upload Fleet CSV", type="csv")

with st.sidebar:
    st.header("\U0001F4CD Route Inputs")
    origin = st.text_input("Origin", value="Leeds, UK")
    destination = st.text_input("Destination", value="Manchester, UK")
    vehicle_type = st.text_input("Vehicle Type", value="Small Van")
    avg_mpg = st.number_input("Average MPG", value=10.0)

    if st.button("\U0001F6A6 Optimize Route"):
        model = load_model()
        routes = get_route_alternatives(origin, destination, GOOGLE_MAPS_API_KEY)

        if routes:
            optimized = estimate_emissions_for_routes(routes, avg_mpg, vehicle_type, model)

            st.subheader("\U0001F4CA Route Comparison")
            st.dataframe(pd.DataFrame(optimized)[["summary", "distance_km", "duration_min", "predicted_emissions", "adjusted_emissions"]])

            best = optimized[0]
            st.success(f"✅ Best Route: {best['summary']} — {best['adjusted_emissions']:.2f} kg CO₂ (adjusted)")

            weather_origin = get_weather(best['start_location']['lat'], best['start_location']['lng'])
            weather_dest = get_weather(best['end_location']['lat'], best['end_location']['lng'])

            st.subheader("\U0001F326 Weather at Origin")
            st.write(weather_origin)
            st.subheader("\U0001F326 Weather at Destination")
            st.write(weather_dest)
        else:
            st.error("No route found.")

if uploaded_file:
    model = load_model()
    df = pd.read_csv(uploaded_file)
    result = predict_emissions(df, model)

    st.subheader("\U0001F4C8 Fleet Emissions Prediction")
    st.dataframe(result[['Fuel', 'Distance', 'MPG', 'Type', 'Predicted_Emissions', 'CO2_Emissions']].head(10))
    st.download_button("\U0001F4E5 Download Results", result.to_csv(index=False), "emissions_predictions.csv")
