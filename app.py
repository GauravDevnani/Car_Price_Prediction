import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

try:
    model = pk.load(open('model.pkl', 'rb'))
    cars_data = pd.read_csv("Cardetails.csv")
except FileNotFoundError as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

st.title('Car Price Prediction ML Model')

def get_brand_name(car_name):
    if not isinstance(car_name, str) or not car_name:
        return "Unknown"
    try:
        return car_name.split(' ')[0].strip().lower()
    except IndexError:
        return "Unknown"

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox("Select Car Brand", cars_data['name'].unique())
year = st.slider("Car Manufactured Year", 1994, 2024)
km_driven = st.slider("No. of kms driven", 11, 200000)
fuel = st.selectbox("Fuel Type", cars_data['fuel'].unique())
seller_type = st.selectbox("Seller Type", cars_data['seller_type'].unique())
transmission = st.selectbox("Transmission Type", cars_data['transmission'].unique())
owner = st.selectbox("Owner", cars_data['owner'].unique())
mileage = st.slider("Car Mileage", 10, 40)
engine = st.slider("Engine CC", 700, 5000)
max_power = st.slider("Max Power", 0, 2000)
seats = st.slider("No. of seats", 5, 10)

if st.button("Predict"):
    input_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    # Preprocessing (Crucial - Adapt to your model's training)
    # Example: One-Hot Encoding (Highly Recommended)
    input_data = pd.get_dummies(input_data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

    # Example: Scaling (If your model was trained on scaled data)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler() # replace with your scaler
    # numerical_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    # input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Ensure all columns match the model's expected columns
    try:
        prediction = model.predict(input_data)
        st.markdown(f"Car price is going to be ₹{prediction[0]:,.2f}")
    except ValueError as e:
        st.error(f"Prediction Error: {e}. Please ensure the input data is correct and matches the model's expected format.")
