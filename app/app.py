import streamlit as st
import pandas as pd
import sys
import logging
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_model import load_model, predict


# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    filename="logs/app.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

st.title("Real Estate Price Prediction")

st.write(
    "Enter the details of the property to predict the price using Linear Regression and Random Forest."
)

# -------------------------------
# Input Widgets for Features
# -------------------------------
def user_input_features():
    try:
        year_sold = st.number_input("Year Sold", min_value=1800, max_value=2025, value=2006)
        property_tax = st.number_input("Property Tax ($)", min_value=0, max_value=100000, value=397)
        insurance = st.number_input("Insurance ($)", min_value=0, max_value=50000, value=102)
        beds = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        baths = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        sqft = st.number_input("Square Footage", min_value=100, max_value=90000, value=1500)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=1986)
        lot_size = st.number_input("Lot Size", min_value=100, max_value=90000, value=11325)
        basement = st.selectbox("Basement (1=Yes, 0=No)", [0,1])
        popular = st.selectbox("Popular Area (1=Yes, 0=No)", [0,1])
        recession = st.selectbox("Built During Recession (1=Yes, 0=No)", [0,1])
        property_age = st.number_input("Property Age", min_value=0, max_value=300, value=20)
        property_type_Condo = st.selectbox("Property Type Condo (1=Yes, 0=No)", [0,1])

        data = {
            'year_sold': year_sold,
            'property_tax': property_tax,
            'insurance': insurance,
            'beds': beds,
            'baths': baths,
            'sqft': sqft,
            'year_built': year_built,
            'lot_size': lot_size,
            'basement': basement,
            'popular': popular,
            'recession': recession,
            'property_age': property_age,
            'property_type_Condo': property_type_Condo
        }

        features = pd.DataFrame([data])
        return features

    except Exception as e:
        logging.error(f"Error in input widgets: {e}")
        st.error("An error occurred while creating input fields.")

# -------------------------------
# Main Prediction
# -------------------------------
input_df = user_input_features()

if st.button("Predict Price"):
    try:
        # Load both models
        lr_model, rf_model = load_model()  # your function should return both models

        # Make predictions
        lr_pred = predict(lr_model, input_df)
        rf_pred = predict(rf_model, input_df)

        st.subheader("Predicted Prices")
        st.write(f"Linear Regression Prediction: ${lr_pred[0]:,.2f}")
        st.write(f"Random Forest Prediction: ${rf_pred[0]:,.2f}")

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("An error occurred during prediction. Check error.log for details.")