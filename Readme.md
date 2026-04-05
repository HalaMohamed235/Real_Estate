# Real Estate Price Prediction App

## Project Overview
This project is a modularized Machine Learning application that predicts house prices based on various property attributes. Originally developed as a Jupyter Notebook, the project has been refactored into a structured Python application and deployed using Streamlit Cloud.

The app allows users to input property details and receive price estimates using two different algorithms: Linear Regression and Random Forest.

## Live Demo
https://hala-realestate.streamlit.app/

## Features
- Dual Model Support: Compare predictions from Linear Regression and Random Forest models.
- Model Persistence: Models are saved and loaded using pickle for efficient inference.
- Interactive UI: A user-friendly Streamlit dashboard for real-time predictions.
- Robustness: Integrated logging and error handling for production-ready debugging.
- Clean Architecture: Fully modular code structure following software engineering best practices.

## Folder Structure

```text
Real_Estate/
├── app/
│   └── app.py              # Main Streamlit interface
├── data/
│   └── final.csv           # Project dataset
├── logs/
│   └── app.log             # Automated system logs
├── models/
│   ├── random_forest.pkl    # Serialized Random Forest model
│   └── linear_regression.pkl # Serialized Linear Regression model
├── src/
│   ├── __init__.py
│   ├── train_model.py      # Script for training and saving models
│   └── predict_model.py    # Functions for loading models and inference
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Input Features
The model makes predictions based on the following 13 features:
- Property Details: Square footage (sqft), Beds, Baths, Lot Size, Year Built.
- Financials: Property Tax, Insurance.
- Market Conditions: Year Sold, Recession (Boolean), Popularity index.
- Structural: Basement (Boolean), Property Age, Property Type (Condo).

## Installation & Usage

1. Clone the Repository:
   git clone https://github.com/HalaMohamed235/Real_Estate
   cd Real_Estate

2. Install Dependencies:
   pip install -r requirements.txt

3. Run the App Locally:
   streamlit run app/app.py

## Implementation Details
- Modularization: Logic is separated into training (train_model.py) and prediction (predict_model.py) to ensure the code is maintainable and reusable.
- Logging: The application tracks model loading and prediction events in logs/app.log.
- Deployment: The requirements.txt file ensures all necessary libraries (pandas, scikit-learn, streamlit) are installed automatically during deployment to Streamlit Cloud.