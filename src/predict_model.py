import pickle
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def load_model():
    """
    Load both Linear Regression and Random Forest models.
    Returns:
        lr_model: LinearRegression model
        rf_model: RandomForestRegressor model
    """
    try:
        with open("models/linear_regression.pkl", "rb") as f:
            lr_model = pickle.load(f)

        with open("models/random_forest.pkl", "rb") as f:
            rf_model = pickle.load(f)

        return lr_model, rf_model

    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise e


def predict(model, input_df):
    """
    Predict using a given model and input DataFrame
    """
    try:
        return model.predict(input_df)
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise e