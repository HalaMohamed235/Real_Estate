import pandas as pd
import pickle
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging (inside this file only)
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def load_data():
    try:
        df = pd.read_csv("data/final.csv")
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def train_linear_regression(X_train, y_train):
    try:
        lr = LinearRegression()
        lr_model = lr.fit(X_train, y_train)
        logging.info("Linear Regression model trained successfully.")
        return lr_model
    except Exception as e:
        logging.error(f"Failed to train Linear Regression: {e}")
        raise

def train_random_forest(X_train, y_train, n_estimators=200):
    try:
        rf = RandomForestRegressor(n_estimators=n_estimators, criterion='absolute_error', random_state=42)
        rf_model = rf.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        return rf_model
    except Exception as e:
        logging.error(f"Failed to train Random Forest: {e}")
        raise

def evaluate_model(model, X, y, model_name="Model"):
    try:
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        logging.info(f"{model_name} MAE: {mae}")
        return mae
    except Exception as e:
        logging.error(f"Failed to evaluate {model_name}: {e}")
        raise

def save_model(model, filename):
    try:
        os.makedirs("models", exist_ok=True)
        with open(f"models/{filename}", "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved as models/{filename}")
    except Exception as e:
        logging.error(f"Failed to save model {filename}: {e}")
        raise

def main():
    df = load_data()

    # Features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=X.get("property_type_Condo", None), random_state=42
        )
        logging.info(f"Train/Test split done: {X_train.shape}, {X_test.shape}")
    except Exception as e:
        logging.error(f"Train/test split failed: {e}")
        raise

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate
    evaluate_model(lr_model, X_train, y_train, "Linear Regression Train")
    evaluate_model(rf_model, X_train, y_train, "Random Forest Train")
    evaluate_model(lr_model, X_test, y_test, "Linear Regression Test")
    evaluate_model(rf_model, X_test, y_test, "Random Forest Test")

    # Save models
    save_model(lr_model, "linear_regression.pkl")
    save_model(rf_model, "random_forest.pkl")

if __name__ == "__main__":
    main()