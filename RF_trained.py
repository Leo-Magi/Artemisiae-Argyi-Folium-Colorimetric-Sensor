import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        X = data.drop(columns=[data.columns[0], data.columns[1]])
        y = data[data.columns[1]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Data preprocessed successfully.")
        return X_scaled, y
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def evaluate_model(model, X_train, X_test, y_train, y_test):
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_train = sqrt(mse_train)
        rmse_test = sqrt(mse_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        logger.info("Model evaluated successfully.")
        return mse_train, rmse_train, mae_train, r2_train, mse_test, rmse_test, mae_test, r2_test
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

try:
    # Load data
    data = load_data('')

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=70)

    # Set parameter grid
    param_grid = {
        'n_estimators': [ 130,140,150,160],
        'max_depth': [3,5,7,10,15,20],
        'min_samples_split': [2,3,4],
        'min_samples_leaf': [1],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=70),
                               param_grid=param_grid,
                               cv=5,
                               scoring='r2',
                               verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_rf = grid_search.best_estimator_

    # Evaluate
    mse_train, rmse_train, mae_train, r2_train, mse_test, rmse_test, mae_test, r2_test = evaluate_model(
        best_rf, X_train, X_test, y_train, y_test)

    # Output results
    print("Best parameters:", grid_search.best_params_)
    print("Training MSE:", mse_train)
    print("Training RMSE:", rmse_train)
    print("Training MAE:", mae_train)
    print("Training R²:", r2_train)
    print("Test MSE:", mse_test)
    print("Test RMSE:", rmse_test)
    print("Test MAE:", mae_test)
    print("Test R²:", r2_test)

except Exception as e:
    print(f"An error occurred: {e}")
