import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any

def train_linear_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    params: Dict[str, Any]
) -> LinearRegression:
    print("\n📈 Обучение линейной регрессии...")
    
    model = LinearRegression(
        fit_intercept=params['fit_intercept'],
        positive=params['positive']
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:

    # Предсказания
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Метрики для train
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    # Метрики для validation
    val_metrics = {
        'R2': r2_score(y_val, y_val_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred)
    }
    
    # Метрики для test
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Коэффициент': model.coef_
    })
    feature_importance = feature_importance.sort_values('Коэффициент', key=abs, ascending=False)
    
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    return metrics, feature_importance


def save_model(model: LinearRegression, save_path: str = 'models/linear_regression.pkl'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)

def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    print("\n Топ важности признаков:")
    print("-" * 55)
    
    top_features = feature_importance.head(top_n)
    for idx, row in top_features.iterrows():
        print(f"   {row['Признак']:35} {row['Коэффициент']:10.4f}")
 
   