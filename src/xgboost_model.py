import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    verbose: bool = False
) -> xgb.XGBRegressor:

    print("\n⚡ Обучение XGBoost...")
    
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=params['random_state'],
        early_stopping_rounds=params.get('early_stopping_rounds', 20),
        eval_metric=params.get('eval_metric', 'rmse'),
        verbose=verbose
    )
    
    # Обучаем с валидационной выборкой
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
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
        'Важность': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Важность', ascending=False)
    
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return metrics, feature_importance


def save_model(model: xgb.XGBRegressor, save_path: str = 'models/xgboost.json'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)

def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    
    print("\n📊 Топ важности признаков (XGBoost Feature Importance):")
    print("-" * 55)
    
    top_features = feature_importance.head(top_n)
    
    for idx, row in top_features.iterrows():
        print(f"   {row['Признак']:35} {row['Важность']:10.4f}")