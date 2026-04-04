import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any


def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    params: Dict[str, Any],
    verbose: bool = False
) -> xgb.XGBRegressor:
    """
    Обучение модели XGBoost
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Обучающая выборка признаков
    y_train : pd.Series
        Обучающая выборка целевой переменной
    params : dict
        Параметры модели из params.yaml
    verbose : bool
        Показывать ли вывод обучения
    
    Returns:
    --------
    xgb.XGBRegressor
        Обученная модель
    """
    print("\n⚡ Обучение XGBoost...")
    
    # Создаем модель с параметрами
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=params['random_state'],
        verbose=verbose
    )
    
    # Обучаем
    model.fit(X_train, y_train)
    
    # У XGBoost атрибут называется n_estimators (без подчеркивания)
    print(f"✅ Модель обучена (деревьев: {model.n_estimators})")
    
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Оценка качества модели
    """
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики для train
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    # Метрики для test
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # Важность признаков (Feature Importance)
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Важность': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Важность', ascending=False)
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return metrics, feature_importance


def save_model(model: xgb.XGBRegressor, save_path: str = 'models/xgboost.json'):
    """
    Сохранение модели в файл
    
    XGBoost рекомендует использовать .json формат для сохранения
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    print(f"💾 Модель сохранена: {save_path}")


def load_model(load_path: str = 'models/xgboost.json') -> xgb.XGBRegressor:
    """Загрузка модели из файла"""
    model = xgb.XGBRegressor()
    model.load_model(load_path)
    return model


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод топ-N наиболее важных признаков для XGBoost"""
    print("\n📊 Топ важности признаков (XGBoost Feature Importance):")
    print("-" * 55)
    
    top_features = feature_importance.head(top_n)
    
    for idx, row in top_features.iterrows():
        print(f"   {row['Признак']:35} {row['Важность']:10.4f}")
    
    print(f"\n   💡 XGBoost Feature Importance показывает вклад каждого признака")
    print(f"   (сумма всех важностей = {feature_importance['Важность'].sum():.2f})")