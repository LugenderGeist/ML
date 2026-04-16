import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Dict, Any
import joblib
import os


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Нормализация признаков
    
    Parameters:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        Данные для нормализации
    method : str
        Метод нормализации: 'standard', 'minmax', 'robust'
    
    Returns:
    --------
    X_train_norm, X_val_norm, X_test_norm : pd.DataFrame
        Нормализованные данные
    scaler : object
        Обученный скейлер
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'minmax', or 'robust'")
    
    # Обучаем скейлер только на тренировочных данных!
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразуем обратно в DataFrame для сохранения имен признаков
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_scaler(scaler: object, save_path: str = 'models/scaler.pkl'):
    """Сохранение скейлера"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"💾 Скейлер сохранен: {save_path}")


def load_scaler(load_path: str = 'models/scaler.pkl') -> object:
    """Загрузка скейлера"""
    return joblib.load(load_path)


def add_normalization_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Добавляет параметры нормализации в словарь параметров"""
    if 'preprocessing' not in params:
        params['preprocessing'] = {}
    
    defaults = {
        'normalize': True,           # Включить нормализацию
        'normalization_method': 'standard'  # standard, minmax, robust
    }
    
    for key, value in defaults.items():
        if key not in params['preprocessing']:
            params['preprocessing'][key] = value
    
    return params