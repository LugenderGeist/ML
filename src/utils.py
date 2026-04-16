import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def save_metrics(metrics: dict, model_name: str, save_path: str = 'metrics/'):
    os.makedirs(save_path, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_serializable[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            metrics_serializable[key] = convert_to_serializable(value)
    
    with open(f'{save_path}/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)


def select_all_features(
    df: pd.DataFrame, 
    target: str = 'Смерти/д.н.'
) -> List[str]:
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f"❌ Целевая переменная '{target}' не найдена!")
        return None
    
    # Берем все числовые признаки, кроме целевой переменной
    selected_features = [col for col in numeric_df.columns if col != target]
    
    return selected_features

def prepare_data_for_training(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = 'Смерти/д.н.', 
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
):

    X = df[features]
    y = df[target]
    
    # Сначала отделяем тестовую выборку (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Затем из оставшихся 90% отделяем валидацию (20% от исходных)
    val_relative_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test