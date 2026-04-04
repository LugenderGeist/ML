import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List


def save_metrics(metrics: dict, model_name: str, save_path: str = 'metrics/'):
    """Сохранение метрик модели в JSON файл"""
    os.makedirs(save_path, exist_ok=True)
    
    # Преобразуем numpy типы в Python типы для JSON
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
    
    print(f"✅ Метрики сохранены: {save_path}/{model_name}_metrics.json")


def print_metrics_table(metrics: dict, model_name: str):
    """Краткий вывод метрик"""
    print(f"\n{model_name}:")
    print(f"  R² на тесте: {metrics['test']['R2']:.4f}")
    print(f"  RMSE на тесте: {metrics['test']['RMSE']:.4f}")
    print(f"  MAE на тесте: {metrics['test']['MAE']:.4f}")


def select_features_for_training(
    df: pd.DataFrame, 
    target: str = 'Смерти/д.н.', 
    correlation_threshold: float = 0.3
) -> Tuple[List[str], pd.Series]:
    """Отбор признаков для обучения на основе корреляции с целевой переменной"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f"❌ Целевая переменная '{target}' не найдена!")
        return None, None
    
    correlations = numeric_df.corr()[target].abs().sort_values(ascending=False)
    selected_features = correlations[correlations > correlation_threshold].index.tolist()
    
    if target in selected_features:
        selected_features.remove(target)
    
    print(f"🎯 Отобрано {len(selected_features)} признаков с корреляцией > {correlation_threshold*100:.0f}%")
    
    return selected_features, correlations


def prepare_data_for_training(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = 'Смерти/д.н.', 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """Подготовка данных для обучения"""
    from sklearn.model_selection import train_test_split
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Train: {X_train.shape[0]} строк, Test: {X_test.shape[0]} строк")
    
    return X_train, X_test, y_train, y_test