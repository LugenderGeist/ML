import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def save_metrics(metrics: dict, model_name: str, save_path: str = 'metrics/'):
    """Сохранение метрик модели в JSON файл"""
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
    
    print(f"✅ Метрики сохранены: {save_path}/{model_name}_metrics.json")


def print_metrics_table(metrics: dict, model_name: str):
    """Краткий вывод метрик"""
    print(f"\n{model_name}:")
    print(f"  R² на тесте: {metrics['test']['R2']:.4f}")
    print(f"  RMSE на тесте: {metrics['test']['RMSE']:.4f}")
    print(f"  MAE на тесте: {metrics['test']['MAE']:.4f}")
    
    # Если есть валидационные метрики
    if 'validation' in metrics:
        print(f"\n  Валидация:")
        print(f"    R²: {metrics['validation']['R2']:.4f}")
        print(f"    RMSE: {metrics['validation']['RMSE']:.4f}")
        print(f"    MAE: {metrics['validation']['MAE']:.4f}")


def select_all_features(
    df: pd.DataFrame, 
    target: str = 'Смерти/д.н.'
) -> List[str]:
    """
    Используем ВСЕ числовые признаки (без автоматического отбора)
    Ты сама решишь, какие удалить, посмотрев на тепловую карту!
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f"❌ Целевая переменная '{target}' не найдена!")
        return None
    
    # Берем все числовые признаки, кроме целевой переменной
    selected_features = [col for col in numeric_df.columns if col != target]
    
    print(f"\n🎯 Используем ВСЕ числовые признаки: {len(selected_features)}")
    print(f"   (Ты можешь удалить ненужные вручную после просмотра тепловой карты)")
    
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
    """
    Подготовка данных для обучения с разделением на train/validation/test
    """
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
    
    print(f"\n📊 Разделение данных (train/val/test = {train_size:.0%}/{val_size:.0%}/{test_size:.0%}):")
    print(f"   Train: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
    print(f"   Validation: {X_val.shape[0]} строк, {X_val.shape[1]} признаков")
    print(f"   Test: {X_test.shape[0]} строк, {X_test.shape[1]} признаков")
    
    return X_train, X_val, X_test, y_train, y_val, y_test