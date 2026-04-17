import pandas as pd
import numpy as np
import json
import os

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
    
    print(f"✅ Метрики сохранены: {save_path}/{model_name}_metrics.json")


def print_metrics_table(metrics: dict, model_name: str):
    print(f"\n{model_name}:")
    print(f"  R² на тесте: {metrics['test']['R2']:.4f}")
    print(f"  RMSE на тесте: {metrics['test']['RMSE']:.4f}")
    print(f"  MAE на тесте: {metrics['test']['MAE']:.4f}")
    
    if 'validation' in metrics:
        print(f"\n  Валидация:")
        print(f"    R²: {metrics['validation']['R2']:.4f}")
        print(f"    RMSE: {metrics['validation']['RMSE']:.4f}")
        print(f"    MAE: {metrics['validation']['MAE']:.4f}")