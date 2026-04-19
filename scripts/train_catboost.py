import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.catboost_model import train_catboost, evaluate_model, save_model, print_feature_importance
from src.utils import save_metrics

def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def main():
    
    # Загрузка параметров
    params = load_params()
    
    # Загрузка подготовленных данных
    prepared_data = joblib.load('data/prepared_data.pkl')
    X_train = prepared_data['X_train']
    X_val = prepared_data['X_val']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_val = prepared_data['y_val']
    y_test = prepared_data['y_test']
    features = prepared_data['features']
    
    # Обучение
    model = train_catboost(X_train, y_train, X_val, y_val, params['catboost'], verbose=False)
    
    # Оценка
    metrics, importance = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_feature_importance(importance, top_n=8)
    
    # Сохранение
    save_model(model, 'models/catboost.cbm')
    save_metrics(metrics, 'catboost')
    importance.to_csv('metrics/catboost_features.csv', index=False)

def print_detailed_importance(importance, top_n=15):
    """Детальный вывод важности признаков"""
    print("\n FEATURE IMPORTANCE (CatBoost):")
    print("=" * 70)
    print(f"{'Признак':<35} {'Важность':>10} {'Вклад %':>10}")
    print("-" * 70)
    
    total = importance['Важность'].sum()
    for idx, row in importance.head(top_n).iterrows():
        percent = (row['Важность'] / total) * 100
        print(f"{row['Признак']:<35} {row['Важность']:>10.4f} {percent:>9.1f}%")
    
    # Сохраняем с процентами
    importance['Вклад_процент'] = (importance['Важность'] / total)
    importance.to_csv('metrics/catboost_features_with_percent.csv', index=False)
    print(f"\n Сохранено: metrics/catboost_features_with_percent.csv")

if __name__ == "__main__":
    main()