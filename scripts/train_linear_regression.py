import os
import sys
sys.path.append('C:/Users/Александра/vs_code/ML')

import joblib
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from src.linear_regression import train_linear_regression, evaluate_model, save_model, get_feature_importance, print_feature_importance
from src.utils import save_metrics, print_metrics_table


def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def save_equation(model, features, intercept=True, save_path='metrics/linear_regression_equation.txt'):
    """Сохранение уравнения линейной регрессии в файл"""
    equation = ""
    if intercept:
        equation += f"{model.intercept_:.4f}"
    
    for name, coef in zip(features, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.6f} * {name}"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("УРАВНЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Смерти/д.н. = {equation}\n\n")
        f.write("-" * 70 + "\n")
        f.write("КОЭФФИЦИЕНТЫ:\n")
        f.write("-" * 70 + "\n")
        for name, coef in zip(features, model.coef_):
            f.write(f"{name:35} {coef:+.6f}\n")
        f.write(f"\n{'Intercept':35} {model.intercept_:+.6f}\n")
    
    print(f"✅ Уравнение сохранено: {save_path}")
    
    print("\n📐 УРАВНЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ:")
    print("-" * 60)
    print(f"Смерти/д.н. = {equation}")
    
    return equation


def main():
    print("=" * 80)
    print("📈 ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 80)
    
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
    model = train_linear_regression(X_train, y_train, params['linear_regression'])
    
    # Сохранение уравнения
    save_equation(model, features, intercept=params['linear_regression']['fit_intercept'])
    
    # Расчет важности признаков (относительные вклады)
    feature_importance = get_feature_importance(model, features)
    
    # Сохраняем в CSV
    feature_importance.to_csv('metrics/linear_regression_features.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ Важность признаков сохранена: metrics/linear_regression_features.csv")
    
    # Выводим в консоль
    print_feature_importance(feature_importance, top_n=10)
    
    # Оценка модели
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_metrics_table(metrics, "Линейная регрессия")
    
    # Сохранение метрик
    save_model(model, 'models/linear_regression.pkl')
    save_metrics(metrics, 'linear_regression')


if __name__ == "__main__":
    main()