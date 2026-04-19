import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import warnings
warnings.filterwarnings('ignore')

from src.linear_regression import train_linear_regression, evaluate_model, save_model, print_feature_importance
from src.utils import save_metrics, print_metrics_table

def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def save_equation(model, features, intercept=True, save_path='metrics/linear_regression_equation.txt'):
    # Формируем уравнение
    equation = ""
    if intercept:
        equation += f"{model.intercept_:.4f}"
    
    for name, coef in zip(features, model.coef_):
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.6f} * {name}"
    
    # Сохраняем в файл
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
    
    print(f" Уравнение сохранено: {save_path}")
    
    # Выводим в консоль
    print("\n УРАВНЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ:")
    print("-" * 60)
    print(f"Смерти/д.н. = {equation}")
    
    return equation


def main():
    print("=" * 80)
    print(" ЛИНЕЙНАЯ РЕГРЕССИЯ")
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
    
    # Оценка
    metrics, importance = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_metrics_table(metrics, "Линейная регрессия")
    print_feature_importance(importance, top_n=8)
    
    # Сохранение
    save_model(model, 'models/linear_regression.pkl')
    save_metrics(metrics, 'linear_regression')
    importance.to_csv('metrics/linear_regression_features.csv', index=False)


if __name__ == "__main__":
    main()