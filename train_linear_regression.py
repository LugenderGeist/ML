import pandas as pd
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