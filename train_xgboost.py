import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.xgboost_model import train_xgboost, evaluate_model, save_model, print_feature_importance
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
    model = train_xgboost(X_train, y_train, X_val, y_val, params['xgboost'], verbose=False)
    
    # Оценка
    metrics, importance = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_feature_importance(importance, top_n=8)
    
    # Сохранение
    save_model(model, 'models/xgboost.json')
    save_metrics(metrics, 'xgboost')
    importance.to_csv('metrics/xgboost_features.csv', index=False)


if __name__ == "__main__":
    main()