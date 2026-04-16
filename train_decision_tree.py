import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.decision_tree import train_decision_tree, evaluate_model, save_model, visualize_tree, print_feature_importance
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
    model = train_decision_tree(X_train, y_train, params['decision_tree'])
    
    # Оценка
    metrics, importance = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_feature_importance(importance, top_n=8)
    
    # Визуализация
    visualize_tree(model, features, max_depth=3, save_path='plots/decision_tree.png')
    
    # Сохранение
    save_model(model, 'models/decision_tree.pkl')
    save_metrics(metrics, 'decision_tree')
    importance.to_csv('metrics/decision_tree_features.csv', index=False)


if __name__ == "__main__":
    main()