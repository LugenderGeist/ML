import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import warnings
warnings.filterwarnings('ignore')

from src.linear_regression import train_linear_regression, evaluate_model as evaluate_lr, save_model as save_lr, print_feature_importance as print_lr_importance
from src.decision_tree import train_decision_tree, evaluate_model as evaluate_dt, save_model as save_dt, visualize_tree, print_feature_importance as print_dt_importance
from src.catboost_model import train_catboost, evaluate_model as evaluate_cb, save_model as save_cb, print_feature_importance as print_cb_importance
from src.xgboost_model import train_xgboost, evaluate_model as evaluate_xgb, save_model as save_xgb, print_feature_importance as print_xgb_importance
from src.neural_network import train_neural_network, evaluate_model as evaluate_nn, save_model as save_nn, print_feature_importance as print_nn_importance
from prepare_data import normalize_features, save_scaler, add_normalization_params, heatmap
from src.utils import save_metrics, select_all_features, prepare_data_for_training

def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return add_normalization_params(params)

def main():
    print("=" * 80)
    print("🏥 АНАЛИЗ ДАННЫХ И ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    # Параметры
    try:
        params = load_params()
    except FileNotFoundError:
        print("⚠️ Файл params.yaml не найден, использую параметры по умолчанию")
        params = {
            'general': {'random_state': 42, 'train_size': 0.7, 'val_size': 0.2, 'test_size': 0.1, 'correlation_threshold': 0.3},
            'preprocessing': {'normalize': True, 'normalization_method': 'standard'},
            'linear_regression': {'fit_intercept': True, 'positive': False},
            'decision_tree': {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42},
            'catboost': {'iterations': 800, 'learning_rate': 0.03, 'depth': 9, 'l2_leaf_reg': 3.0, 'random_seed': 42},
            'xgboost': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 9, 'subsample': 0.8, 
                       'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42,
                       'early_stopping_rounds': 20, 'eval_metric': 'rmse'},
            'neural_network': {'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001, 
                              'layer_1_neurons': 64, 'layer_2_neurons': 32, 'layer_3_neurons': 16,
                              'activation': 'relu', 'dropout_rate': 0.2, 'patience': 20, 'random_state': 42}
        }
    
    # Обработка данных
    file_path = 'data/cancer_reg1.csv'
    df = pd.read_csv(file_path)

    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)

    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)

    heatmap(df, save_path='plots/full_correlation.png')

    features = select_all_features(df, target='Смерти/д.н.')

    if features is None or len(features) == 0:
        print("❌ Нет признаков для обучения!")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_training(
        df, features,
        train_size=params['general']['train_size'],
        val_size=params['general']['val_size'],
        test_size=params['general']['test_size'],
        random_state=params['general']['random_state']
    )

    if params['preprocessing']['normalize']:
        print("\n" + "=" * 80)
        print("🔄 НОРМАЛИЗАЦИЯ ДАННЫХ")
        print("=" * 80)
        
        X_train, X_val, X_test, scaler = normalize_features(
            X_train, X_val, X_test,
            method=params['preprocessing']['normalization_method']
        )
        save_scaler(scaler, 'models/scaler.pkl')
    else:
        print("\n⚠️ Нормализация отключена")
    
    # Обучение моделей
    
    # ЛИНЕЙНАЯ РЕГРЕССИЯ
    print("\n" + "=" * 80)
    print("📈 ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 80)
    
    model_lr = train_linear_regression(X_train, y_train, params['linear_regression'])
    metrics_lr, importance_lr = evaluate_lr(model_lr, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_lr_importance(importance_lr, top_n=8)
    save_lr(model_lr, 'models/linear_regression.pkl')
    save_metrics(metrics_lr, 'linear_regression')
    importance_lr.to_csv('metrics/linear_regression_features.csv', index=False)
    
    # ДЕРЕВО РЕШЕНИЙ 
    print("\n" + "=" * 80)
    print("🌳 ДЕРЕВО РЕШЕНИЙ")
    print("=" * 80)
    
    model_dt = train_decision_tree(X_train, y_train, params['decision_tree'])
    metrics_dt, importance_dt = evaluate_dt(model_dt, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_dt_importance(importance_dt, top_n=8)
    visualize_tree(model_dt, features, max_depth=3, save_path='plots/decision_tree.png')
    save_dt(model_dt, 'models/decision_tree.pkl')
    save_metrics(metrics_dt, 'decision_tree')
    importance_dt.to_csv('metrics/decision_tree_features.csv', index=False)
    
    # CATBOOST
    print("\n" + "=" * 80)
    print("🐱 CATBOOST")
    print("=" * 80)
    
    model_cb = train_catboost(X_train, y_train, X_val, y_val, params['catboost'], verbose=False)
    metrics_cb, importance_cb = evaluate_cb(model_cb, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_cb_importance(importance_cb, top_n=8)
    save_cb(model_cb, 'models/catboost.cbm')
    save_metrics(metrics_cb, 'catboost')
    importance_cb.to_csv('metrics/catboost_features.csv', index=False)
    
    # XGBOOST
    print("\n" + "=" * 80)
    print("⚡ XGBOOST")
    print("=" * 80)
    
    model_xgb = train_xgboost(X_train, y_train, X_val, y_val, params['xgboost'], verbose=False)
    metrics_xgb, importance_xgb = evaluate_xgb(model_xgb, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_xgb_importance(importance_xgb, top_n=8)
    save_xgb(model_xgb, 'models/xgboost.json')
    save_metrics(metrics_xgb, 'xgboost')
    importance_xgb.to_csv('metrics/xgboost_features.csv', index=False)
    
    # НЕЙРОННАЯ СЕТЬ
    print("\n" + "=" * 80)
    print("🧠 НЕЙРОННАЯ СЕТЬ")
    print("=" * 80)

    model_nn, _ = train_neural_network(X_train, y_train, X_val, y_val, params['neural_network'], verbose=False)
    metrics_nn, importance_nn = evaluate_nn(model_nn, X_train, X_val, X_test, y_train, y_val, y_test, features)
    print_nn_importance(importance_nn, top_n=8)
    save_nn(model_nn, 'models/neural_network.keras')
    save_metrics(metrics_nn, 'neural_network')
    
    # Сравнение моделей
    print("\n" + "=" * 80)
    print("🏆 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Модель': ['Линейная регрессия', 'Дерево решений', 'CatBoost', 'XGBoost', 'Нейронная сеть'],
        'R² (test)': [
            metrics_lr['test']['R2'], 
            metrics_dt['test']['R2'],
            metrics_cb['test']['R2'],
            metrics_xgb['test']['R2'],
            metrics_nn['test']['R2']
        ],
        'RMSE (test)': [
            metrics_lr['test']['RMSE'], 
            metrics_dt['test']['RMSE'],
            metrics_cb['test']['RMSE'],
            metrics_xgb['test']['RMSE'],
            metrics_nn['test']['RMSE']
        ],
        'MAE (test)': [
            metrics_lr['test']['MAE'], 
            metrics_dt['test']['MAE'],
            metrics_cb['test']['MAE'],
            metrics_xgb['test']['MAE'],
            metrics_nn['test']['MAE']
        ]
    })
    comparison = comparison.sort_values('R² (test)', ascending=False)
    
    print("\n" + comparison.to_string(index=False))
    comparison.to_csv('metrics/models_comparison.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)

if __name__ == "__main__":
    main()