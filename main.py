import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from src.correlation_plots import plot_full_correlation_heatmap, plot_high_correlation_heatmap
from src.linear_regression import train_linear_regression, evaluate_model as evaluate_lr, save_model as save_lr, print_feature_importance as print_lr_importance
from src.decision_tree import train_decision_tree, evaluate_model as evaluate_dt, save_model as save_dt, visualize_tree, print_feature_importance as print_dt_importance
from src.catboost_model import train_catboost, evaluate_model as evaluate_cb, save_model as save_cb, print_feature_importance as print_cb_importance
from src.xgboost_model import train_xgboost, evaluate_model as evaluate_xgb, save_model as save_xgb, print_feature_importance as print_xgb_importance
from src.neural_network import train_neural_network, evaluate_model as evaluate_nn, save_model as save_nn, print_feature_importance as print_nn_importance
from src.utils import save_metrics, print_metrics_table, select_features_for_training, prepare_data_for_training


def load_params(config_path='config/params.yaml'):
    """Загрузка гиперпараметров из YAML файла"""
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def main():
    """Основная функция - анализ данных и обучение моделей"""
    print("=" * 80)
    print("🏥 АНАЛИЗ ДАННЫХ И ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    # ==================== ШАГ 1: ЗАГРУЗКА ПАРАМЕТРОВ ====================
    try:
        params = load_params()
    except FileNotFoundError:
        print("⚠️ Файл config/params.yaml не найден, использую параметры по умолчанию")
        params = {
            'general': {'random_state': 42, 'test_size': 0.2, 'correlation_threshold': 0.3},
            'linear_regression': {'fit_intercept': True, 'positive': False},
            'decision_tree': {'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 10, 'random_state': 42},
            'catboost': {'iterations': 500, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3.0, 'random_seed': 42},
            'xgboost': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8, 
                       'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42},
            'neural_network': {'hidden_layer_sizes': [64, 32, 16], 'activation': 'relu', 'solver': 'adam',
                              'alpha': 0.0001, 'batch_size': 32, 'learning_rate_init': 0.001,
                              'max_iter': 500, 'early_stopping': True, 'validation_fraction': 0.1,
                              'n_iter_no_change': 20, 'random_state': 42}
        }
    
    # ==================== ШАГ 2: ЗАГРУЗКА ДАННЫХ ====================
    file_path = 'data/cancer_reg1.csv'
    df = pd.read_csv(file_path)
    print(f"\n✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # ==================== ШАГ 3: УДАЛЕНИЕ ПРОПУСКОВ ====================
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    
    if rows_removed > 0:
        print(f"🗑️ Удалено пропусков: {rows_removed} строк")
    else:
        print("✅ Пропусков в данных нет")
    
    # ==================== ШАГ 4: СОЗДАНИЕ ПАПОК ====================
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # ==================== ШАГ 5: ТЕПЛОВЫЕ КАРТЫ ====================
    correlation_matrix = plot_full_correlation_heatmap(df, save_path='plots/full_correlation.png')
    high_corr_features = plot_high_correlation_heatmap(
        df,
        threshold=params['general']['correlation_threshold'],
        save_path='plots/high_correlation.png'
    )
    
    # ==================== ШАГ 6: ОТБОР ПРИЗНАКОВ ====================
    features, _ = select_features_for_training(
        df, 
        correlation_threshold=params['general']['correlation_threshold']
    )
    
    if features is None or len(features) == 0:
        print("❌ Нет признаков для обучения!")
        return
    
    # Сохраняем список признаков
    with open('models/features.json', 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    # ==================== ШАГ 7: ПОДГОТОВКА ДАННЫХ ====================
    X_train, X_test, y_train, y_test = prepare_data_for_training(
        df, features,
        test_size=params['general']['test_size'],
        random_state=params['general']['random_state']
    )
    
    # ==================== ШАГ 8: ОБУЧЕНИЕ МОДЕЛЕЙ ====================
    all_metrics = {}
    
    # 8.1 ЛИНЕЙНАЯ РЕГРЕССИЯ
    print("\n" + "=" * 80)
    print("📈 ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 80)
    
    model_lr = train_linear_regression(X_train, y_train, params['linear_regression'])
    metrics_lr, importance_lr = evaluate_lr(model_lr, X_train, X_test, y_train, y_test, features)
    print_metrics_table(metrics_lr, "Линейная регрессия")
    print_lr_importance(importance_lr, top_n=8)
    save_lr(model_lr, 'models/linear_regression.pkl')
    save_metrics(metrics_lr, 'linear_regression')
    importance_lr.to_csv('metrics/linear_regression_features.csv', index=False)
    all_metrics['linear_regression'] = metrics_lr
    
    # 8.2 ДЕРЕВО РЕШЕНИЙ
    print("\n" + "=" * 80)
    print("🌳 ДЕРЕВО РЕШЕНИЙ")
    print("=" * 80)
    
    model_dt = train_decision_tree(X_train, y_train, params['decision_tree'])
    metrics_dt, importance_dt = evaluate_dt(model_dt, X_train, X_test, y_train, y_test, features)
    print_metrics_table(metrics_dt, "Дерево решений")
    print_dt_importance(importance_dt, top_n=8)
    visualize_tree(model_dt, features, max_depth=3, save_path='plots/decision_tree.png')
    save_dt(model_dt, 'models/decision_tree.pkl')
    save_metrics(metrics_dt, 'decision_tree')
    importance_dt.to_csv('metrics/decision_tree_features.csv', index=False)
    all_metrics['decision_tree'] = metrics_dt
    
    # 8.3 CATBOOST
    print("\n" + "=" * 80)
    print("🐱 CATBOOST")
    print("=" * 80)
    
    model_cb = train_catboost(X_train, y_train, params['catboost'], verbose=False)
    metrics_cb, importance_cb = evaluate_cb(model_cb, X_train, X_test, y_train, y_test, features)
    print_metrics_table(metrics_cb, "CatBoost")
    print_cb_importance(importance_cb, top_n=8)
    save_cb(model_cb, 'models/catboost.cbm')
    save_metrics(metrics_cb, 'catboost')
    importance_cb.to_csv('metrics/catboost_features.csv', index=False)
    all_metrics['catboost'] = metrics_cb
    
    # 8.4 XGBOOST
    print("\n" + "=" * 80)
    print("⚡ XGBOOST")
    print("=" * 80)
    
    model_xgb = train_xgboost(X_train, y_train, params['xgboost'], verbose=False)
    metrics_xgb, importance_xgb = evaluate_xgb(model_xgb, X_train, X_test, y_train, y_test, features)
    print_metrics_table(metrics_xgb, "XGBoost")
    print_xgb_importance(importance_xgb, top_n=8)
    save_xgb(model_xgb, 'models/xgboost.json')
    save_metrics(metrics_xgb, 'xgboost')
    importance_xgb.to_csv('metrics/xgboost_features.csv', index=False)
    all_metrics['xgboost'] = metrics_xgb
    
    # 8.5 НЕЙРОННАЯ СЕТЬ (sklearn)
    print("\n" + "=" * 80)
    print("🧠 НЕЙРОННАЯ СЕТЬ (MLP)")
    print("=" * 80)
    
    model_nn, scaler_nn = train_neural_network(X_train, y_train, params['neural_network'], verbose=False)
    metrics_nn, importance_nn = evaluate_nn(model_nn, scaler_nn, X_train, X_test, y_train, y_test, features)
    print_metrics_table(metrics_nn, "Нейронная сеть")
    print_nn_importance(importance_nn, top_n=8)
    save_nn(model_nn, scaler_nn, 'models/neural_network.pkl')
    save_metrics(metrics_nn, 'neural_network')
    all_metrics['neural_network'] = metrics_nn
    
    # ==================== ШАГ 9: СРАВНЕНИЕ МОДЕЛЕЙ ====================
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
    
    # ==================== ИТОГИ ====================
    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    
    best_model = comparison.iloc[0]['Модель']
    best_r2 = comparison.iloc[0]['R² (test)']
    print(f"\n🥇 Лучшая модель: {best_model} (R² = {best_r2:.4f})")


if __name__ == "__main__":
    main()