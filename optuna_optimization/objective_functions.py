import optuna
import numpy as np
import joblib
import yaml


def load_data():
    """Загрузка подготовленных данных"""
    data = joblib.load('data/prepared_data.pkl')
    return (
        data['X_train'], data['X_val'], data['X_test'],
        data['y_train'], data['y_val'], data['y_test'],
        data['features']
    )


def objective_linear_regression(trial):
    """Целевая функция для линейной регрессии"""
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    
    from src.linear_regression import train_linear_regression, evaluate_model
    
    params = {
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'positive': trial.suggest_categorical('positive', [True, False]),
    }
    
    model = train_linear_regression(X_train, y_train, params)
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    return metrics['validation']['R2']


def objective_decision_tree(trial):
    """Целевая функция для дерева решений"""
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    
    from src.decision_tree import train_decision_tree, evaluate_model
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 30),
        'random_state': 42,
    }
    
    model = train_decision_tree(X_train, y_train, params)
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    return metrics['validation']['R2']


def objective_catboost(trial):
    """Целевая функция для CatBoost"""
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    
    from src.catboost_model import train_catboost, evaluate_model
    
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_seed': 42,
    }
    
    model = train_catboost(X_train, y_train, X_val, y_val, params, verbose=False)
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    return metrics['validation']['R2']


def objective_xgboost(trial):
    """Целевая функция для XGBoost"""
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    
    from src.xgboost_model import train_xgboost, evaluate_model
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
        'random_state': 42,
        'early_stopping_rounds': 20,
        'eval_metric': 'rmse',
    }
    
    model = train_xgboost(X_train, y_train, X_val, y_val, params, verbose=False)
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    return metrics['validation']['R2']


def objective_neural_network(trial):
    """Целевая функция для нейронной сети"""
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_data()
    
    from src.neural_network import train_neural_network, evaluate_model
    
    params = {
        'epochs': 200,
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'layer_1_neurons': trial.suggest_int('layer_1_neurons', 32, 256, step=32),
        'layer_2_neurons': trial.suggest_int('layer_2_neurons', 16, 128, step=16),
        'layer_3_neurons': trial.suggest_int('layer_3_neurons', 8, 64, step=8),
        'activation': 'relu',
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'patience': trial.suggest_int('patience', 10, 30),
        'random_state': 42,
    }
    
    model, _ = train_neural_network(X_train, y_train, X_val, y_val, params, verbose=False)
    metrics, _ = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, features)
    
    return metrics['validation']['R2']


# Словарь для выбора функции по модели
OBJECTIVES = {
    'linear_regression': objective_linear_regression,
    'decision_tree': objective_decision_tree,
    'catboost': objective_catboost,
    'xgboost': objective_xgboost,
    'neural_network': objective_neural_network,
}