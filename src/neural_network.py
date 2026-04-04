import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    verbose: bool = False
) -> Tuple[MLPRegressor, StandardScaler]:
    """
    Обучение нейронной сети (MLPRegressor)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Обучающая выборка признаков
    y_train : pd.Series
        Обучающая выборка целевой переменной
    params : dict
        Параметры модели из params.yaml
    verbose : bool
        Показывать ли вывод обучения
    
    Returns:
    --------
    MLPRegressor
        Обученная модель
    StandardScaler
        Обученный scaler для нормализации данных
    """
    print("\n🧠 Обучение нейронной сети (MLP)...")
    
    # Нормализация данных (важно для нейросетей!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Создаем модель
    model = MLPRegressor(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        batch_size=params['batch_size'],
        learning_rate_init=params['learning_rate_init'],
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping'],
        validation_fraction=params['validation_fraction'],
        n_iter_no_change=params['n_iter_no_change'],
        random_state=params['random_state'],
        verbose=verbose
    )
    
    # Обучаем
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ Модель обучена (итераций: {model.n_iter_})")
    print(f"   Потери на обучении: {model.loss_:.4f}")
    
    return model, scaler


def evaluate_model(
    model: MLPRegressor,
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Оценка качества модели
    """
    # Нормализация данных
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Предсказания
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Метрики для train
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    # Метрики для test
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # Для нейросети sklearn можно получить важность признаков через веса
    # Но это сложно интерпретировать
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Важность': 0  # Placeholder
    })
    
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return metrics, feature_importance


def save_model(model: MLPRegressor, scaler: StandardScaler, save_path: str = 'models/neural_network.pkl'):
    """
    Сохранение модели и scaler в файл
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Сохраняем модель и scaler вместе
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, save_path)
    print(f"💾 Модель и scaler сохранены: {save_path}")


def load_model(load_path: str = 'models/neural_network.pkl') -> Tuple[MLPRegressor, StandardScaler]:
    """Загрузка модели и scaler из файла"""
    model_data = joblib.load(load_path)
    return model_data['model'], model_data['scaler']


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод информации о признаках для нейросети"""
    print("\n📊 Информация о признаках:")
    print("-" * 55)
    print("   ⚠️ Для нейронных сетей сложно интерпретировать важность признаков")
    print("   💡 Веса модели распределены по всем нейронам")
    print(f"   📊 Всего признаков: {len(feature_importance)}")
    print(f"   🎯 Рекомендуется использовать другие модели для анализа важности")