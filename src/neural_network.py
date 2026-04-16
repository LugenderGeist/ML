import pandas as pd
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


def set_seed(random_state: int = 42):
    """Фиксация seed для воспроизводимости результатов"""
    tf.random.set_seed(random_state)
    np.random.seed(random_state)


def build_neural_network(
    input_dim: int,
    params: Dict[str, Any]
) -> keras.Model:
    """
    Построение архитектуры нейронной сети
    
    Parameters:
    -----------
    input_dim : int
        Количество входных признаков
    params : dict
        Параметры модели из params.yaml
    
    Returns:
    --------
    keras.Model
        Собранная модель
    """
    model = keras.Sequential()
    
    # Входной слой + первый скрытый слой
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(
        params['layer_1_neurons'],
        activation=params['activation']
    ))
    model.add(layers.Dropout(params['dropout_rate']))
    
    # Второй скрытый слой
    model.add(layers.Dense(
        params['layer_2_neurons'],
        activation=params['activation']
    ))
    model.add(layers.Dropout(params['dropout_rate']))
    
    # Третий скрытый слой (опционально)
    if params.get('layer_3_neurons', 0) > 0:
        model.add(layers.Dense(
            params['layer_3_neurons'],
            activation=params['activation']
        ))
        model.add(layers.Dropout(params['dropout_rate']))
    
    # Выходной слой (регрессия)
    model.add(layers.Dense(1, activation='linear'))
    
    # Компиляция модели
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    verbose: bool = False
) -> Tuple[keras.Model, None]:
    """
    Обучение нейронной сети (данные уже нормализованы в main.py!)
    
    Parameters:
    -----------
    X_train, X_val : pd.DataFrame
        Данные для обучения и валидации (УЖЕ нормализованные)
    y_train, y_val : pd.Series
        Целевые переменные
    params : dict
        Параметры модели
    verbose : bool
        Показывать ли вывод обучения
    
    Returns:
    --------
    keras.Model
        Обученная модель
    None
        Вместо scaler (нормализация выполняется в main.py)
    """
    print("\n🧠 Обучение нейронной сети (TensorFlow/Keras)...")
    
    # Фиксация seed для воспроизводимости
    set_seed(params.get('random_state', 42))
    
    # Данные уже нормализованы в main.py, поэтому просто используем их
    # Никакой дополнительной нормализации здесь не делаем!
    
    # Построение модели
    model = build_neural_network(X_train.shape[1], params)
    
    # Early stopping для предотвращения переобучения
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True,
        verbose=1 if verbose else 0
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=params['patience'] // 2,
        min_lr=0.00001,
        verbose=1 if verbose else 0
    )
    
    # Обучение с валидацией
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1 if verbose else 0
    )
    
    print(f"✅ Модель обучена (эпох: {len(history.history['loss'])})")
    
    return model, None  # Возвращаем None вместо scaler


def evaluate_model(
    model: keras.Model,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Оценка качества модели (данные уже нормализованы)
    """
    # Предсказания (данные уже нормализованы)
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    # Метрики для train
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    # Метрики для validation
    val_metrics = {
        'R2': r2_score(y_val, y_val_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred)
    }
    
    # Метрики для test
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
    # Для нейросетей сложно интерпретировать важность признаков
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Важность': 0
    })
    
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return metrics, feature_importance


def save_model(model: keras.Model, scaler: None, save_path: str = 'models/neural_network.keras'):
    """
    Сохранение модели в файл (scaler не сохраняется, т.к. нормализация в main.py)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 Модель сохранена: {save_path}")


def load_model(load_path: str = 'models/neural_network.keras') -> keras.Model:
    """Загрузка модели из файла"""
    return keras.models.load_model(load_path)


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод информации о признаках для нейросети"""
    print("\n📊 Информация о признаках:")
    print("-" * 55)
    print("   ⚠️ Для нейронных сетей сложно интерпретировать важность признаков")
    print("   💡 Веса модели распределены по всем нейронам")
    print(f"   📊 Всего признаков: {len(feature_importance)}")
    print("   🎯 Рекомендуется использовать другие модели для анализа важности")