import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
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
    """Построение архитектуры нейронной сети"""
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(params['layer_1_neurons'], activation=params['activation']))
    model.add(layers.Dropout(params['dropout_rate']))
    
    model.add(layers.Dense(params['layer_2_neurons'], activation=params['activation']))
    model.add(layers.Dropout(params['dropout_rate']))
    
    if params.get('layer_3_neurons', 0) > 0:
        model.add(layers.Dense(params['layer_3_neurons'], activation=params['activation']))
        model.add(layers.Dropout(params['dropout_rate']))
    
    model.add(layers.Dense(1, activation='linear'))
    
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
) -> Tuple[keras.Model, StandardScaler]:
    """
    Обучение нейронной сети с валидационной выборкой
    """
    print("\n🧠 Обучение нейронной сети (TensorFlow/Keras)...")
    
    set_seed(params.get('random_state', 42))
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/neural_network_scaler.pkl')
    
    # Построение модели
    model = build_neural_network(X_train.shape[1], params)
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True,
        verbose=1 if verbose else 0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=params['patience'] // 2,
        min_lr=0.00001,
        verbose=1 if verbose else 0
    )
    
    # Обучение с валидацией
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[early_stop, reduce_lr],
        verbose=1 if verbose else 0
    )
    
    print(f"✅ Модель обучена (эпох: {len(history.history['loss'])})")
    
    return model, scaler


def evaluate_model(
    model: keras.Model,
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Оценка качества модели на train, validation и test
    """
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
    y_val_pred = model.predict(X_val_scaled, verbose=0).flatten()
    y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    train_metrics = {
        'R2': r2_score(y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'MAE': mean_absolute_error(y_train, y_train_pred)
    }
    
    val_metrics = {
        'R2': r2_score(y_val, y_val_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'MAE': mean_absolute_error(y_val, y_val_pred)
    }
    
    test_metrics = {
        'R2': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    }
    
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


def save_model(model: keras.Model, scaler: StandardScaler, save_path: str = 'models/neural_network.keras'):
    """Сохранение модели и scaler"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    joblib.dump(scaler, save_path.replace('.keras', '_scaler.pkl'))
    print(f"💾 Модель сохранена: {save_path}")


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод информации о признаках"""
    print("\n📊 Информация о признаках:")
    print("-" * 55)
    print("   ⚠️ Для нейронных сетей сложно интерпретировать важность признаков")
    print(f"   📊 Всего признаков: {len(feature_importance)}")