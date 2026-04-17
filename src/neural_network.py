import pandas as pd
import numpy as np
import os
import tempfile
import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

def set_seed(random_state: int = 42):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

def build_neural_network(input_dim: int, params: Dict[str, Any]) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(params['layer_1_neurons'], activation=params['activation'], name='dense_1'),
        layers.Dropout(params['dropout_rate'], name='dropout_1'),
        layers.Dense(params['layer_2_neurons'], activation=params['activation'], name='dense_2'),
        layers.Dropout(params['dropout_rate'], name='dropout_2'),
        layers.Dense(params['layer_3_neurons'], activation=params['activation'], name='dense_3'),
        layers.Dropout(params['dropout_rate'], name='dropout_3'),
        layers.Dense(1, activation='linear', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_neural_network(
    X_train, y_train, X_val, y_val,
    params: Dict[str, Any],
    verbose: bool = True,
    use_tensorboard: bool = True
) -> Tuple[keras.Model, object]:
    print("\n🧠 Обучение нейронной сети...")
    set_seed(params.get('random_state', 42))

    model = build_neural_network(X_train.shape[1], params)

    callbacks_list = []
    
    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=params['patience'],
        restore_best_weights=True,
        verbose=1 if verbose else 0
    )
    callbacks_list.append(early_stop)
    
    # Reduce LR
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=params['patience'] // 2,
        min_lr=0.00001,
        verbose=1 if verbose else 0
    )
    callbacks_list.append(reduce_lr)
    
    if use_tensorboard:
        # Создаем временную папку в системе
        temp_dir = tempfile.mkdtemp()
        log_dir = os.path.join(temp_dir, 'tensorboard_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\n📊 TensorBoard логи будут сохранены во временную папку:")
        print(f"   {log_dir}")
        print(f"   Для просмотра выполните в отдельном терминале:")
        print(f"   tensorboard --logdir={log_dir}")
        print(f"   Затем откройте http://localhost:6006\n")
        
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks_list.append(tensorboard_callback)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks_list,
        verbose=1 if verbose else 0
    )

    print(f"✅ Обучение завершено")
    return model, history


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    y_train_pred = model.predict(X_train, verbose=0).flatten()
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    
    metrics = {
        'train': {
            'R2': r2_score(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred)
        },
        'validation': {
            'R2': r2_score(y_val, y_val_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'MAE': mean_absolute_error(y_val, y_val_pred)
        },
        'test': {
            'R2': r2_score(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred)
        }
    }
    feature_importance = pd.DataFrame({'Признак': feature_names, 'Важность': 0})
    return metrics, feature_importance


def save_model(model, save_path: str = 'models/neural_network.keras'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 Модель сохранена: {save_path}")


def print_feature_importance(feature_importance, top_n=8):
    print("\n📊 Информация о признаках (нейросеть):")
    print("-" * 55)
    print("   ⚠️ Важность признаков для нейросети не интерпретируется.")