import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    verbose: bool = False
) -> CatBoostRegressor:
    """
    Обучение модели CatBoost с валидационной выборкой
    """
    print("\n🐱 Обучение CatBoost...")
    
    model = CatBoostRegressor(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=params['random_seed'],
        verbose=verbose
    )
    
    # Обучаем с валидационной выборкой
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    print(f"✅ Модель обучена (деревьев: {model.tree_count_})")
    print(f"   Лучшее значение loss на валидации: {model.get_best_score()['learn'][list(model.get_best_score()['learn'].keys())[0]]:.4f}")
    
    return model


def evaluate_model(
    model: CatBoostRegressor,
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
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
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
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Важность': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Важность', ascending=False)
    
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return metrics, feature_importance


def save_model(model: CatBoostRegressor, save_path: str = 'models/catboost.cbm'):
    """Сохранение модели в файл"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    print(f"💾 Модель сохранена: {save_path}")


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод топ-N наиболее важных признаков"""
    print("\n📊 Топ важности признаков (CatBoost Feature Importance):")
    print("-" * 55)
    
    top_features = feature_importance.head(top_n)
    
    for idx, row in top_features.iterrows():
        print(f"   {row['Признак']:35} {row['Важность']:10.4f}")
    
    print(f"\n   💡 CatBoost Feature Importance показывает вклад каждого признака")
    print(f"   (сумма всех важностей = {feature_importance['Важность'].sum():.2f})")