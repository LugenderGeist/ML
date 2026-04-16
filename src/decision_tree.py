import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any


def train_decision_tree(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    params: Dict[str, Any]
) -> DecisionTreeRegressor:
    """Обучение модели дерева решений"""
    print("\n🌳 Обучение дерева решений...")
    
    model = DecisionTreeRegressor(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=params['random_state']
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Модель обучена (глубина: {model.get_depth()}, листьев: {model.get_n_leaves()})")
    
    return model


def evaluate_model(
    model: DecisionTreeRegressor,
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


def save_model(model: DecisionTreeRegressor, save_path: str = 'models/decision_tree.pkl'):
    """Сохранение модели в файл"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"💾 Модель сохранена: {save_path}")


def visualize_tree(
    model: DecisionTreeRegressor, 
    feature_names: list,
    max_depth: int = 3,
    save_path: str = 'plots/decision_tree.png'
):
    """Визуализация дерева решений"""
    plt.figure(figsize=(20, 12))
    
    plot_tree(
        model, 
        feature_names=feature_names,
        filled=True, 
        rounded=True,
        fontsize=10,
        max_depth=max_depth
    )
    
    plt.title(f'Дерево решений (первые {max_depth} уровня)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 8):
    """Вывод топ-N наиболее важных признаков"""
    print("\n📊 Топ важности признаков (Feature Importance):")
    print("-" * 55)
    
    top_features = feature_importance.head(top_n)
    
    for idx, row in top_features.iterrows():
        print(f"   {row['Признак']:35} {row['Важность']:10.4f}")
    
    print(f"\n   💡 Feature Importance показывает вклад признака в уменьшение неопределенности")
    print(f"   (сумма всех важностей = {feature_importance['Важность'].sum():.2f})")