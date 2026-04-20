import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict, Any


def train_linear_regression(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    params: Dict[str, Any]
) -> LinearRegression:
    """Обучение модели линейной регрессии"""
    print("\n📈 Обучение линейной регрессии...")
    
    model = LinearRegression(
        fit_intercept=params['fit_intercept'],
        positive=params['positive']
    )
    
    model.fit(X_train, y_train)
    
    print(f"✅ Модель обучена (коэффициентов: {model.coef_.shape[0]})")
    
    return model


def evaluate_model(
    model: LinearRegression,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    feature_names: list
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Оценка качества модели"""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
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
    
    metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    return metrics, None


def save_model(model: LinearRegression, save_path: str = 'models/linear_regression.pkl'):
    """Сохранение модели в файл"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"💾 Модель сохранена: {save_path}")


def get_feature_importance(model: LinearRegression, feature_names: list) -> pd.DataFrame:
    """
    Расчет относительной важности признаков для линейной регрессии.
    Используется сумма абсолютных значений коэффициентов.
    """
    # Берем абсолютные значения коэффициентов
    abs_coefficients = np.abs(model.coef_)
    
    # Суммируем для расчета процентов
    total = abs_coefficients.sum()
    
    # Рассчитываем процент вклада каждого признака
    importance_percent = (abs_coefficients / total) * 100
    
    # Сортируем по убыванию
    sorted_idx = np.argsort(importance_percent)[::-1]
    
    results = []
    for i in sorted_idx:
        results.append({
            'Признак': feature_names[i],
            'Коэффициент': model.coef_[i],
            'Вклад_процент': importance_percent[i]
        })
    
    df = pd.DataFrame(results)
    return df


def print_feature_importance(feature_importance: pd.DataFrame, top_n: int = 10):
    """Вывод топ-N наиболее важных признаков"""
    print("\n📊 Топ важности признаков (Линейная регрессия):")
    print("-" * 65)
    print(f"{'Признак':<40} {'Коэффициент':>12} {'Вклад %':>10}")
    print("-" * 65)
    
    for idx, row in feature_importance.head(top_n).iterrows():
        sign = "➕" if row['Коэффициент'] > 0 else "➖"
        print(f"{row['Признак']:<40} {row['Коэффициент']:12.6f} {row['Вклад_процент']:9.2f}% {sign}")
    
    print(f"\n   💡 Вклад показывает долю влияния признака в модели (сумма = 100%)")
    print(f"   ➕ положительный коэффициент = увеличение смертности")
    print(f"   ➖ отрицательный коэффициент = снижение смертности")