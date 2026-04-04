import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional


def plot_full_correlation_heatmap(
    df: pd.DataFrame, 
    save_path: str = 'plots/full_correlation.png',
    figsize: Tuple[int, int] = (20, 16)
) -> pd.DataFrame:
    """
    Построение полной тепловой карты корреляций для всех числовых признаков
    """
    print("\n📊 Построение полной тепловой карты корреляций...")
    
    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        print("❌ Нет числовых данных для построения корреляций!")
        return pd.DataFrame()
    
    # Вычисляем корреляционную матрицу
    correlation_matrix = numeric_df.corr()
    
    # Настройка стиля
    plt.figure(figsize=figsize)
    
    # Используем симметричную цветовую карту
    cmap = sns.diverging_palette(250, 10, s=80, l=55, center='light', as_cmap=True)
    
    sns.heatmap(
        correlation_matrix,
        annot=False,
        fmt='.2f',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Корреляция", "ticks": [-1, -0.5, 0, 0.5, 1]},
        xticklabels=True,
        yticklabels=True
    )
    
    plt.title('Полная матрица корреляций всех признаков', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график, не показывая
    print(f"✅ Полная тепловая карта сохранена: {save_path}")
    
    return correlation_matrix


def plot_high_correlation_heatmap(
    df: pd.DataFrame,
    target: str = 'Смерти/д.н.',
    threshold: float = 0.3,
    save_path: str = 'plots/high_correlation.png',
    figsize: Tuple[int, int] = (14, 12)
) -> List[str]:
    """
    Построение тепловой карты только для признаков с высокой корреляцией с целевой переменной
    """
    print(f"\n📊 Построение тепловой карты для признаков с корреляцией > {threshold*100:.0f}%...")
    
    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f"❌ Целевая переменная '{target}' не найдена в данных!")
        return []
    
    # Вычисляем корреляции с целевой переменной
    correlations_abs = numeric_df.corr()[target].abs().sort_values(ascending=False)
    correlations_raw = numeric_df.corr()[target].sort_values(ascending=False)
    
    # Отбираем признаки с высокой корреляцией
    high_corr_features = correlations_abs[correlations_abs > threshold].index.tolist()
    
    if target in high_corr_features:
        high_corr_features.remove(target)
    
    if len(high_corr_features) == 0:
        print(f"⚠️ Не найдено признаков с корреляцией > {threshold*100:.0f}%")
        return []
    
    # Создаем датафрейм только с отобранными признаками + целевой переменной
    selected_columns = high_corr_features + [target]
    filtered_df = numeric_df[selected_columns]
    
    # Вычисляем корреляционную матрицу
    correlation_matrix = filtered_df.corr()
    
    # Настройка стиля
    plt.figure(figsize=figsize)
    
    # Используем симметричную цветовую карту
    cmap = sns.diverging_palette(250, 10, s=80, l=55, center='light', as_cmap=True)
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Корреляция", "ticks": [-1, -0.5, 0, 0.5, 1]},
        annot_kws={"size": 8}
    )
    
    plt.title(f'Тепловая карта корреляций (признаки с корреляцией > {threshold*100:.0f}%)\nЦелевая переменная: {target}', 
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график, не показывая
    print(f"✅ Тепловая карта высоких корреляций сохранена: {save_path}")
    
    return high_corr_features


def plot_correlation_with_target(
    df: pd.DataFrame,
    target: str = 'Смерти/д.н.',
    top_n: int = 15,
    save_path: str = 'plots/correlation_with_target.png',
    figsize: Tuple[int, int] = (10, 8)
) -> pd.Series:
    """
    Построение bar plot корреляций признаков с целевой переменной
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f"❌ Целевая переменная '{target}' не найдена!")
        return pd.Series()
    
    # Вычисляем корреляции
    correlations = numeric_df.corr()[target].drop(target).sort_values(ascending=False)
    
    # Берем топ N
    top_correlations = correlations.head(top_n)
    
    # Создаем график
    plt.figure(figsize=figsize)
    
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_correlations.values]
    
    bars = plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Добавляем значения на столбцы
    for i, (bar, value) in enumerate(zip(bars, top_correlations.values)):
        if value > 0:
            plt.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)
        else:
            plt.text(value - 0.03, i, f'{value:.3f}', va='center', ha='right', fontsize=9)
    
    plt.yticks(range(len(top_correlations)), top_correlations.index)
    plt.xlabel('Корреляция с целевой переменной', fontsize=12)
    plt.title(f'Топ-{top_n} признаков, коррелирующих с "{target}"', fontsize=14, fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Порог 0.3')
    plt.axvline(x=-0.3, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.legend(['Нулевая корреляция', 'Порог 0.3'], loc='lower right')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем график, не показывая
    print(f"✅ График корреляций сохранен: {save_path}")
    
    return top_correlations