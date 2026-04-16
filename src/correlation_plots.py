import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def plot_full_correlation_heatmap(
    df: pd.DataFrame, 
    save_path: str = 'plots/full_correlation.png',
    figsize: Tuple[int, int] = (20, 16)
) -> pd.DataFrame:
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        print("❌ Нет числовых данных для построения корреляций!")
        return pd.DataFrame()
    
    # Вычисляем корреляционную матрицу
    correlation_matrix = numeric_df.corr()
    
    # Конвертируем в проценты и добавляем знак %
    correlation_percent = correlation_matrix.copy()
    for i in range(len(correlation_percent)):
        for j in range(len(correlation_percent)):
            value = correlation_matrix.iloc[i, j] * 100
            correlation_percent.iloc[i, j] = f"{value:.0f}%"
    
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(250, 10, s=80, l=55, center='light', as_cmap=True)
    
    # Рисуем тепловую карту с аннотациями в процентах со знаком %
    sns.heatmap(
        correlation_matrix,  # Оригинальные значения для цветовой шкалы
        annot=correlation_percent,  # Проценты со знаком % для подписей
        fmt='',  # Пустой формат, т.к. мы уже подготовили строки
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Корреляция", "ticks": [-1, -0.5, 0, 0.5, 1]},
        annot_kws={"size": 6}
    )
    
    plt.title('Полная матрица корреляций всех признаков (в процентах)', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Тепловая карта сохранена: {save_path}")
    
    return correlation_matrix