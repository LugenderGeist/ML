import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
warnings.filterwarnings('ignore')

def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def select_all_features(df: pd.DataFrame, target: str = 'Смерти/д.н.') -> List[str]:
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target not in numeric_df.columns:
        print(f" Целевая переменная '{target}' не найдена!")
        return None
    
    selected_features = [col for col in numeric_df.columns if col != target]
    
    return selected_features


def split_data(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = 'Смерти/д.н.', 
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
):
    X = df[features]
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    val_relative_size = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'minmax', or 'robust'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_scaler(scaler: object, save_path: str = 'models/scaler.pkl'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    # Медицина
    df['смерти_на_диагнозы'] = df['ср. кол-во смертей в г.'] / (df['ср. кол-во рака в г.'] + 1)
    df['без_страховки'] = 100 - df['% с частной мед. страховкой'] - df['% с гос. мед. страховкой']
    
    # Финансы
    df['доход_на_бедность'] = df['медиан. доход'] / (df['% бедности'] + 1)
    df['бедность_на_доход'] = df['% бедности'] * df['медиан. доход']
    df['бедность_квадрат'] = df['% бедности'] ** 2
    df['безработица_бедность'] = df['% >16 не работают'] * df['% бедности']
    df['занятость_доход'] = df['% >16 работают'] * df['медиан. доход']
    df['уровень_занятости'] = df['% >16 работают'] / (df['% >16 не работают'] + 1)
    
    # Образование тут все нужны
    df['высшее_образование_на_бедность'] = df['% >25 оконч. бакалавр'] / (df['% бедности'] + 1)
    df['образование_доход'] = df['% >25 оконч. бакалавр'] * df['медиан. доход']
    df['взрослые_образованные'] = df['% >25 оконч. бакалавр'] / (df['% >25 оконч. 11 классов'] + 1)
     
    return df

def heatmap(
    df: pd.DataFrame, 
    save_path: str = 'plots/full_correlation.png',
    figsize: tuple = (20, 16)
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] == 0:
        print(" Нет числовых данных для построения корреляций!")
        return pd.DataFrame()
    
    correlation_matrix = numeric_df.corr()
    correlation_percent = correlation_matrix.copy()
    
    for i in range(len(correlation_percent)):
        for j in range(len(correlation_percent)):
            value = correlation_matrix.iloc[i, j] * 100
            correlation_percent.iloc[i, j] = f"{value:.0f}%"
    
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(250, 10, s=80, l=55, center='light', as_cmap=True)
    
    sns.heatmap(
        correlation_matrix,
        annot=correlation_percent,
        fmt='',
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Корреляция", "ticks": [-1, -0.5, 0, 0.5, 1]},
        annot_kws={"size": 6}
    )
    
    plt.title('Тепловая карта', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Тепловая карта сохранена: {save_path}")
    
    return correlation_matrix


def main():
    print("=" * 80)
    print("📊 ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)
    
    # Загрузка параметров
    params = load_params()
    
    # Загрузка данных
    file_path = 'data/cancer_reg1.csv'
    df = pd.read_csv(file_path)
    print(f"\n Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Удаление пропусков
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f" Удалено пропусков: {rows_removed} строк")
    
    # Создание папок
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Доп. признаки    
    original_columns_count = df.shape[1]
    df = create_interaction_features(df)
    new_features_count = df.shape[1] - original_columns_count
    
    # Тепловая карта
    heatmap(df, save_path='plots/full_correlation.png')
    
    features = select_all_features(df, target='Смерти/д.н.')
    
    if features is None or len(features) == 0:
        print(" Нет признаков для обучения!")
        return
    
    with open('models/features.json', 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    # Подготовка выборки
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, features,
        train_size=params['general']['train_size'],
        val_size=params['general']['val_size'],
        test_size=params['general']['test_size'],
        random_state=params['general']['random_state']
    )
    
    # Нормализация
    if params['preprocessing']['normalize']:
        print("\n" + "=" * 80)
        print(" НОРМАЛИЗАЦИЯ ДАННЫХ")
        print("=" * 80)
        print(f"   Метод: {params['preprocessing']['normalization_method']}")
        
        X_train, X_val, X_test, scaler = normalize_features(
            X_train, X_val, X_test,
            method=params['preprocessing']['normalization_method']
        )
        save_scaler(scaler, 'models/scaler.pkl')
    else:
        print("\n Нормализация отключена")
    
    # Подготовка данных
    prepared_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'features': features
    }
    
    joblib.dump(prepared_data, 'data/prepared_data.pkl')

if __name__ == "__main__":
    main()