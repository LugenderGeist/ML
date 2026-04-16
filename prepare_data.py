import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.correlation_plots import plot_full_correlation_heatmap
from src.preprocessing import normalize_features, save_scaler
from src.utils import select_all_features, prepare_data_for_training


def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def main():
    print("=" * 80)
    print("📊 ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)
    
    # Загрузка параметров
    params = load_params()
    
    # Загрузка данных
    file_path = 'data/cancer_reg1.csv'
    df = pd.read_csv(file_path)
    print(f"\n✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Удаление пропусков
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"🗑️ Удалено пропусков: {rows_removed} строк")
    
    # Создание папок
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Тепловая карта (до нормализации)
    plot_full_correlation_heatmap(df, save_path='plots/full_correlation.png')
    
    # Выбор всех признаков
    features = select_all_features(df, target='Смерти/д.н.')
    
    if features is None or len(features) == 0:
        print("❌ Нет признаков для обучения!")
        return
    
    # Сохраняем список признаков
    with open('models/features.json', 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    
    # Разделение данных
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_training(
        df, features,
        train_size=params['general']['train_size'],
        val_size=params['general']['val_size'],
        test_size=params['general']['test_size'],
        random_state=params['general']['random_state']
    )
    
    # Нормализация (если включена)
    if params['preprocessing']['normalize']:
        print("\n" + "=" * 80)
        print("🔄 НОРМАЛИЗАЦИЯ ДАННЫХ")
        print("=" * 80)
        print(f"   Метод: {params['preprocessing']['normalization_method']}")
        
        X_train, X_val, X_test, scaler = normalize_features(
            X_train, X_val, X_test,
            method=params['preprocessing']['normalization_method']
        )
        save_scaler(scaler, 'models/scaler.pkl')
    else:
        print("\n⚠️ Нормализация отключена")
    
    # Сохраняем подготовленные данные
    prepared_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'features': features
    }
    
    os.makedirs('data', exist_ok=True)
    joblib.dump(prepared_data, 'data/prepared_data.pkl')
    print("\n✅ Подготовленные данные сохранены: data/prepared_data.pkl")


if __name__ == "__main__":
    main()