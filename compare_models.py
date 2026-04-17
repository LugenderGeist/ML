import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

def load_metrics(model_name):
    file_path = f'metrics/{model_name}_metrics.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def main():
    print("=" * 80)
    print("🏆 СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    models = ['linear_regression', 'decision_tree', 'catboost', 'xgboost', 'neural_network']
    model_names = ['Линейная регрессия', 'Дерево решений', 'CatBoost', 'XGBoost', 'Нейронная сеть']
    
    results = []
    
    for model, name in zip(models, model_names):
        metrics = load_metrics(model)
        if metrics and 'test' in metrics:
            results.append({
                'Модель': name,
                'R² (test)': metrics['test']['R2'],
                'RMSE (test)': metrics['test']['RMSE'],
                'MAE (test)': metrics['test']['MAE']
            })
            if 'validation' in metrics:
                results[-1]['R² (val)'] = metrics['validation']['R2']
    
    if results:
        comparison = pd.DataFrame(results)
        comparison = comparison.sort_values('R² (test)', ascending=False)
        
        print("\n" + comparison.to_string(index=False))
        comparison.to_csv('metrics/models_comparison.csv', index=False)
        
        best_model = comparison.iloc[0]['Модель']
        best_r2 = comparison.iloc[0]['R² (test)']
        print(f"\n Лучшая модель: {best_model} (R² = {best_r2:.4f})")
    else:
        print(" Нет метрик для сравнения!")


if __name__ == "__main__":
    main()