import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')


def load_metrics(model_name):
    file_path = f'metrics/{model_name}_metrics.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def create_plot_csv(model_name, model_name_ru, metrics):
    plot_data = pd.DataFrame({
        'dataset': ['train', 'validation', 'test'],
        'R2': [
            metrics['train']['R2'],
            metrics['validation']['R2'],
            metrics['test']['R2']
        ],
        'RMSE': [
            metrics['train']['RMSE'],
            metrics['validation']['RMSE'],
            metrics['test']['RMSE']
        ],
        'MAE': [
            metrics['train']['MAE'],
            metrics['validation']['MAE'],
            metrics['test']['MAE']
        ]
    })
    
    csv_path = f'metrics/{model_name}_plot_data.csv'
    plot_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f" Создан CSV для графиков: {csv_path}")
    return csv_path

def main():
    print("=" * 80)
    print(" СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    
    models = ['linear_regression', 'decision_tree', 'catboost', 'xgboost', 'neural_network']
    model_names_ru = ['Линейная регрессия', 'Дерево решений', 'CatBoost', 'XGBoost', 'Нейронная сеть']
    
    results = []
    
    for model, name_ru in zip(models, model_names_ru):
        metrics = load_metrics(model)
        if metrics and 'test' in metrics:
            # Создаем CSV для DVC plots
            create_plot_csv(model, name_ru, metrics)
            
            # Собираем для сравнения
            results.append({
                'Модель': name_ru,
                'R²_test': metrics['test']['R2'],
                'RMSE_test': metrics['test']['RMSE'],
                'MAE_test': metrics['test']['MAE'],
                'R²_val': metrics['validation']['R2'] if 'validation' in metrics else None
            })
            
            # Вывод в консоль
            print(f"\n{name_ru}:")
            print(f"  R² test: {metrics['test']['R2']:.4f}")
            print(f"  RMSE test: {metrics['test']['RMSE']:.4f}")
            print(f"  MAE test: {metrics['test']['MAE']:.4f}")
    
    if results:
        # Сортируем результаты
        results_sorted = sorted(results, key=lambda x: x['R²_test'], reverse=True)
        
        print("\n" + "=" * 80)
        print(" РЕЗУЛЬТАТЫ (отсортировано по R² test):")
        print("=" * 80)
        for r in results_sorted:
            print(f"{r['Модель']:25} R²: {r['R²_test']:.4f} | RMSE: {r['RMSE_test']:.2f} | MAE: {r['MAE_test']:.2f}")
        
        # Сохраняем JSON для сравнения
        comparison_data = {
            'models': results_sorted,
            'best_model': results_sorted[0]['Модель'],
            'best_r2': results_sorted[0]['R²_test'],
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('metrics/models_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)

        # Сохраняем CSV для удобного просмотра
        df = pd.DataFrame(results_sorted)
        df.to_csv('metrics/models_comparison.csv', index=False, encoding='utf-8-sig')
 
        print(f"\n Лучшая модель: {results_sorted[0]['Модель']} (R² = {results_sorted[0]['R²_test']:.4f})")
    else:
        print(" Нет метрик для сравнения!")


if __name__ == "__main__":
    main()