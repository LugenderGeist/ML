import optuna
import json
import yaml
from datetime import datetime
from .config import OPTUNA_ENABLED, OPTUNA_SETTINGS, MODELS_TO_OPTIMIZE, PATHS
from .objective_functions import OBJECTIVES


def save_best_params(best_params, model_name):
    """Сохранение лучших параметров в файл"""
    output_file = f'metrics/optuna_best_{model_name}.json'
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"✅ Лучшие параметры для {model_name} сохранены в {output_file}")
    return output_file


def update_params_yaml(best_params, model_name):
    """Обновление params.yaml лучшими параметрами"""
    with open(PATHS['params'], 'r') as f:
        params = yaml.safe_load(f)
    
    # Обновляем параметры модели
    if model_name in params:
        for key, value in best_params.items():
            if key in params[model_name]:
                params[model_name][key] = value
    
    # Сохраняем резервную копию
    backup_file = f'params_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    with open(backup_file, 'w') as f:
        yaml.dump(params, f)
    
    # Сохраняем обновленный файл
    with open(PATHS['params'], 'w') as f:
        yaml.dump(params, f)
    
    print(f"✅ Параметры для {model_name} обновлены в {PATHS['params']}")
    print(f"📁 Резервная копия: {backup_file}")


def optimize_model(model_name):
    """Оптимизация одной модели"""
    if model_name not in OBJECTIVES:
        print(f"❌ Модель '{model_name}' не найдена!")
        return None
    
    print(f"\n{'='*60}")
    print(f"🔍 Оптимизация: {model_name}")
    print(f"{'='*60}")
    
    # Создаем study
    study = optuna.create_study(
        study_name=f"{OPTUNA_SETTINGS['study_name']}_{model_name}",
        storage=OPTUNA_SETTINGS['storage'],
        direction=OPTUNA_SETTINGS['direction'],
        load_if_exists=True,
    )
    
    # Оптимизируем
    study.optimize(
        OBJECTIVES[model_name],
        n_trials=OPTUNA_SETTINGS['n_trials'],
        timeout=OPTUNA_SETTINGS['timeout'],
        n_jobs=OPTUNA_SETTINGS['n_jobs'],
        show_progress_bar=True,
    )
    
    # Результаты
    print(f"\n📊 Лучшие параметры для {model_name}:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print(f"🏆 Лучшее значение R²: {study.best_value:.4f}")
    
    # Сохраняем
    save_best_params(study.best_params, model_name)
    
    return study.best_params


def run_all_optimizations():
    """Запуск оптимизации для всех выбранных моделей"""
    if not OPTUNA_ENABLED:
        print("⚠️ Optuna отключен. Установи OPTUNA_ENABLED = True в config.py")
        return {}
    
    print("=" * 60)
    print("🚀 ЗАПУСК OPTUNA ОПТИМИЗАЦИИ")
    print("=" * 60)
    print(f"Количество попыток на модель: {OPTUNA_SETTINGS['n_trials']}")
    print()
    
    results = {}
    
    # Оптимизация по порядку
    models_order = ['linear_regression', 'decision_tree', 'catboost', 'xgboost', 'neural_network']
    
    for model_name in models_order:
        if MODELS_TO_OPTIMIZE.get(model_name, False):
            best_params = optimize_model(model_name)
            if best_params:
                results[model_name] = best_params
                
                # Обновляем params.yaml после каждой модели
                if input(f"\nОбновить {PATHS['params']} лучшими параметрами для {model_name}? (y/n): ").lower() == 'y':
                    update_params_yaml(best_params, model_name)
    
    # Сохраняем общие результаты
    with open(PATHS['results'], 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Все результаты сохранены в {PATHS['results']}")
    return results


def run_single_optimization(model_name):
    """Запуск оптимизации для одной модели"""
    if not OPTUNA_ENABLED:
        print("⚠️ Optuna отключен. Установи OPTUNA_ENABLED = True в config.py")
        return None
    
    if model_name not in MODELS_TO_OPTIMIZE or not MODELS_TO_OPTIMIZE[model_name]:
        print(f"❌ Модель '{model_name}' не выбрана для оптимизации в config.py")
        return None
    
    best_params = optimize_model(model_name)
    
    if best_params and input(f"\nОбновить {PATHS['params']} лучшими параметрами? (y/n): ").lower() == 'y':
        update_params_yaml(best_params, model_name)
    
    return best_params


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        run_single_optimization(sys.argv[1])
    else:
        run_all_optimizations()