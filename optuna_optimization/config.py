# ============================================
# ГЛАВНЫЙ ВЫКЛЮЧАТЕЛЬ
# ============================================
OPTUNA_ENABLED = True  # Поставь True, чтобы включить оптимизацию
# ============================================

# Настройки оптимизации
OPTUNA_SETTINGS = {
    'n_trials': 1000,           # Количество экспериментов
    'study_name': 'ml_optimization',
    'storage': None,           # 'sqlite:///optuna_study.db' для сохранения результатов
    'direction': 'maximize',   # maximize R²
    'timeout': None,           # Максимальное время в секундах
    'n_jobs': 1,               # Параллельных запусков
}

# Какие модели оптимизировать
MODELS_TO_OPTIMIZE = {
    'linear_regression': False,
    'decision_tree': False,
    'catboost': False,
    'xgboost': True,
    'neural_network': False,
}

# Пути к файлам
PATHS = {
    'params': 'params.yaml',
    'prepared_data': 'data/prepared_data.pkl',
    'results': 'metrics/optuna_results.json',
}