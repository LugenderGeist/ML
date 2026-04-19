import joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.neural_network import train_neural_network, evaluate_model, save_model, print_feature_importance
from src.utils import save_metrics, print_metrics_table

def load_params(config_path='params.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def plot_learning_curves(history, save_path='plots/neural_network_learning_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Кривые обучения сохранены: {save_path}")


def plot_weight_distributions(model, save_path='plots/neural_network_weights.png'):
    weights = []
    names = []
    for layer in model.layers:
        if 'dense' in layer.name:
            w = layer.get_weights()[0].flatten()
            weights.append(w)
            names.append(layer.name)
    
    fig, axes = plt.subplots(1, len(weights), figsize=(5*len(weights), 4))
    if len(weights) == 1:
        axes = [axes]
    for i, (w, name) in enumerate(zip(weights, names)):
        axes[i].hist(w, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Веса слоя {name}')
        axes[i].axvline(x=0, color='r', linestyle='--')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Гистограммы весов сохранены: {save_path}")


def interpret_weights(model, feature_names, save_path='metrics/neural_network_interpretation.txt'):
    first_layer_weights = model.layers[0].get_weights()[0]
    feature_importance = np.abs(first_layer_weights).mean(axis=1)
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ИНТЕРПРЕТАЦИЯ ВЕСОВ НЕЙРОННОЙ СЕТИ\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Топ-10 наиболее важных признаков (по среднему весу в первом слое):\n")
        f.write("-" * 50 + "\n")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            f.write(f"{i+1:2}. {feature_names[idx]:<35} {feature_importance[idx]:.6f}\n")
        
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Архитектура сети:\n")
        f.write("-" * 70 + "\n")
        for i, layer in enumerate(model.layers):
            f.write(f"Слой {i+1}: {layer.name}\n")
            if 'dense' in layer.name.lower():
                weights = layer.get_weights()
                if len(weights) > 0:
                    f.write(f"  - Веса: {weights[0].shape}\n")
                    f.write(f"  - Смещения: {weights[1].shape}\n")


def main():
    print("=" * 80)
    print(" НЕЙРОННАЯ СЕТЬ")
    print("=" * 80)
    
    params = load_params()
    
    data = joblib.load('data/prepared_data.pkl')
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    features = data['features']

    # Обучение (TensorBoard включен)
    model, history = train_neural_network(
        X_train, y_train, X_val, y_val,
        params['neural_network'],
        verbose=True,
        use_tensorboard=True
    )
    
    # Визуализация
    plot_learning_curves(history)
    plot_weight_distributions(model)
    interpret_weights(model, features)
    
    # Оценка
    metrics, importance = evaluate_model(
        model, X_train, X_val, X_test,
        y_train, y_val, y_test, features
    )
    print_metrics_table(metrics, "Нейронная сеть")
    print_feature_importance(importance)
    
    # Сохранение
    save_model(model, 'models/neural_network.keras')
    save_metrics(metrics, 'neural_network')


if __name__ == "__main__":
    main()