import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Criar diretório para imagens se não existir
os.makedirs("images", exist_ok=True)

# 1. Gráfico de Acurácia por Época
def plot_accuracy_curve():
    epochs = np.arange(1, 31)
    train_acc = 0.7 + 0.2 * (1 - np.exp(-epochs/10)) + 0.02 * np.random.randn(30)
    val_acc = 0.65 + 0.15 * (1 - np.exp(-epochs/8)) + 0.02 * np.random.randn(30)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, 'b-', label='Treino')
    plt.plot(epochs, val_acc, 'r-', label='Validação')
    plt.title('Curva de Aprendizado: Acurácia', fontsize=14)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('images/accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Matriz de Confusão Estilizada
def plot_confusion_matrix():
    # Dados simulados
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    cm = np.random.randint(50, 200, size=(10, 10))
    np.fill_diagonal(cm, np.random.randint(800, 1000, size=10))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão: Fashion MNIST', fontsize=14)
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Comparação de Modelos
def plot_model_comparison():
    models = ['Basic CNN', 'ResNet50', 'MobileNetV2', 'EfficientNet']
    accuracy = [0.89, 0.94, 0.92, 0.95]
    inference_time = [15, 45, 25, 35]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Acurácia
    bars1 = ax1.bar(models, accuracy, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
    ax1.set_title('Comparação de Acurácia', fontsize=14)
    ax1.set_ylabel('Acurácia', fontsize=12)
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Tempo de Inferência
    bars2 = ax2.bar(models, inference_time, color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
    ax2.set_title('Tempo de Inferência', fontsize=14)
    ax2.set_ylabel('Tempo (ms)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Pipeline de Processamento
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6))
    steps = ['Input\nImage', 'Pre-\nprocessing', 'Feature\nExtraction', 'Classification', 'Post-\nprocessing', 'Output']
    x = np.arange(len(steps))
    
    # Criar boxes para cada etapa
    for i, step in enumerate(steps):
        rect = plt.Rectangle((i-0.4, 0.2), 0.8, 0.6, 
                           facecolor=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f', '#9b59b6', '#1abc9c'][i],
                           alpha=0.7)
        ax.add_patch(rect)
        ax.text(i, 0.5, step, ha='center', va='center', fontsize=12, fontweight='bold')
        
        if i < len(steps)-1:
            ax.arrow(i+0.4, 0.5, 0.2, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    ax.set_xlim(-0.5, len(steps)-0.5)
    ax.set_ylim(0, 1)
    ax.set_title('Pipeline de Processamento de Imagens', fontsize=14, pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('images/pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_accuracy_curve()
    plot_confusion_matrix()
    plot_model_comparison()
    plot_pipeline()
