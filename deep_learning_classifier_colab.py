# %% [markdown]
"""
# 🔥 Classificador de Imagens com Deep Learning
## Usando TensorFlow, Keras e Programação Funcional

Este notebook implementa um classificador de imagens moderno usando redes neurais profundas.
Características principais:
- 🎯 Suporte a múltiplos datasets
- 🚀 Programação funcional com Lambda
- 📊 Visualizações interativas
- 🧮 Arquitetura flexível

Autor: Cascade AI
"""

# %% [markdown]
"""
## 1. Importação das Bibliotecas

Primeiro, vamos importar todas as bibliotecas necessárias para nosso projeto.
"""

# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List, Callable, Dict, Any
import logging

# Configuração de logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %% [markdown]
"""
## 2. Configuração dos Datasets

Definimos os datasets suportados e suas configurações específicas.
- Fashion MNIST: Imagens de roupas 28x28
- CIFAR-10: Imagens coloridas 32x32
- MNIST: Dígitos escritos à mão 28x28

💡 **Dica**: Cada dataset tem suas próprias características e desafios únicos.
"""

# %%
# Tipos de datasets suportados
SUPPORTED_DATASETS = {
    'fashion_mnist': keras.datasets.fashion_mnist,
    'cifar10': keras.datasets.cifar10,
    'mnist': keras.datasets.mnist
}

# Configurações dos datasets
DATASET_CONFIGS = {
    'fashion_mnist': {
        'input_shape': (28, 28),
        'classes': ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        'channels': 1
    },
    'cifar10': {
        'input_shape': (32, 32),
        'classes': ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                   "Dog", "Frog", "Horse", "Ship", "Truck"],
        'channels': 3
    },
    'mnist': {
        'input_shape': (28, 28),
        'classes': [str(i) for i in range(10)],
        'channels': 1
    }
}

# %% [markdown]
"""
## 3. Implementação do Classificador

Nossa classe principal `ImageClassifier` implementa:
- 🔄 Carregamento e pré-processamento de dados
- 🏗️ Construção do modelo
- 📈 Treinamento e visualização

### 3.1 Definição da Classe
"""

# %%
class ImageClassifier:
    def __init__(self, dataset_name: str = 'fashion_mnist'):
        """Inicializa o classificador com o dataset especificado."""
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.model = None
        self.history = None
        
        # Funções Lambda para pré-processamento
        self.normalize = lambda x: x.astype('float32') / 255.0
        self.reshape = lambda x: x.reshape((-1,) + self.config['input_shape'] + (self.config['channels'],))

# %% [markdown]
"""
### 3.2 Carregamento de Dados

O método `load_data` implementa:
- 📥 Carregamento do dataset
- 🔍 Normalização usando Lambda
- ✂️ Separação em conjuntos de treino/validação/teste

💡 **Dica**: A normalização é crucial para o treinamento eficiente da rede neural.
"""

# %%
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Carrega e pré-processa os dados do dataset escolhido."""
        logger.info(f"Carregando dataset {self.dataset_name}")
        
        # Carrega o dataset
        dataset = SUPPORTED_DATASETS[self.dataset_name]
        (X_train_full, y_train_full), (X_test, y_test) = dataset.load_data()
        
        # Aplica normalização e reshape usando funções Lambda
        X_train_full = self.reshape(self.normalize(X_train_full))
        X_test = self.reshape(self.normalize(X_test))
        
        # Separa conjunto de validação
        validation_size = 5000
        X_valid, X_train = X_train_full[:validation_size], X_train_full[validation_size:]
        y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]
        
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

# %% [markdown]
"""
### 3.3 Criação do Modelo

O método `create_model` implementa:
- 🏗️ Arquitetura flexível com Lambda
- 🔧 Regularização L2 e Dropout
- 📊 Otimizador Adam com learning rate otimizado

💡 **Dica**: A arquitetura pode ser facilmente customizada alterando a lista `architecture`.
"""

# %%
    def create_model(self, architecture: List[int] = [300, 100]) -> keras.Model:
        """Cria um modelo de rede neural com arquitetura flexível."""
        logger.info("Criando modelo com arquitetura: %s", architecture)
        
        # Função Lambda para criar camadas densas
        dense_layer = lambda units: keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )
        
        # Construção do modelo
        input_shape = self.config['input_shape'] + (self.config['channels'],)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            *[dense_layer(units) for units in architecture],
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(self.config['classes']), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

# %% [markdown]
"""
### 3.4 Treinamento do Modelo

O método `train` implementa:
- 🏃‍♂️ Treinamento com early stopping
- 📊 Monitoramento de métricas
- 💾 Salvamento do melhor modelo

💡 **Dica**: O early stopping ajuda a evitar overfitting.
"""

# %%
    def train(self, epochs: int = 30, batch_size: int = 32) -> Dict[str, List[float]]:
        """Treina o modelo com os dados carregados."""
        logger.info(f"Iniciando treinamento por {epochs} épocas")
        
        (X_train, y_train), (X_valid, y_valid), _ = self.load_data()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping]
        )
        
        return self.history.history

# %% [markdown]
"""
### 3.5 Visualização dos Resultados

O método `plot_learning_curves` implementa:
- 📈 Gráficos de acurácia e loss
- 🎨 Estilo customizado com Lambda
- 📊 Comparação treino vs validação

💡 **Dica**: As curvas de aprendizado são essenciais para diagnosticar o desempenho do modelo.
"""

# %%
    def plot_learning_curves(self) -> None:
        """Plota as curvas de aprendizado do modelo."""
        if self.history is None:
            logger.error("Nenhum histórico de treinamento encontrado")
            return
            
        create_subplot = lambda metric, title: plt.subplot(1, 2, metric + 1).set(
            title=title,
            xlabel='Época',
            ylabel=title
        )
        
        plt.figure(figsize=(12, 4))
        
        # Plota accuracy
        create_subplot(0, 'Acurácia')
        plt.plot(self.history.history['accuracy'], label='Treino')
        plt.plot(self.history.history['val_accuracy'], label='Validação')
        plt.legend()
        
        # Plota loss
        create_subplot(1, 'Loss')
        plt.plot(self.history.history['loss'], label='Treino')
        plt.plot(self.history.history['val_loss'], label='Validação')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# %% [markdown]
"""
## 4. Demonstração de Uso

Vamos treinar modelos em diferentes datasets para demonstrar a flexibilidade do classificador.

💡 **Dica**: Experimente com diferentes arquiteturas e hiperparâmetros!
"""

# %%
def main():
    """Função principal para demonstração do classificador."""
    datasets = ['fashion_mnist', 'cifar10']
    
    for dataset in datasets:
        logger.info(f"\nTreinando modelo para dataset: {dataset}")
        
        classifier = ImageClassifier(dataset)
        classifier.create_model()
        classifier.train(epochs=10)
        classifier.plot_learning_curves()

# %%
if __name__ == "__main__":
    main()

# %% [markdown]
"""
## 5. Conclusão

Este notebook demonstrou:
- 🎯 Implementação moderna de deep learning
- 🚀 Uso de programação funcional
- 📊 Visualização efetiva de resultados
- 🔧 Práticas de código limpo e organizado

Experimente modificar os parâmetros e arquiteturas para melhorar ainda mais os resultados!
"""
