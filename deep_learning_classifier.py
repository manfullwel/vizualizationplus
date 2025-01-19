"""
Deep Learning Image Classifier
============================

Este módulo implementa um classificador de imagens usando redes neurais profundas,
com suporte a diferentes datasets e arquiteturas flexíveis.

Características principais:
- Uso de funções Lambda para processamento de dados
- Implementação com tuplas para melhor organização
- Suporte a múltiplos datasets de imagens
- Arquitetura modular e extensível

Autor: Cascade AI
Data: 2025-01-18
"""

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

class ImageClassifier:
    """
    Classificador de imagens usando redes neurais profundas com arquitetura flexível.
    Implementa conceitos avançados de programação funcional e boas práticas de código.
    """
    
    def __init__(self, dataset_name: str = 'fashion_mnist'):
        """
        Inicializa o classificador com o dataset especificado.
        
        Args:
            dataset_name: Nome do dataset a ser usado (fashion_mnist, cifar10, mnist)
        """
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.model = None
        self.history = None
        
        # Funções Lambda para pré-processamento
        self.normalize = lambda x: x.astype('float32') / 255.0
        self.reshape = lambda x: x.reshape((-1,) + self.config['input_shape'] + (self.config['channels'],))
        
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Carrega e pré-processa os dados do dataset escolhido.
        
        Returns:
            Tuple contendo os conjuntos de treino e teste pré-processados
        """
        logger.info(f"Carregando dataset {self.dataset_name}")
        
        # Carrega o dataset
        dataset = SUPPORTED_DATASETS[self.dataset_name]
        (X_train_full, y_train_full), (X_test, y_test) = dataset.load_data()
        
        # Aplica normalização e reshape usando funções Lambda
        X_train_full = self.reshape(self.normalize(X_train_full))
        X_test = self.reshape(self.normalize(X_test))
        
        # Separa conjunto de validação (usando slice notation pythônica)
        validation_size = 5000
        X_valid, X_train = X_train_full[:validation_size], X_train_full[validation_size:]
        y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]
        
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    
    def create_model(self, architecture: List[int] = [300, 100]) -> keras.Model:
        """
        Cria um modelo de rede neural com arquitetura flexível.
        
        Args:
            architecture: Lista com o número de neurônios em cada camada oculta
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Criando modelo com arquitetura: %s", architecture)
        
        # Função Lambda para criar camadas densas
        dense_layer = lambda units: keras.layers.Dense(
            units=units,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(0.01)
        )
        
        # Construção do modelo usando Sequential API
        input_shape = self.config['input_shape'] + (self.config['channels'],)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),
            *[dense_layer(units) for units in architecture],
            keras.layers.Dropout(0.5),  # Adiciona regularização
            keras.layers.Dense(len(self.config['classes']), activation='softmax')
        ])
        
        # Compilação com otimizador moderno
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs: int = 30, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Treina o modelo com os dados carregados.
        
        Args:
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch para treinamento
            
        Returns:
            Histórico de treinamento
        """
        logger.info(f"Iniciando treinamento por {epochs} épocas")
        
        # Carrega os dados
        (X_train, y_train), (X_valid, y_valid), _ = self.load_data()
        
        # Callback para early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Treina o modelo
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping]
        )
        
        return self.history.history
    
    def plot_learning_curves(self) -> None:
        """
        Plota as curvas de aprendizado do modelo.
        """
        if self.history is None:
            logger.error("Nenhum histórico de treinamento encontrado")
            return
            
        # Função Lambda para criar subplots
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

def main():
    """
    Função principal para demonstração do classificador.
    """
    # Exemplo de uso com diferentes datasets
    datasets = ['fashion_mnist', 'cifar10']
    
    for dataset in datasets:
        logger.info(f"\nTreinando modelo para dataset: {dataset}")
        
        # Inicializa e treina o classificador
        classifier = ImageClassifier(dataset)
        classifier.create_model()
        classifier.train(epochs=10)  # Reduzido para demonstração
        classifier.plot_learning_curves()

if __name__ == "__main__":
    main()
