# Classificador de Imagens com Deep Learning

Este projeto implementa um classificador de imagens moderno e flexível usando redes neurais profundas, com suporte a múltiplos datasets e arquiteturas customizáveis.

## Características Principais

- Suporte a múltiplos datasets (Fashion MNIST, CIFAR-10, MNIST)
- Uso de programação funcional com funções Lambda
- Implementação organizada com tuplas e tipos tipados
- Arquitetura de rede neural flexível e customizável
- Logging para debug e monitoramento
- Visualização de curvas de aprendizado
- Regularização e early stopping

## Requisitos

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- Pandas

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```python
from deep_learning_classifier import ImageClassifier

# Criar e treinar um classificador para Fashion MNIST
classifier = ImageClassifier(dataset_name='fashion_mnist')
classifier.create_model(architecture=[300, 100])  # Duas camadas ocultas
classifier.train(epochs=30, batch_size=32)
classifier.plot_learning_curves()

# Experimentar com CIFAR-10
cifar_classifier = ImageClassifier(dataset_name='cifar10')
cifar_classifier.create_model(architecture=[512, 256, 128])  # Arquitetura mais profunda
cifar_classifier.train(epochs=50, batch_size=64)
```

## Melhorias em Relação ao Código Original

1. **Programação Funcional**:
   - Uso de funções Lambda para operações de pré-processamento
   - Composição de funções para criar camadas do modelo

2. **Organização e Tipagem**:
   - Uso de tuplas para retornos múltiplos
   - Type hints para melhor documentação
   - Constantes e configurações organizadas em dicionários

3. **Flexibilidade**:
   - Suporte a múltiplos datasets
   - Arquitetura de rede customizável
   - Parâmetros de treinamento ajustáveis

4. **Boas Práticas**:
   - Logging para debug
   - Documentação completa
   - Early stopping para evitar overfitting
   - Regularização L2 e Dropout
