import torch.nn as nn
import torch


# ======== Определение архитектуры нейронной сети ======== #
class MultimodalClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, hidden_layers=3, verbose=True):
        """
        Многослойная архитектура классификационной модели.

        Аргументы:
        input_size (int): Размер входного тензора.
        num_classes (int): Количество классов (размер выходного слоя).
        hidden_size (int): Размерность скрытых слоев.
        hidden_layers (int): Количество скрытых слоев.
        verbose (bool): Флаг для вывода информации о промежуточных размерах.
        """
        super(MultimodalClassificationModel, self).__init__()

        # Переменная для управления выводом размеров
        self.verbose = verbose

        # Список слоев нейросети
        layers = []

        # Первый линейный слой с BatchNorm и активацией
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization для улучшения стабильности
        layers.append(nn.LeakyReLU(0.1))  # LeakyReLU с небольшим углом на отрицательной стороне
        layers.append(nn.Dropout(0.5))  # Dropout для регуляризации

        # Дополнительные скрытые слои
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.5))

        # Выходной слой
        layers.append(nn.Linear(hidden_size, num_classes))

        # Объединяем слои в последовательную модель
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print(f"Входной тензор: {x.shape}")  # Размерность входного тензора

        # Прогон через все слои модели
        x = self.model(x)

        if self.verbose:
            print(f"Размер после последнего линейного слоя: {x.shape}")  # Размерность после всех скрытых слоев

        # Sigmoid для многоклассовой классификации
        x = torch.sigmoid(x)

        if self.verbose:
            print(f"После Sigmoid: {x.shape}")  # Размерность после применения Sigmoid

        return x