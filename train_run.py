import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MultimodalClassificationModel
from dataset import MultimodalVideoDataset
from utils import train_model

# ======== Парсер командной строки ========
def parse_arguments():
    parser = argparse.ArgumentParser(description="Запуск обучения мультимодальной классификационной модели")

    # Пути к данным и сохранению модели
    parser.add_argument('--video_folder_path', type=str, required=True, help="Путь до папки с видео")
    parser.add_argument('--csv_file_path', type=str, required=True, help="Путь до CSV файла с мета-данными")
    parser.add_argument('--labels_file_path', type=str, required=True, help="Путь до файла с метками")

    # Параметры моделей
    parser.add_argument('--bert_model_name', type=str, default="bert-base-uncased", help="Путь или имя модели BERT (по умолчанию: 'bert-base-uncased')")
    parser.add_argument('--xclip_model_name', type=str, default="microsoft/xclip-base-patch16-zero-shot", help="Путь или имя модели XCLIP (по умолчанию: 'microsoft/xclip-base-patch16-zero-shot')")
    parser.add_argument('--ast_model_name', type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", help="Путь или имя модели AST (по умолчанию: 'MIT/ast-finetuned-audioset-10-10-0.4593')")

    # Параметры сети
    parser.add_argument('--hidden_size', type=int, default=1024, help="Размер скрытых слоев (по умолчанию: 1024)")
    parser.add_argument('--hidden_layers', type=int, default=5, help="Количество скрытых слоев (по умолчанию: 5)")

    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=16, help="Размер мини-батча (по умолчанию: 16)")
    parser.add_argument('--num_epochs', type=int, default=35, help="Количество эпох (по умолчанию: 35)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Начальная скорость обучения (по умолчанию: 0.001)")
    parser.add_argument('--data_split_ratio', type=float, default=0.8, help="Коэффициент разделения на обучающую и валидационную выборку (по умолчанию: 0.8)")

    # Путь до сохранения весов модели
    parser.add_argument('--save_model_path', type=str, default="best_model.pth", help="Путь для сохранения модели (по умолчанию: 'best_model.pth')")

    return parser.parse_args()


# ======== Создание и запуск обучения нейросети ========
if __name__ == "__main__":
    # Парсинг параметров командной строки
    args = parse_arguments()

    # Параметры сети и устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Считывание списка меток из файла
    with open(args.labels_file_path, "r", encoding="utf-8") as f:
        label_list = [line.strip() for line in f.readlines()]

    # Инициализация модели
    input_size = 2319  # Размер объединенного эмбеддинга (примерный размер, подставьте реальный)
    num_classes = len(label_list)

    # Создание модели с переданными параметрами скрытых слоев
    model = MultimodalClassificationModel(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        verbose=True
    )
    print(model)

    # Создание датасета и DataLoader'ов
    dataset = MultimodalVideoDataset(
        csv_file=args.csv_file_path,
        video_folder=args.video_folder_path,
        label_list=label_list,
        bert_model_name=args.bert_model_name,
        xclip_model_name=args.xclip_model_name,
        ast_model_name=args.ast_model_name,
        max_workers=12,
        device=device,
        num_frames=32
    )

    # Разделение датасета на обучающую и валидационную выборки
    train_size = int(args.data_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Инициализация модели
    model = MultimodalClassificationModel(input_size=input_size, num_classes=num_classes, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, verbose=False)

    # Определение функции потерь и оптимизатора
    criterion = nn.BCELoss()  # Используем Binary Cross-Entropy для многоклассовой задачи
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Адаптивный learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Запуск обучения
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.num_epochs,
        device=device,
        scheduler=scheduler,
        label_list=label_list,
        save_model_path=args.save_model_path
    )
