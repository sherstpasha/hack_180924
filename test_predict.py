import argparse
import torch
from utils import predict_single_video


def parse_arguments():
    parser = argparse.ArgumentParser(description="Запуск модели тегирования видео")
    
    parser.add_argument('--model_folder_path', type=str, required=True, help="Путь до папки с моделью")
    parser.add_argument('--video_folder_path', type=str, required=True, help="Путь до папки с видео")
    parser.add_argument('--label_list_path', type=str, required=True, help="Путь до файла с названиями меток")
    parser.add_argument('--title', type=str, default="Example Video Title", help="Название видео")
    parser.add_argument('--description', type=str, default="This is an example description of the video.", help="Описание видео")

    return parser.parse_args()

# ======== Пример использования ======== #
if __name__ == "__main__":
    # Парсинг параметров командной строки
    args = parse_arguments()
    
    # Устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Считывание списка меток из файла
    with open(args.label_list_path, "r", encoding="utf-8") as f:
        label_list = [line.strip() for line in f.readlines()]

    # Получение предсказанных меток
    predicted_tags = predict_single_video(
        model_path=args.model_folder_path,
        video_path=args.video_folder_path,
        title=args.title,
        description=args.description,
        label_list=label_list, 
        device=device
    )
    
    print(f"Предсказанные метки: {predicted_tags}")

