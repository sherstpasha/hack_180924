import argparse
import torch
from utils import predict_single_video


def parse_arguments():
    parser = argparse.ArgumentParser(description="Запуск модели тегирования видео")
    
    parser.add_argument('--model_folder_path', type=str, required=True, help="Путь до папки с моделью")
    parser.add_argument('--video_folder_path', type=str, required=True, help="Путь до папки с видео")
    
    parser.add_argument('--title', type=str, default="Example Video Title", help="Название видео")
    parser.add_argument('--description', type=str, default="This is an example description of the video.", help="Описание видео")


# ======== Пример использования ======== #
if __name__ == "__main__":
    # Парсинг параметров командной строки
    args = parse_arguments()
    
    # Устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Получение предсказанных меток
    predicted_tags = predict_single_video(
        model_path=args.model_folder_path,
        video_path=args.video_folder_path,
        title=args.title,
        description=args.description,
        label_list=None, 
        device=device
    )
    
    print(f"Предсказанные метки: {predicted_tags}")
