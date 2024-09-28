import torch
from utils import predict_single_video



# ======== Пример использования ======== #
if __name__ == "__main__":
    model_path = "path/to/best_model.pth"  # Путь до модели
    video_path = "path/to/video.mp4"  # Путь до видеофайла
    title = "Example Video Title"  # Название видео
    description = "This is an example description of the video."  # Описание видео

    # Устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Получение предсказанных меток
    predicted_tags = predict_single_video(
        model_path=model_path,
        video_path=video_path,
        title=title,
        description=description,
        label_list=None, 
        device=device
    )
    
    print(f"Предсказанные метки: {predicted_tags}")