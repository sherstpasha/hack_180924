import os
import pandas as pd

# 1. Пути к исходному CSV-файлу и папке с видео
csv_file_path = r"C:\Users\user\Desktop\train_dataset_tag_video\baseline\train_data_categories.csv"
video_folder_path = r"C:\Users\user\Desktop\train_dataset_tag_video\videos"

# 2. Чтение CSV-файла
data = pd.read_csv(csv_file_path)

# 3. Получение списка всех video_id из CSV-файла
csv_video_ids = set(data["video_id"].tolist())

# 4. Получение списка всех файлов в указанной папке
# Считаем, что каждый видеофайл имеет формат .mp4
available_video_ids = set([filename.split(".")[0] for filename in os.listdir(video_folder_path) if filename.endswith(".mp4")])

# 5. Фильтрация строк в CSV, у которых нет соответствующего видеофайла
existing_video_ids = csv_video_ids.intersection(available_video_ids)
filtered_data = data[data["video_id"].isin(existing_video_ids)]

# 6. Сохранение отфильтрованного CSV в новый файл
output_csv_path = r"C:\Users\user\Desktop\train_dataset_tag_video\baseline\filtered_train_data_categories.csv"
filtered_data.to_csv(output_csv_path, index=False)

# 7. Вывод результата
print(f"Фильтрация завершена! Сохраненный файл: {output_csv_path}")
print(f"Количество строк в исходном CSV: {len(data)}")
print(f"Количество строк в отфильтрованном CSV: {len(filtered_data)}")
