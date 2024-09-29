## Запуск обучения модели MultimodalClassificationModel

### Пример команды для запуска обучения
```bash
python train_run.py \
    --video_folder_path "C:/Users/user/train_tag_video_2/videos_2" \
    --csv_file_path "C:/Users/user/train_dataset_tag_video/baseline/train_data_categories.csv" \
    --labels_file_path "C:/Users/user/train_dataset_tag_video/baseline/processed_tags.txt" \
    --hidden_size 2024 \
    --hidden_layers 50 \
    --save_model_path "best_model_2024_50.pth" \
    --bert_model_name "C:/path/to/local/bert_model" \
    --xclip_model_name "C:/path/to/local/xclip_model" \
    --ast_model_name "C:/path/to/local/ast_model"
```

### Параметры командной строки
- `--video_folder_path`: Путь до папки с видео.
- `--csv_file_path`: Путь до CSV файла с мета-данными.
- `--labels_file_path`: Путь до файла с метками.
- `--hidden_size`: Размер скрытых слоев (по умолчанию: `1024`).
- `--hidden_layers`: Количество скрытых слоев (по умолчанию: `5`).
- `--batch_size`: Размер мини-батча (по умолчанию: `16`).
- `--num_epochs`: Количество эпох (по умолчанию: `35`).
- `--learning_rate`: Начальная скорость обучения (по умолчанию: `0.001`).
- `--data_split_ratio`: Доля обучающей выборки (по умолчанию: `0.8`).
- `--save_model_path`: Путь для сохранения модели (по умолчанию: `"best_model.pth"`).
- `--bert_model_name`: Путь или имя модели BERT (по умолчанию: `"bert-base-uncased"`).
- `--xclip_model_name`: Путь или имя модели XCLIP (по умолчанию: `"microsoft/xclip-base-patch16-zero-shot"`).
- `--ast_model_name`: Путь или имя модели AST (по умолчанию: `"MIT/ast-finetuned-audioset-10-10-0.4593"`).

### Пример запуска обучения с основными параметрами
```bash
python train_run.py --video_folder_path "C:/path/to/videos" --csv_file_path "C:/path/to/data.csv" --labels_file_path "C:/path/to/labels.txt"
```

### Пример запуска тегирования одиночного видео
```bash
python test_predict.py \
    --model_folder_path best_final_model.pth \
    --video_folder_path "C:/path/to/videos/test.mp4" \
    --label_list_path "C:/path/to/labels.txt" \
    --title "Название видео" \
    --description "Описание видео"
```
