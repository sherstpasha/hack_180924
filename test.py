import os
import pandas as pd
import torch
import cv2
import librosa
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, XCLIPModel, BertTokenizer, BertModel, ASTForAudioClassification
from PIL import Image
from torchvision import transforms
from moviepy.editor import VideoFileClip
import numpy as np
from transformers import ASTFeatureExtractor
import concurrent.futures
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm  # Импортируем tqdm для отображения прогресса
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, confusion_matrix


# ======== Класс датасета ======== #
class MultimodalVideoDataset(Dataset):
    def __init__(self, csv_file, video_folder, label_list,
                 bert_model_name="bert-base-uncased",
                 xclip_model_name="microsoft/xclip-base-patch32",
                 ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 num_frames=8, max_workers=4, device='cpu'):
        self.data = pd.read_csv(csv_file)
        self.video_folder = video_folder
        self.label_list = label_list
        self.num_frames = num_frames
        self.max_workers = max_workers

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)

        self.xclip_feature_extractor = AutoFeatureExtractor.from_pretrained(xclip_model_name)
        self.xclip_tokenizer = AutoTokenizer.from_pretrained(xclip_model_name)
        self.xclip_model = XCLIPModel.from_pretrained(xclip_model_name).to(device)

        self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(ast_model_name)
        self.ast_model = ASTForAudioClassification.from_pretrained(ast_model_name).to(device)

        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.device = device

    def extract_audio_from_video(self, video_path, audio_save_path):
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_save_path, codec='pcm_s16le')
        audio_clip.close()
        video_clip.close()

    def extract_video_frames(self, video_path, num_frames=8):
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // num_frames, 1)

        for frame_idx in range(0, total_frames, frame_interval):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = self.video_transform(frame_pil)
            frames.append(frame_tensor)

            if len(frames) >= num_frames:
                break

        video_capture.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames).to(self.device)

    def process_text(self, text_data):
        bert_text_tokens = self.bert_tokenizer(
            text_data, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)  # Добавляем .to(self.device)
        bert_text_embedding = self.bert_model(**bert_text_tokens).pooler_output.squeeze()
        return bert_text_embedding

    def process_audio(self, audio_path):
        output_dim = self.ast_model.config.num_labels  # Получаем число выходных признаков из конфигурации модели
        if not os.path.exists(audio_path):
            return torch.zeros((1, output_dim)).to(self.device)  # Используем output_dim вместо out_features

        waveform, sample_rate = librosa.load(audio_path, sr=None)

        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        max_length = 32000
        if len(waveform) > max_length:
            waveform = waveform[:max_length]

        if len(waveform) < 400:
            waveform = np.pad(waveform, (0, 400 - len(waveform)), mode='constant')

        # Убеждаемся, что waveform является одномерным numpy массивом
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        audio_input = self.ast_feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
        ).to(self.device)  # Добавляем .to(self.device)
        with torch.no_grad():
            audio_embedding = self.ast_model(**audio_input).logits.squeeze()

        return audio_embedding

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row["video_id"]
        title = str(row["title"])
        description = str(row["description"])
        # tags = str(row["tags"]).split(",")
        tags = [tag.strip().lower() for tag in str(row["tags"]).split(",")]
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
        audio_path = os.path.join(self.video_folder, f"{video_id}.wav")
        

        text_data = title + " " + description

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_video = executor.submit(self.extract_video_frames, video_path, self.num_frames)
            future_text = executor.submit(self.process_text, text_data)
            future_audio = executor.submit(self.process_audio, audio_path)

            video_frames_tensor = future_video.result().unsqueeze(0).to(self.device)
            text_embedding = future_text.result().to(self.device)
            audio_embedding = future_audio.result().to(self.device)

            # Токенизация для XCLIP с переносом на устройство
            xclip_text_inputs = self.xclip_tokenizer(
                text_data, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                xclip_output = self.xclip_model(
                    pixel_values=video_frames_tensor,
                    input_ids=xclip_text_inputs['input_ids'],
                    attention_mask=xclip_text_inputs['attention_mask']
                )
                video_text_embedding = torch.cat(
                    (
                        xclip_output.text_embeds.squeeze(0).squeeze(0),
                        xclip_output.video_embeds.squeeze(0)
                    ),
                    dim=-1
                )

        combined_embedding = torch.cat((video_text_embedding, audio_embedding.squeeze(0), text_embedding), dim=-1)

        labels = torch.zeros(len(self.label_list)).to(self.device)
        lower_label_list = [label.lower() for label in self.label_list]
        for tag in tags:
            if tag in lower_label_list:
                labels[lower_label_list.index(tag)] = 1

        return {
            "combined_embedding": combined_embedding,
            "labels": labels
        }

    def __len__(self):
        return len(self.data)


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

# ======== Функция для обучения нейронной сети ======== #
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', early_stopping_patience=5, save_model_path='best_model.pth', scheduler=None, label_list=[], results_dir='results'):
    model.to(device)

    # Создаем директорию для хранения результатов
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Для ранней остановки
    best_loss = float('inf')
    patience_counter = 0
    
    # Хранение статистики потерь, точности и F1-меры
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_scores = []  # Хранение F1-меры на валидации

    # Внешний прогресс-бар для всех эпох
    with tqdm(total=num_epochs, desc='Обучение модели', unit='эпоха') as epoch_bar:
        for epoch in range(num_epochs):
            # ======== Тренировка ========
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Внутренний прогресс-бар для каждой эпохи (тренировка)
            with tqdm(total=len(train_loader), desc=f'Эпоха {epoch + 1}/{num_epochs} - Обучение', leave=False, unit='batch') as train_bar:
                for batch_idx, data in enumerate(train_loader):
                    inputs, labels = data["combined_embedding"].to(device), data["labels"].to(device)

                    # Обнуление градиентов
                    optimizer.zero_grad()

                    # Прямой проход
                    outputs = model(inputs)

                    # Вычисление функции потерь
                    loss = criterion(outputs, labels)
                    loss.backward()  # Обратное распространение
                    optimizer.step()  # Обновление параметров

                    # Суммирование потерь
                    running_loss += loss.item()

                    # Прогнозы
                    predicted = (outputs > 0.5).float()
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.numel()

                    # Обновление прогресс-бара обучения
                    train_bar.set_postfix(loss=loss.item())
                    train_bar.update(1)

            # Подсчет средней потери и точности на тренировочном наборе
            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct_predictions / total_samples
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            # ======== Валидация ========
            model.eval()
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_samples = 0

            # Хранение предсказаний и реальных меток для отчета
            all_labels = []
            all_predictions = []

            # Прогресс-бар для валидации
            with tqdm(total=len(val_loader), desc=f'Эпоха {epoch + 1}/{num_epochs} - Валидация', leave=False, unit='batch') as val_bar:
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data["combined_embedding"].to(device), data["labels"].to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item()

                        # Прогнозы
                        predicted = (outputs > 0.5).float()

                        # Сохранение предсказаний и меток для дальнейшего анализа
                        all_labels.extend(labels.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                        val_correct_predictions += (predicted == labels).sum().item()
                        val_total_samples += labels.numel()

                        # Обновление прогресс-бара валидации
                        val_bar.set_postfix(loss=loss.item())
                        val_bar.update(1)

            # Подсчет средней потери и точности на валидационном наборе
            epoch_val_loss = val_running_loss / len(val_loader)
            epoch_val_acc = val_correct_predictions / val_total_samples
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # ======== Расчет F1-меры ========
            all_labels_flat = [label for sublist in all_labels for label in sublist]
            all_predictions_flat = [pred for sublist in all_predictions for pred in sublist]
            epoch_f1_score = f1_score(all_labels_flat, all_predictions_flat, average='macro')
            val_f1_scores.append(epoch_f1_score)

            # Печать статистики по эпохе
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Потери на обучении: {epoch_train_loss:.4f}, Точность: {epoch_train_acc:.4f}, Потери на валидации: {epoch_val_loss:.4f}, Точность: {epoch_val_acc:.4f}, F1-мера: {epoch_f1_score:.4f}")

            # ======== Форматированный вывод предсказаний и реальных меток ========
            print("\nСравнение предсказаний и реальных меток на валидации:")
            sample_pairs = []

            for pred, true in zip(all_predictions, all_labels):
                pred_labels = [label_list[i] for i, value in enumerate(pred) if value == 1]
                true_labels = [label_list[i] for i, value in enumerate(true) if value == 1]
                sample_pairs.append((pred_labels, true_labels))

            # Выбираем 3 случайных примера из всей выборки
            random_samples = random.sample(sample_pairs, min(3, len(sample_pairs)))

            # Форматированный вывод трех случайных примеров
            for pred_labels, true_labels in random_samples:
                print(f"({', '.join(true_labels)}) -> ({', '.join(pred_labels)})")

            # Сохранение модели, если валидационная ошибка улучшилась
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_model_path)
            else:
                patience_counter += 1

            # Ранняя остановка, если модель не улучшалась в течение `early_stopping_patience` эпох
            if patience_counter >= early_stopping_patience:
                print(f"Ранняя остановка на эпохе {epoch + 1}. Потери на валидации не улучшались в течение {early_stopping_patience} эпох.")
                break

            epoch_bar.update(1)

    # ======== Сохранение графиков ========
    epochs = range(1, len(train_losses) + 1)

    # Построение графика потерь
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Потери на обучении')
    plt.plot(epochs, val_losses, label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('График потерь')
    plt.legend()
    plt.savefig(f"{results_dir}/losses.png")

    # Построение графика точности
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Точность на обучении')
    plt.plot(epochs, val_accuracies, label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title('График точности')
    plt.legend()
    plt.savefig(f"{results_dir}/accuracies.png")

    print("Обучение завершено.")


# ======== Создание и запуск обучения нейросети ======== #
if __name__ == "__main__":
    # Параметры сети
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Создание датасета
    video_folder_path = r"C:\Users\user\Desktop\train_dataset_tag_video\videos"
    csv_file_path = r"C:\Users\user\Desktop\train_dataset_tag_video\baseline\filtered_train_data_categories.csv"
    labels_file_path = r"C:\Users\user\Desktop\train_dataset_tag_video\baseline\processed_tags.txt"

    with open(labels_file_path, "r", encoding="utf-8") as f:
        label_list = [line.strip() for line in f.readlines()]


    # Инициализация модели с 5 скрытыми слоями и увеличенной размерностью скрытых слоев
    input_size = 2319  # Размер объединенного эмбеддинга (примерный размер, подставьте реальный)
    num_classes = len(label_list)  # Количество выходных классов
    hidden_size = 1024  # Увеличенная размерность скрытых слоев
    hidden_layers = 5  # Количество скрытых слоев

    model = MultimodalClassificationModel(input_size, num_classes, hidden_size=hidden_size, hidden_layers=hidden_layers, verbose=True)
    print(model)

    # Инициализация датасета и DataLoader
    dataset = MultimodalVideoDataset(
        csv_file=csv_file_path,
        video_folder=video_folder_path,
        label_list=label_list,
        max_workers=4,
        device=device,
    )

    # Разделение датасета на обучающий и валидационный
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Инициализация модели
    model = MultimodalClassificationModel(input_size=input_size, num_classes=len(label_list), verbose=False)

    # Определение функции потерь и оптимизатора
    criterion = nn.BCELoss()  # Используем Binary Cross-Entropy для многоклассовой задачи
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Адаптивный learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Запуск обучения
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, scheduler=scheduler, label_list=label_list)