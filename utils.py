import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score
import torch
from dataset import SingleVideoDataset
from model import MultimodalClassificationModel


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', early_stopping_patience=5, save_model_path='best_model.pth', scheduler=None, label_list=[], results_dir='results'):
    model.to(device)

    # Создаем директорию для хранения результатов
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Для ранней остановки и отслеживания лучших метрик
    best_loss = float('inf')
    best_f1 = 0.0  # Инициализация лучшей F1-меры
    patience_counter = 0

    # Хранение статистики потерь, точности и F1-меры
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1_scores = []  # Хранение F1-меры на валидации

    # Определяем пути для сохранения моделей с суффиксами
    save_model_path_loss = save_model_path.replace('.pth', '_loss.pth')
    save_model_path_f1 = save_model_path.replace('.pth', '_f1.pth')

    # Внешний прогресс-бар для всех эпох
    with tqdm(total=num_epochs, desc='Обучение модели', unit='эпоха') as epoch_bar:
        for epoch in range(num_epochs):
            # ======== Тренировка ========
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Хранение предсказаний и реальных меток для отчета
            all_train_labels = []
            all_train_predictions = []

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

                    # Если ни один из выходов не перешел порог, назначаем метку с максимальным значением
                    for i, pred in enumerate(predicted):
                        if pred.sum() == 0:  # Если все метки ниже порога
                            max_index = outputs[i].argmax()  # Находим индекс максимального значения
                            predicted[i][max_index] = 1  # Устанавливаем этот индекс как предсказанную метку
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.numel()

                    # Сохранение предсказаний и меток для дальнейшего анализа
                    all_train_labels.extend(labels.cpu().numpy())
                    all_train_predictions.extend(predicted.cpu().numpy())

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
            all_val_labels = []
            all_val_predictions = []

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

                        # Если ни один из выходов не перешел порог, назначаем метку с максимальным значением
                        for i, pred in enumerate(predicted):
                            if pred.sum() == 0:  # Если все метки ниже порога
                                max_index = outputs[i].argmax()  # Находим индекс максимального значения
                                predicted[i][max_index] = 1  # Устанавливаем этот индекс как предсказанную метку

                        # Сохранение предсказаний и меток для дальнейшего анализа
                        all_val_labels.extend(labels.cpu().numpy())
                        all_val_predictions.extend(predicted.cpu().numpy())

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
            all_labels_flat = [label for sublist in all_val_labels for label in sublist]
            all_predictions_flat = [pred for sublist in all_val_predictions for pred in sublist]
            epoch_f1_score = f1_score(all_labels_flat, all_predictions_flat, average='macro')
            val_f1_scores.append(epoch_f1_score)

            # Печать статистики по эпохе
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Потери на обучении: {epoch_train_loss:.4f}, Точность: {epoch_train_acc:.4f}, Потери на валидации: {epoch_val_loss:.4f}, Точность: {epoch_val_acc:.4f}, F1-мера: {epoch_f1_score:.4f}")

            # ======== Форматированный вывод предсказаний и реальных меток на обучении ========
            print("\nПримеры предсказаний на тренировочных данных:")
            train_sample_pairs = []

            for pred, true in zip(all_train_predictions, all_train_labels):
                pred_labels = [label_list[i] for i, value in enumerate(pred) if value == 1]
                true_labels = [label_list[i] for i, value in enumerate(true) if value == 1]
                train_sample_pairs.append((pred_labels, true_labels))

            # Выбираем 3 случайных примера из всей выборки
            train_random_samples = random.sample(train_sample_pairs, min(10, len(train_sample_pairs)))

            for pred_labels, true_labels in train_random_samples:
                print(f"Обучение: ({', '.join(true_labels)}) -> ({', '.join(pred_labels)})")

            # ======== Форматированный вывод предсказаний и реальных меток на валидации ========
            print("\nПримеры предсказаний на валидационных данных:")
            val_sample_pairs = []

            for pred, true in zip(all_val_predictions, all_val_labels):
                pred_labels = [label_list[i] for i, value in enumerate(pred) if value == 1]
                true_labels = [label_list[i] for i, value in enumerate(true) if value == 1]
                val_sample_pairs.append((pred_labels, true_labels))

            val_random_samples = random.sample(val_sample_pairs, min(10, len(val_sample_pairs)))

            for pred_labels, true_labels in val_random_samples:
                print(f"Валидация: ({', '.join(true_labels)}) -> ({', '.join(pred_labels)})")

            # ======== Сохранение модели при улучшении потерь на валидации ========
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_model_path_loss)
                print(f"Сохранена лучшая модель по потере на валидации: {save_model_path_loss}")
            else:
                patience_counter += 1

            # ======== Сохранение модели при улучшении F1-меры на валидации ========
            if epoch_f1_score > best_f1:
                best_f1 = epoch_f1_score
                torch.save(model.state_dict(), save_model_path_f1)
                print(f"Сохранена лучшая модель по F1-мере на валидации: {save_model_path_f1}")

            # Ранняя остановка, если модель не улучшалась в течение `early_stopping_patience` эпох
            # if patience_counter >= early_stopping_patience:
            #     print(f"Ранняя остановка на эпохе {epoch + 1}. Потери на валидации не улучшались в течение {early_stopping_patience} эпох.")
            #     break

            epoch_bar.update(1)

    print("Обучение завершено.")


def predict_single_video(model_path, video_path, title, description, label_list=None,
                         bert_model_name="bert-base-uncased",
                         xclip_model_name="microsoft/xclip-base-patch16-zero-shot",
                         ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                         num_frames=32, device='cpu'):
    """
    Функция для предсказания меток для одного видео.
    
    Параметры:
    - model_path: Путь до обученной модели (.pth файл).
    - video_path: Путь до видеофайла.
    - title: Название видео.
    - description: Описание видео.
    - label_list: Список меток (если None, то возвращает индексы).
    - bert_model_name: Название модели для обработки текста.
    - xclip_model_name: Название модели XCLIP для обработки видео.
    - ast_model_name: Название модели AST для обработки аудио.
    - num_frames: Количество кадров для обработки из видео.
    - device: Устройство для вычислений (CPU или GPU).
    
    Возвращает:
    - Список предсказанных меток (если label_list передан).
    - Список предсказанных индексов (если label_list не передан).
    """
    # Загрузка модели
    model = MultimodalClassificationModel(input_size=1792, num_classes=141, hidden_size=512, hidden_layers=2, verbose=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Создание экземпляра SingleVideoDataset
    video_dataset = SingleVideoDataset(
        video_path=video_path,
        title=title,
        description=description,
        label_list=[] if label_list is None else label_list,
        bert_model_name=bert_model_name,
        xclip_model_name=xclip_model_name,
        ast_model_name=ast_model_name,
        num_frames=num_frames,
        device=device
    )
    
    # Получение комбинированного эмбеддинга из видео, текста и аудио
    with torch.no_grad():
        sample = video_dataset[0]  # SingleVideoDataset всегда возвращает один элемент
        combined_embedding = sample.to(device)

        # Предсказание
        output = model(combined_embedding)
        
        # Применение порога для бинарной классификации (0.5 по умолчанию)
        predicted_labels = (output > 0.5).squeeze().tolist()

        # Если ни один из выходов не перешел порог, выбираем максимальный
        if sum(predicted_labels) == 0:
            max_index = output.argmax().item()  # Находим индекс максимального значения
            predicted_labels[max_index] = 1  # Устанавливаем этот индекс как предсказанную метку

        
    # Проверка, передан ли список меток
    if label_list is None:
        # Если список меток не передан, возвращаем индексы предсказанных меток
        predicted_indices = [i for i, value in enumerate(predicted_labels) if value == 1]
        return predicted_indices
    else:
        # Если список меток передан, возвращаем текстовые метки
        predicted_tags = [label_list[i] for i, value in enumerate(predicted_labels) if value == 1]
        return predicted_tags
