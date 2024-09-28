import pandas as pd

# 1. Загрузка CSV-файла с иерархией меток
csv_file = r"C:\Users\user\Desktop\train_dataset_tag_video\baseline\IAB_tags.csv"  # Замените на путь к вашему CSV-файлу
data = pd.read_csv(csv_file)

# 2. Создание списка меток, объединяя уровни
labels = set()

for index, row in data.iterrows():
    # Объединяем непустые уровни иерархии в строку с разделителем ": "
    label = ": ".join([str(x) for x in row if pd.notna(x)])
    if label:
        labels.add(label)

# Преобразуем в отсортированный список для удобства
label_list = sorted(labels)

# 3. Сохранение меток в текстовый файл
output_file = "labels_list.txt"  # Имя выходного файла
with open(output_file, "w", encoding="utf-8") as f:
    for label in label_list:
        f.write(f"{label}\n")

# 4. Вывод подтверждения и примера сохраненных меток
print(f"Список всех меток сохранен в файл: {output_file}")
print("Пример меток:")
for label in label_list[:5]:  # Показываем первые 5 меток
    print(label)
