# import pandas as pd
# import numpy as np

# # Чтение данных
# df_train = pd.read_csv("./dataset/arxiv/train.csv")

# # Анализ распределения
# all_labels = []
# for label_string in df_train["label"]:
#     labels = [lbl.strip().replace('"', "") for lbl in str(label_string).split(",")]
#     all_labels.extend(labels)

# all_labels = np.array(all_labels)
# unique, counts = np.unique(all_labels, return_counts=True)

# # Фильтруем классы с достаточным количеством примеров
# MIN_SAMPLES = 50  # минимальное количество примеров для класса
# valid_classes = unique[counts >= MIN_SAMPLES]

# print(f"Всего уникальных классов: {len(unique)}")
# print(f"Классов с >= {MIN_SAMPLES} примерами: {len(valid_classes)}")
# print(f"Процент сохраненных классов: {len(valid_classes)/len(unique)*100:.1f}%")

# # Посмотрим на эти классы
# print(f"\nКлассы с достаточным количеством данных (>= {MIN_SAMPLES}):")
# for cls in valid_classes[:20]:  # покажем первые 20
#     cnt = counts[unique == cls][0]
#     print(f"  {cls}: {cnt} примеров ({cnt/len(all_labels)*100:.2f}%)")


import pandas as pd
import numpy as np
from collections import defaultdict

# Чтение данных
df_train = pd.read_csv("./dataset/arxiv/train.csv")


# Функция для извлечения меток из строки
def extract_labels(label_string):
    labels = [
        lbl.strip().replace('"', "").replace("'", "")
        for lbl in str(label_string).split(",")
    ]
    return labels


# Анализ распределения
all_labels = []
for label_string in df_train["label"]:
    labels = extract_labels(label_string)
    all_labels.extend(labels)

all_labels = np.array(all_labels)
unique, counts = np.unique(all_labels, return_counts=True)

# Фильтруем классы с достаточным количеством примеров
MIN_SAMPLES = 50  # минимальное количество примеров для класса
valid_classes = unique[counts >= MIN_SAMPLES]

print(f"Всего уникальных классов: {len(unique)}")
print(f"Классов с >= {MIN_SAMPLES} примерами: {len(valid_classes)}")
print(f"Процент сохраненных классов: {len(valid_classes)/len(unique)*100:.1f}%")

# Создаем словарь для хранения индексов статей по классам
class_to_indices = defaultdict(list)

# Собираем индексы статей для каждого валидного класса
for idx, label_string in enumerate(df_train["label"]):
    labels = extract_labels(label_string)
    for label in labels:
        if label in valid_classes:
            class_to_indices[label].append(idx)

# Определяем минимальное количество статей среди всех классов для андерсемплинга
min_samples_per_class = min(len(indices) for indices in class_to_indices.values())
print(
    f"\nМинимальное количество примеров в классе после фильтрации: {min_samples_per_class}"
)

# Андерсемплинг: для каждого класса берем случайные min_samples_per_class индексов
undersampled_indices = set()

for label, indices in class_to_indices.items():
    if len(indices) > min_samples_per_class:
        # Выбираем случайные min_samples_per_class индексов
        selected = np.random.choice(indices, min_samples_per_class, replace=False)
    else:
        # Берем все индексы, если их меньше минимального количества
        selected = indices

    undersampled_indices.update(selected)

print(f"Количество уникальных статей после андерсемплинга: {len(undersampled_indices)}")

# Создаем новый датафрейм с андерсемплированными данными
df_train_balanced = df_train.iloc[list(undersampled_indices)].reset_index(drop=True)

# Проверяем распределение после андерсемплинга
print("\nРаспределение после андерсемплинга:")
all_labels_balanced = []
for label_string in df_train_balanced["label"]:
    labels = extract_labels(label_string)
    # Фильтруем только валидные метки
    valid_labels = [label for label in labels if label in valid_classes]
    all_labels_balanced.extend(valid_labels)

all_labels_balanced = np.array(all_labels_balanced)
unique_balanced, counts_balanced = np.unique(all_labels_balanced, return_counts=True)

for cls, cnt in zip(unique_balanced, counts_balanced):
    print(f"  {cls}: {cnt} примеров")

# Сохраняем сбалансированный датасет
output_path = "./dataset/arxiv/train_balanced.csv"
df_train_balanced.to_csv(output_path, index=False)
print(f"\nСбалансированный датасет сохранен в: {output_path}")

# Дополнительная статистика
print(f"\nДополнительная статистика:")
print(f"Размер исходного датасета: {len(df_train)} строк")
print(f"Размер сбалансированного датасета: {len(df_train_balanced)} строк")
print(f"Процент сохраненных данных: {len(df_train_balanced)/len(df_train)*100:.1f}%")

# Коэффициент дисбаланса после андерсемплинга
if len(counts_balanced) > 0:
    imbalance_ratio_balanced = max(counts_balanced) / min(counts_balanced)
    print(
        f"Коэффициент дисбаланса после андерсемплинга: {imbalance_ratio_balanced:.1f}"
    )
