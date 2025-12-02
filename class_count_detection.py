import pandas as pd
import numpy as np

# Чтение CSV файла
df_train = pd.read_csv("./dataset/arxiv/train.csv")

# Получаем метки
y_train = df_train["label"]

# Преобразуем строки с несколькими метками в список меток
# Например, "cs.AI,cs.LG" -> ["cs.AI", "cs.LG"]
all_labels = []
for label_string in y_train:
    # Убираем кавычки и пробелы, разделяем по запятой
    labels = [lbl.strip().replace('"', "") for lbl in str(label_string).split(",")]
    all_labels.extend(labels)

# Преобразуем в numpy массив для использования np.unique
all_labels = np.array(all_labels)

# Посчитайте распределение классов
unique, counts = np.unique(all_labels, return_counts=True)
print("Распределение классов:")
for cls, cnt in zip(unique, counts):
    print(f"Класс {cls}: {cnt} примеров ({cnt/len(all_labels)*100:.1f}%)")

# Коэффициент дисбаланса
if len(counts) > 0:
    imbalance_ratio = max(counts) / min(counts)
    print(f"\nКоэффициент дисбаланса: {imbalance_ratio:.1f}")
    # >5 - сильный дисбаланс
else:
    print("Нет меток для анализа")

# Дополнительная статистика
print(f"\nВсего уникальных меток: {len(unique)}")
print(f"Всего размеченных примеров (с учетом мульти-меток): {len(all_labels)}")
print(f"Количество строк в датасете: {len(df_train)}")
print(f"Среднее количество меток на статью: {len(all_labels)/len(df_train):.2f}")
