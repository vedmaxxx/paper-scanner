import pandas as pd
import numpy as np


# Функция для извлечения меток из строки
def extract_labels(label_string):
    labels = [
        lbl.strip().replace('"', "").replace("'", "")
        for lbl in str(label_string).split(",")
    ]
    return labels


# Чтение данных
# df_dataset = pd.read_csv("./dataset/arxiv/train.csv")


def filter_dataset(df_dataset):

    # Анализ распределения
    all_labels = []
    for label_string in df_dataset["label"]:
        labels = extract_labels(label_string)
        all_labels.extend(labels)

    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)

    # Фильтруем классы с более чем 15 примерами
    MIN_SAMPLES = 15  # минимальное количество примеров для класса
    frequent_classes = set(unique[counts > MIN_SAMPLES])  # частые классы

    print(f"Всего уникальных классов: {len(unique)}")
    print(f"Частых классов (>{MIN_SAMPLES} примеров): {len(frequent_classes)}")
    print(
        f"Редких классов (≤{MIN_SAMPLES} примеров): {len(unique) - len(frequent_classes)}"
    )

    # Фильтруем строки: оставляем только те, где есть хотя бы один частый класс
    filtered_rows = []
    removed_rows_count = 0

    for idx, row in df_dataset.iterrows():
        labels = extract_labels(row["label"])

        # Проверяем, есть ли в строке хотя бы один частый класс
        has_frequent_class = any(label in frequent_classes for label in labels)

        if has_frequent_class:
            # Оставляем только частые метки
            frequent_labels = [label for label in labels if label in frequent_classes]
            new_row = row.copy()
            new_row["label"] = ",".join(frequent_labels)
            filtered_rows.append(new_row)
        else:
            removed_rows_count += 1

    # Создаем новый датафрейм
    df_filtered = pd.DataFrame(filtered_rows).reset_index(drop=True)

    # Сохраняем отфильтрованный датасет
    output_path = "./filtered/source_filtered.csv"
    df_filtered.to_csv(output_path, index=False)

    print(f"\n=== РЕЗУЛЬТАТЫ ФИЛЬТРАЦИИ ===")
    print(f"Размер исходного датасета: {len(df_dataset)} строк")
    print(f"Размер отфильтрованного датасета: {len(df_filtered)} строк")
    print(
        f"Удалено строк: {removed_rows_count} ({removed_rows_count/len(df_dataset)*100:.1f}%)"
    )

    # Анализ отфильтрованного датасета
    if len(df_filtered) > 0:
        all_labels_filtered = []
        for label_string in df_filtered["label"]:
            labels = extract_labels(label_string)
            all_labels_filtered.extend(labels)

        all_labels_filtered = np.array(all_labels_filtered)
        unique_filtered, counts_filtered = np.unique(
            all_labels_filtered, return_counts=True
        )

        print(f"Количество примеров в отфильтрованном датасете: {len(df_filtered)}")

        print(f"\n=== РАСПРЕДЕЛЕНИЕ В ОТФИЛЬТРОВАННОМ ДАТАСЕТЕ ===")
        print(f"Уникальных классов осталось: {len(unique_filtered)}")

        # Статистика по классам
        min_count = counts_filtered.min() if len(counts_filtered) > 0 else 0
        max_count = counts_filtered.max() if len(counts_filtered) > 0 else 0
        avg_count = counts_filtered.mean() if len(counts_filtered) > 0 else 0

        print(f"Минимальное количество примеров в классе: {min_count}")
        print(f"Максимальное количество примеров в классе: {max_count}")
        print(f"Среднее количество примеров в классе: {avg_count:.1f}")

        # Проверяем, все ли классы имеют >15 примеров
        rare_in_filtered = sum(1 for cnt in counts_filtered if cnt <= MIN_SAMPLES)
        if rare_in_filtered == 0:
            print(f"✓ Все классы имеют >{MIN_SAMPLES} примеров")
        else:
            print(
                f"⚠ Внимание: {rare_in_filtered} классов имеют ≤{MIN_SAMPLES} примеров"
            )

    print(f"\nОтфильтрованный датасет сохранен в: {output_path}")
    print()

    return df_filtered
