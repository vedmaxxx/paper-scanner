import pandas as pd
import os
from sklearn.model_selection import train_test_split

INPUT_FILE = "./dataset/source/raw_source.csv"
OUTPUT_DIR = "./dataset/formatted/"


# Очищаем метки
def clean_labels(label_str):
    if isinstance(label_str, str):
        label_str = (
            label_str.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )
        labels = [l.strip() for l in label_str.split(",")]
        return ",".join(sorted(labels))
    return ""


def format_dataset():
    # Создаем папку
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== ФОРМАТИРОВАНИЕ ДАТАСЕТА ===")

    df = pd.read_csv(INPUT_FILE)

    # Объединяем заголовок и абстракт
    df["text"] = df["titles"] + ". " + df["abstracts"]

    df["label"] = df["terms"].apply(clean_labels)

    df = df[df["label"] != ""]
    df = df[["text", "label"]]

    print(f"Готово! Файлы в папке '{OUTPUT_DIR}':")
    print(f"  formatted_source_dataset.csv: {len(df)} записей")
    df.to_csv(f"{OUTPUT_DIR}/formatted_source.csv", index=False)
    print()
    return df
