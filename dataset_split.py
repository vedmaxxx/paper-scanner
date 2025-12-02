import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Укажите путь к вашему файлу
INPUT_FILE = (
    "./dataset/arxiv/train_balanced_source.csv"  # измените на путь к вашему файлу
)
OUTPUT_DIR = "dataset/arxiv"

# Создаем папку
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Читаем файл
df = pd.read_csv(INPUT_FILE)

# Объединяем заголовок и абстракт
df["text"] = df["titles"] + ". " + df["abstracts"]


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


df["label"] = df["terms"].apply(clean_labels)

# Убираем пустые метки
df = df[df["label"] != ""]
df = df[["text", "label"]]

# Сохраняем исходный обработанный файл
df.to_csv(f"{OUTPUT_DIR}/source_processed_dataset.csv", index=False)


# Разделяем
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Сохраняем
train_df.to_csv(f"{OUTPUT_DIR}/train-balanced.csv", index=False)
valid_df.to_csv(f"{OUTPUT_DIR}/valid-balanced.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test-balanced.csv", index=False)

print(f"Готово! Файлы в папке '{OUTPUT_DIR}':")
print(f"  train.csv: {len(train_df)} записей")
print(f"  valid.csv: {len(valid_df)} записей")
print(f"  test.csv:  {len(test_df)} записей")
