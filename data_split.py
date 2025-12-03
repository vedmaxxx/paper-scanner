import pandas as pd
import os
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "./dataset/"


def split_dataset(dataset_df):
    print("=== РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ, ТЕСТОВУЮ И ВАЛИДИРУЮЩУЮ ВЫБОРКИ ===")
    train_df, temp_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Сохранение в файлы
    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    valid_df.to_csv(f"{OUTPUT_DIR}/valid.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print(f"  train.csv: {len(train_df)} записей")
    print(f"  valid.csv: {len(valid_df)} записей")
    print(f"  test.csv:  {len(test_df)} записей")
    print()
