import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Читаем и преобразуем данные
df = pd.read_csv('dataset/train.csv')  # Ваш исходный файл
df_t = pd.read_csv('dataset/test.csv')  # Ваш исходный файл
classes = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

def get_labels(row):
    labels = []
    for cls in classes:
        if row[cls] == 1:
            labels.append(cls)
    return ','.join(labels)

# Создаем новый DataFrame
new_df = pd.DataFrame()
new_df['text'] = df['ABSTRACT']
new_df['label'] = df.apply(get_labels, axis=1)

# Разделяем на train и valid (80/20)
train_df, valid_df = train_test_split(new_df, test_size=0.2, random_state=42)

# Создаем папку
os.makedirs('dataset/formatted', exist_ok=True)

# Сохраняем
train_df.to_csv('dataset/formatted/train.csv', index=False)
valid_df.to_csv('dataset/formatted/valid.csv', index=False)

print(f"Train: {len(train_df)} samples")
print(f"Valid: {len(valid_df)} samples")