from deeppavlov import build_model, configs
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import pymorphy2
import warnings

import logging
import os


# Отключаем предупреждения
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Инициализация
nltk.download("stopwords", quiet=True)
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))


# Чтение текста
def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


text = read_text("./dataset/keywords_text.txt")
print(f"Текст: {len(text)} символов, {len(text.split())} слов")

# Загрузка BERT
model = build_model(configs.embedder.bert_embedder)


# Нормализация слов
def normalize_word(word):
    parsed = morph.parse(word)[0]
    return parsed.normal_form


# Предобработка текста
def preprocess_text(text):
    # Очистка
    text = re.sub(r"[^а-яёa-z\-\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Токенизация и фильтрация
    words = []
    for word in text.split():
        if word not in russian_stopwords and len(word) > 2:
            norm_word = normalize_word(word)
            if len(norm_word) > 2:
                words.append(norm_word)
    return words


# Извлечение кандидатов
def extract_candidates_single_word(words):
    # Частотный анализ
    freq = Counter(words)

    # Фильтрация по части речи и частоте
    candidates = []
    for word, count in freq.items():
        if count >= 2 and len(word) > 3:
            pos = morph.parse(word)[0].tag.POS
            if pos in ["NOUN", "ADJF", "ADJS", "VERB"]:  # Сущ, прил, глагол
                candidates.append(word)

    # Сортировка по частоте
    candidates.sort(key=lambda x: freq[x], reverse=True)
    return candidates[:100]


# Упрощенная версия для быстрого старта
def extract_candidates_with_bigrams(words, include_bigrams=True):
    """
    Упрощенная версия с добавлением только биграмм
    """
    # Частотный анализ
    freq = Counter(words)

    # Фильтрация по части речи и частоте
    candidates = []
    for word, count in freq.items():
        if count >= 2 and len(word) > 3:
            pos = morph.parse(word)[0].tag.POS
            if pos in ["NOUN", "ADJF", "ADJS", "VERB"]:
                candidates.append(word)

    # Добавляем биграммы (самый простой способ)
    if include_bigrams and len(words) >= 2:
        bigrams = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigrams.append(bigram)

        bigram_freq = Counter(bigrams)
        for bigram, count in bigram_freq.items():
            if count >= 2:
                # Простая проверка: оба слова длинные
                w1, w2 = bigram.split()
                if len(w1) > 3 and len(w2) > 3:
                    candidates.append(bigram)

    # Сортировка по частоте
    # Для этого нужно получить частоты для всех кандидатов
    all_freq = {}
    for candidate in candidates:
        if " " in candidate:  # это биграмма
            # Частота биграммы
            # w1, w2 = candidate.split()
            # Приблизительная частота - минимальная из частот слов
            # all_freq[candidate] = min(freq.get(w1, 0), freq.get(w2, 0))
            all_freq[candidate] = bigram_freq.get(candidate, 0)
        else:  # одиночное слово
            all_freq[candidate] = freq.get(candidate, 0)

    candidates.sort(key=lambda x: all_freq.get(x, 0), reverse=True)
    return candidates[:120]  # Увеличиваем лимит


def extract_candidates(words, flag="bigrams"):
    if flag == "bigrams":
        return extract_candidates_with_bigrams(words)
    return extract_candidates_single_word(words)


# Получение эмбеддинга текста
def get_text_embedding(text):
    # Разбиваем текст на части если длинный
    if len(text) > 2000:
        # Разбиваем по предложениям
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > 1500:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Получаем эмбеддинги для каждой части
        embeddings = []
        for chunk in chunks[:5]:  # Ограничиваем 5 частями
            try:
                result = model([chunk[:1000]])  # Берем первые 1000 символов
                embeddings.append(result[5])
            except:
                continue

        if embeddings:
            # УСРЕДНЕНИЕ ЧАНКОВ (MEAN POOLING)
            return np.mean(np.vstack(embeddings), axis=0, keepdims=True)

    # Короткий текст обрабатываем целиком
    try:
        result = model([text[:1000]])
        return result[5]
    except:
        return None


def get_text_embedding_simple(text, model, chunk_size=400, overlap=50):
    """
    Простая версия с чанкованием по словам и перекрытием
    """
    # Разбиваем текст на слова
    words = text.split()

    # Если слов мало, обрабатываем целиком
    if len(words) <= chunk_size:
        try:
            result = model([text])
            return result[5] if len(result) > 5 else result[0]
        except:
            return None

    # Создаем чанки с перекрытием
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    # Ограничиваем количество чанков
    chunks = chunks[:8]

    # Получаем эмбеддинги
    embeddings = []
    for chunk in chunks:
        try:
            result = model([chunk])
            if len(result) > 5:
                embeddings.append(result[5])
        except:
            continue

    if not embeddings:
        return None

    # Усредняем
    return np.mean(np.vstack(embeddings), axis=0, keepdims=True)


# Основной процесс
print("Обработка текста...")
words = preprocess_text(text)
candidates = extract_candidates(words, "bigrams")
# candidates = extract_candidates_single_word(words)

print(f"Кандидатов: {len(candidates)}")

# Получаем эмбеддинг текста
text_embedding = get_text_embedding_simple(text, model)
if text_embedding is None:
    print("Ошибка получения эмбеддинга текста")
    exit()

# Получаем эмбеддинги кандидатов
candidate_embeddings = []
valid_candidates = []

for candidate in candidates[:50]:  # Ограничиваем 50 кандидатов
    try:
        result = model([candidate])
        candidate_embeddings.append(result[5])
        valid_candidates.append(candidate)
    except:
        continue

if not candidate_embeddings:
    print("Ошибка получения эмбеддингов кандидатов")
    exit()

# Расчет сходства
candidate_embeddings = np.vstack(candidate_embeddings)
similarities = cosine_similarity(text_embedding, candidate_embeddings)[0]

# Сортировка и фильтрация
results = sorted(zip(valid_candidates, similarities), key=lambda x: x[1], reverse=True)

# Фильтруем результаты
filtered_results = []
seen_words = set()


CANDIDATES_LIMIT = 20

for word, score in results:
    if score > 0.4 and word not in seen_words:
        filtered_results.append((word, score))
        seen_words.add(word)

    if len(filtered_results) >= CANDIDATES_LIMIT:
        break

# Вывод результатов
print(f"\nТоп-{CANDIDATES_LIMIT} ключевых слов:")
for i, (word, score) in enumerate(filtered_results, 1):
    print(f"{i:2}. {word:20} : {score:.4f}")

# Сохранение
with open("./dataset/keywords_output.txt", "w", encoding="utf-8") as f:
    f.write("Ключевые слова:\n")
    for i, (word, score) in enumerate(filtered_results, 1):
        f.write(f"{i:2}. {word:25} : {score:.4f}\n")
