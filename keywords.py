from deeppavlov import build_model, configs
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import re

# Инициализация модели эмбеддингов
model = build_model(configs.embedder.bert_embedder)

# Ваш текст
full_text = "В статье рассматривается процесс составления расписания в вузе. Авторами предложен алгоритм поддержки принятия решений, который реализован в виде программного обеспечения для составления учебного расписания с применением технологии RDF для хранения слабоструктурированных данных. Эффективность принятия решений при составлении учебного расписания показана на примере составления расписания кафедры."


# Улучшенная функция извлечения кандидатов
def extract_candidates(text):
    # 1. Токенизация с учетом русского языка
    words = re.findall(r"\b[а-яА-ЯёЁ]+\b", text.lower())

    # 2. Удаление стоп-слов (базовый список)
    stop_words = {
        "в",
        "на",
        "с",
        "по",
        "для",
        "при",
        "из",
        "от",
        "до",
        "за",
        "и",
        "или",
        "а",
        "но",
        "же",
        "как",
        "что",
        "это",
        "то",
        "не",
        "у",
        "к",
        "же",
        "бы",
        "во",
        "со",
        "об",
        "тот",
        "этот",
    }
    candidates = [word for word in words if word not in stop_words and len(word) > 2]

    # 3. Группировка одинаковых слов (упрощенная нормализация)
    # В реальном проекте здесь должна быть лемматизация
    word_counts = defaultdict(int)
    for word in candidates:
        # Базовая нормализация (можно улучшить с помощью pymystem3)
        if word.endswith("ия"):
            word = word[:-2] + "ие"  # расписания → расписание
        elif word.endswith("ии"):
            word = word[:-2] + "ия"  # технологии → технология
        word_counts[word] += 1

    # Возвращаем уникальные слова с частотой
    return list(word_counts.keys())


# Получение эмбеддингов для всего текста
result = model([full_text])
sentence_embedding = result[5]  # sent_mean_embs

# Получение эмбеддингов для слов-кандидатов
candidates = extract_candidates(full_text)
print(f"Уникальные кандидаты: {candidates}")

candidate_embeddings = []
valid_candidates = []

for candidate in candidates:
    try:
        cand_result = model([candidate])
        cand_embedding = cand_result[5]
        candidate_embeddings.append(cand_embedding)
        valid_candidates.append(candidate)
    except:
        continue

if not candidate_embeddings:
    print("Не удалось получить эмбеддинги для кандидатов")
else:
    # Расчет сходства
    candidate_embeddings = np.vstack(candidate_embeddings)
    sentence_embedding = sentence_embedding.reshape(1, -1)

    similarities = cosine_similarity(sentence_embedding, candidate_embeddings)

    # Сортировка результатов
    scored_keywords = list(zip(valid_candidates, similarities[0]))
    scored_keywords.sort(key=lambda x: x[1], reverse=True)

    # Вывод результатов
    print("\nТоп-10 ключевых слов:")
    for i, (keyword, score) in enumerate(scored_keywords[:10], 1):
        print(f"{i}. {keyword}: {score:.4f}")
