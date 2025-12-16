import warnings
import logging
import os
import re
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import pymorphy2

# Отключаем предупреждения
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class KeywordExtractor:
    """Класс для извлечения ключевых слов из текста с использованием BERT"""
    
    def __init__(self, model_name: str = "bert"):
        """
        Инициализация экстрактора ключевых слов
        
        Args:
            model_name: название модели (пока поддерживается только BERT)
        """
        self.model_name = model_name
        self.model = None
        self.morph = None
        self.russian_stopwords = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Инициализация всех необходимых компонентов"""
        # Инициализация NLTK и стоп-слов
        try:
            nltk.download("stopwords", quiet=True)
            self.russian_stopwords = set(stopwords.words("russian"))
        except:
            self.russian_stopwords = set()
        
        # Инициализация морфологического анализатора
        self.morph = pymorphy2.MorphAnalyzer()
    
    def load_model(self):
        """Загрузка модели BERT"""
        if self.model is None:
            try:
                from deeppavlov import build_model, configs
                self.model = build_model(configs.embedder.bert_embedder)
            except ImportError as e:
                raise ImportError(
                    "Для использования BERT модели необходимо установить deeppavlov. "
                    "Установите: pip install deeppavlov"
                )
        return self.model
    
    def normalize_word(self, word: str) -> str:
        """
        Нормализация слова с использованием pymorphy2
        
        Args:
            word: исходное слово
            
        Returns:
            нормализованная форма слова
        """
        if word.startswith('-'):
            word = word.lstrip('-')
            # Если после удаления дефисов слово стало пустым, возвращаем исходное
            if not word:
                word = '-'
        parsed = self.morph.parse(word)[0]
        return parsed.normal_form
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Предобработка текста: очистка, токенизация, нормализация
        
        Args:
            text: исходный текст
            
        Returns:
            список предобработанных слов
        """
        # Очистка текста
        text = re.sub(r"[^а-яёa-z\-\s]", " ", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        
        # Токенизация и фильтрация
        words = []
        for word in text.split():
            if word not in self.russian_stopwords and len(word) > 2:
                norm_word = self.normalize_word(word)
                if len(norm_word) > 2:
                    words.append(norm_word)
        
        return words

    def extract_candidates(self, words: List[str], 
                                       include_bigrams:bool, 
                                       min_frequency: int) -> List[str]:
        """
        Извлечение кандидатов с учетом биграмм
        
        Args:
            words: список предобработанных слов
            min_frequency: минимальная частота для кандидата
            include_bigrams: включать ли биграммы
            
        Returns:
            список кандидатов (слов и биграмм)
        """
        # Частотный анализ для одиночных слов
        freq = Counter(words)
        
        # Извлечение одиночных слов-кандидатов
        candidates = []
        for word, count in freq.items():
            if count >= min_frequency and len(word) > 3:
                pos = self.morph.parse(word)[0].tag.POS
                if pos in ["NOUN"]:#, "ADJF", "ADJS", "VERB"
                    candidates.append(word)
        
        # Добавление биграмм
        if include_bigrams and len(words) >= 2:
            bigrams = []
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
            
            bigram_freq = Counter(bigrams)
            for bigram, count in bigram_freq.items():
                if count >= min_frequency:
                    # Простая проверка: оба слова длинные
                    w1, w2 = bigram.split()
                    if len(w1) > 3 and len(w2) > 3:
                        candidates.append(bigram)
        
        # Сортировка по частоте
        all_freq = {}
        for candidate in candidates:
            if " " in candidate:  # это биграмма
                all_freq[candidate] = bigram_freq.get(candidate, 0)
            else:  # одиночное слово
                all_freq[candidate] = freq.get(candidate, 0)
        
        candidates.sort(key=lambda x: all_freq.get(x, 0), reverse=True)
        return candidates[:120]
    
    def get_text_embedding(self, text: str, chunk_size: int, 
                                 overlap: int) -> Optional[np.ndarray]:
        """
        Получение эмбеддинга текста с чанкованием
        
        Args:
            text: исходный текст
            chunk_size: размер чанка в словах
            overlap: перекрытие между чанками
            
        Returns:
            эмбеддинг текста или None в случае ошибки
        """
        if self.model is None:
            self.load_model()
        
        # Разбиваем текст на слова
        words = text.split()
        
        # Если слов мало, обрабатываем целиком
        if len(words) <= chunk_size:
            try:
                result = self.model([text])
                return result[5] if len(result) > 5 else result[0]
            except Exception as e:
                print(f"Ошибка получения эмбеддинга: {e}")
                return None
        
        # Создаем чанки с перекрытием
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i: i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        

        # Получаем эмбеддинги для каждого чанка
        embeddings = []
        for chunk in chunks:
            try:
                result = self.model([chunk])
                if len(result) > 5:
                    embeddings.append(result[5])
            except Exception as e:
                print(f"Ошибка обработки чанка: {e}")
                continue
        
        if not embeddings:
            return None
        
        # Усредняем эмбеддинги чанков
        return np.mean(np.vstack(embeddings), axis=0, keepdims=True)
    
    def get_candidate_embeddings(self, candidates: List[str], 
                                limit: int) -> Tuple[List[str], List[np.ndarray]]:
        """
        Получение эмбеддингов для кандидатов
        
        Args:
            candidates: список кандидатов
            limit: максимальное количество кандидатов для обработки
            
        Returns:
            кортеж (валидные кандидаты, их эмбеддинги)
        """
        if self.model is None:
            self.load_model()
        
        candidate_embeddings = []
        valid_candidates = []
        
        for candidate in candidates[:limit]:
            try:
                result = self.model([candidate])
                candidate_embeddings.append(result[5])
                valid_candidates.append(candidate)
            except Exception as e:
                print(f"Ошибка получения эмбеддинга для '{candidate}': {e}")
                continue
        
        return valid_candidates, candidate_embeddings
    
    def calculate_similarity(self, text_embedding: np.ndarray, 
                           candidate_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Расчет сходства между эмбеддингом текста и эмбеддингами кандидатов
        
        Args:
            text_embedding: эмбеддинг текста
            candidate_embeddings: список эмбеддингов кандидатов
            
        Returns:
            массив значений сходства
        """
        candidate_embeddings = np.vstack(candidate_embeddings)
        similarities = cosine_similarity(text_embedding, candidate_embeddings)[0]
        return similarities
    
    def filter_and_rank_keywords(self, candidates: List[str], 
                                similarities: np.ndarray,
                                similarity_threshold: float,
                                max_keywords: int) -> List[Tuple[str, float]]:
        # Сортировка по сходству
        results = sorted(zip(candidates, similarities), 
                        key=lambda x: x[1], reverse=True)
        
        # Фильтрация результатов
        filtered_results = []
        seen_words = set()
        
        for word, score in results:
            if score > similarity_threshold and word not in seen_words:
                filtered_results.append((word, float(score)))
                seen_words.add(word)
            
            if len(filtered_results) >= max_keywords:
                break
        
        return filtered_results
    
    def extract_keywords_from_text(self, text: str, 
                                  use_bigrams: bool = True,
                                  similarity_threshold: float = 0.4,
                                  max_keywords: int = 40) -> List[Tuple[str, float]]:
        print("Обработка текста...")
        
        # Предобработка текста
        words = self.preprocess_text(text)
        
        # Извлечение кандидатов
        candidates = self.extract_candidates(words, use_bigrams, 2)
        
        print(f"Найдено кандидатов: {len(candidates)}")
        
        if not candidates:
            return []
        
        # Получаем эмбеддинг текста
        text_embedding = self.get_text_embedding(text,100,20)
        if text_embedding is None:
            print("Не удалось получить эмбеддинг текста")
            return []
        
        # Получаем эмбеддинги кандидатов
        valid_candidates, candidate_embeddings = self.get_candidate_embeddings(candidates,120)
        
        if not candidate_embeddings:
            print("Не удалось получить эмбеддинги кандидатов")
            return []
        
        # Расчет сходства
        similarities = self.calculate_similarity(text_embedding, candidate_embeddings)
        
        # Фильтрация и ранжирование
        keywords = self.filter_and_rank_keywords(
            valid_candidates, similarities, similarity_threshold, max_keywords
        )
        
        return keywords