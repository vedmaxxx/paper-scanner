import os
import re
from typing import List, Tuple, Optional
from keywords import KeywordExtractor
from file_reader import FileReader

USE_BIGRAMS = False
SIMIL_THRESHOLD = 0.475

class DocumentProcessor:
    """Класс для обработки документов и извлечения ключевых слов"""
    
    def __init__(self):
        """Инициализация процессора документов"""
        self.keyword_extractor = KeywordExtractor()
        self.file_reader = FileReader()
        self.keyword_extractor.load_model()  # Загружаем модель при инициализации
    
    def extract_label_from_file(self, file_path: str) -> str:
        """
        Извлечение названия документа из имени файла
        
        Args:
            file_path: путь к файлу
            
        Returns:
            название документа (имя файла без расширения)
        """
        # Используем имя файла без расширения как название
        filename = os.path.basename(file_path)
        label = os.path.splitext(filename)[0]
        
        # Если название слишком длинное, обрезаем
        if len(label) > 80:
            label = label[:77] + "..."
        
        return label
    
    def read_text_file(self, file_path: str) -> str:
        """
        Чтение текста из файла любого формата
        
        Args:
            file_path: путь к файлу
            
        Returns:
            текст документа
        """
        try:
            text = self.file_reader.read_file(file_path)
            if text is None:
                raise Exception(f"Не удалось прочитать файл: {file_path}")
            return text
        except Exception as e:
            raise Exception(f"Ошибка чтения файла: {str(e)}")
    
    def extract_keywords_from_document(self, file_path: str, 
                                     max_keywords: int = 45) -> List[str]:
        """
        Извлечение ключевых слов из документа
        
        Args:
            file_path: путь к файлу
            max_keywords: максимальное количество ключевых слов
            
        Returns:
            список ключевых слов
        """
        
        # Сначала читаем файл
        text = self.read_text_file(file_path)
        
        # Если текст слишком короткий, добавляем меньше ключевых слов
        if len(text) < 500:
            max_keywords = 15
        if len(text) > 50000:
            max_keywords = 65
        # Извлекаем ключевые слова с помощью KeywordExtractor
        keywords_with_scores = self.keyword_extractor.extract_keywords_from_text(text, USE_BIGRAMS , SIMIL_THRESHOLD,max_keywords)
        print("Ключевые слова с оценками:\n")
        print(keywords_with_scores)
        print("\n")
        # Возвращаем только ключевые слова (без оценок)
        return [keyword for keyword, _ in keywords_with_scores]

    def process_document(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Полная обработка документа
        
        Args:
            file_path: путь к файлу
            
        Returns:
            кортеж (название документа, список ключевых слов)
        """
        # Извлекаем название из имени файла
        label = self.extract_label_from_file(file_path)
        
        # Извлекаем ключевые слова
        keywords = self.extract_keywords_from_document(file_path)
        
        return label, keywords
    
    def add_paper_to_system(self, db, file_path: str) -> dict:
        """
        Добавление документа в систему (в БД)
        
        Args:
            db: экземпляр DocumentDB
            file_path: путь к файлу
            
        Returns:
            словарь с результатом операции
        """
        try:
            # Обработка документа
            label, keywords = self.process_document(file_path)
            
            # Проверка на минимальное количество ключевых слов
            if len(keywords) < 3:
                return {
                    'success': False,
                    'message': f'Из документа извлечено слишком мало ключевых слов: {len(keywords)}'
                }
            
            # Добавление в БД
            doc_id = db.add_document(keywords, label)
            
            return {
                'success': True,
                'doc_id': doc_id,
                'label': label,
                'keywords_count': len(keywords),
                'keywords': keywords[:5]  # Первые 5 для отображения
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_relevant_papers(self, db, file_path: str, 
                          similarity_threshold: float = 0.6,
                          max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Поиск релевантных документов в БД
        
        Args:
            db: экземпляр DocumentDB
            file_path: путь к файлу для поиска
            similarity_threshold: порог сходства (по умолчанию 60%)
            max_results: максимальное количество результатов
            
        Returns:
            список кортежей (название документа, сходство)
        """
        try:
            # Извлекаем ключевые слова из документа
            _, keywords = self.process_document(file_path)
            
            if not keywords:
                print("Не удалось извлечь ключевые слова из документа")
                return []
            
            # Поиск похожих документов в БД
            similar_docs = db.search_by_similar_keywords(
                keywords, 
                threshold=similarity_threshold
            )
            
            # Формируем результаты
            results = []
            for doc in similar_docs:
                if doc['similarity'] >= similarity_threshold:
                    results.append((doc['label'], doc['similarity']))
            
            # Сортируем по убыванию сходства
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Ошибка поиска релевантных документов: {e}")
            return []
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Получение информации о файле
        
        Args:
            file_path: путь к файлу
            
        Returns:
            словарь с информацией о файле
        """
        return self.file_reader.get_file_info(file_path)