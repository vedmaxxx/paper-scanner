import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import itertools
from collections import Counter

class DocumentDB:
    def __init__(self, db_path: str = 'documents.db', model = None):
        """Инициализация базы данных и создание таблицы если её нет"""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.model = model  # Модель DeepPavlov для семантических эмбеддингов
        self.create_table()
    
    def create_table(self):
        """Создание таблицы документов"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keywords TEXT NOT NULL,
                label TEXT NOT NULL
            )
        ''')
        self.conn.commit()
    
    def add_document(self, keywords: List[str], label: str) -> int:
        """
        Добавление документа в базу данных
        
        Args:
            keywords: список ключевых слов/фраз
            label: название документа
            
        Returns:
            id добавленного документа
        """
        keywords_json = json.dumps(keywords, ensure_ascii=False)
        
        self.cursor.execute(
            'INSERT INTO documents (keywords, label) VALUES (?, ?)',
            (keywords_json, label)
        )
        self.conn.commit()
        
        return self.cursor.lastrowid
    
    
    def search_by_similar_keywords(self, query_keywords: List[str], 
                                  threshold: float = 0.3) -> List[Dict]:
        """
        Поиск документов по похожим ключевым словам с использованием
        модифицированного коэффициента Жаккара с учетом семантической близости
        
        Args:
            query_keywords: список ключевых слов для поиска
            threshold: порог сходства (0-1), чем выше, тем строже
            
        Returns:
            Список документов с коэффициентом сходства
        """
        self.cursor.execute(
            'SELECT id, keywords, label FROM documents'
        )
        rows = self.cursor.fetchall()
        
        results = []
        query_set = set(query_keywords)
        
        for row in rows:
            doc_keywords = json.loads(row[1])
            doc_set = set(doc_keywords)
            
            # Рассчитываем коэффициент сходства
            if not query_set or not doc_set:
                similarity = 0
            elif self.model:
                # Используем модифицированный коэффициент Жаккара с семантической близостью
                #print("Входной документ:", query_set)
                #print(row[2],": ",doc_set)
                similarity = self._calculate_semantic_jaccard(query_set, doc_set)
            else:
                # Классический коэффициент Жаккара
                intersection = len(query_set.intersection(doc_set))
                union = len(query_set.union(doc_set))
                similarity = intersection / union if union > 0 else 0
            
            print(row[2]," имеет схожесть: ",similarity)
            if similarity >= threshold:
                results.append({
                    'id': row[0],
                    'keywords': doc_keywords,
                    'label': row[2],
                    'similarity': round(similarity, 3)
                })
        
        # Сортируем по убыванию сходства
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def _calculate_semantic_jaccard(self, set1: set, set2: set) -> float:
        """
        Вычисление модифицированного коэффициента Жаккара 
        с учётом семантической близости всех слов
        """
        # Классический коэффициент Жаккара для точных совпадений
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        classical_score = intersection / union
        
        # Если есть точные совпадения, они уже дают хороший балл
        if classical_score > 0.4:
            #print("Возвращен classic score 158")
            return classical_score
        
        # Если нет модели или множества пусты, возвращаем классический результат
        if not self.model or not set1 or not set2:
            #print("Возвращен classic score 163")
            return classical_score
        
        try:
            # Получаем эмбеддинги для всех слов
            all_words = list(set1 | set2)
            results = self.model(all_words)
            
            if len(results) > 5:
                embeddings = results[5]
                
                # Создаем словарь эмбеддингов
                word_vectors = {}
                valid_words = []
                
                for i, word in enumerate(all_words):
                    if i < len(embeddings):
                        vec = embeddings[i]
                        if np.linalg.norm(vec) > 0:  # Проверяем, что вектор не нулевой
                            word_vectors[word] = vec
                            valid_words.append(word)
                
                # Если для большинства слов нет векторов, возвращаем классический результат
                if len(valid_words) < max(len(set1), len(set2)) * 0.5:
                    #print("Возвращен classic score 187")
                    return classical_score
                
                # Создаем списки слов из каждого множества с валидными векторами
                valid_set1 = [word for word in set1 if word in word_vectors]
                valid_set2 = [word for word in set2 if word in word_vectors]
                
                if not valid_set1 or not valid_set2:
                    #print("Возвращен classic score 195")
                    return classical_score
                
                # Вычисляем матрицу сходств между всеми словами
                similarity_matrix = np.zeros((len(valid_set1), len(valid_set2)))
                
                for i, word1 in enumerate(valid_set1):
                    vec1 = word_vectors[word1]
                    norm1 = np.linalg.norm(vec1)
                    
                    for j, word2 in enumerate(valid_set2):
                        # Если слова одинаковые, сходство = 1
                        if word1 == word2:
                            similarity_matrix[i, j] = 1.0
                        else:
                            vec2 = word_vectors[word2]
                            norm2 = np.linalg.norm(vec2)
                            
                            if norm1 > 0 and norm2 > 0:
                                cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                                if cos_sim > 0.25:
                                    similarity_matrix[i, j] = max(0, cos_sim)  # Ограничиваем снизу 0
                                else:
                                    similarity_matrix[i, j] = 0  # Ограничиваем снизу 0
                
                # Вычисляем семантическое пересечение
                # Среднее максимальное сходство в обе стороны
                sum_a_to_b = 0
                for i in range(len(valid_set1)):
                    max_sim = np.max(similarity_matrix[i, :])
                    sum_a_to_b += max_sim
                
                sum_b_to_a = 0
                for j in range(len(valid_set2)):
                    max_sim = np.max(similarity_matrix[:, j])
                    sum_b_to_a += max_sim
                
                semantic_score = 0.0
                if len(valid_set1) > 0 and len(valid_set2) > 0:
                    # Мягкий коэффициент Жаккара
                    soft_intersection = (sum_a_to_b + sum_b_to_a) / 2
                    soft_union = len(valid_set1) + len(valid_set2) - soft_intersection
                    
                    if soft_union > 0:
                        semantic_score = soft_intersection / soft_union
                #print("semantic_score :",semantic_score)
                combined_score = classical_score + 0.8 * semantic_score
                #print("Возвращен combined_score 250 :",combined_score)
                
                # Ограничиваем сверху 1.0 и снизу 0.0
                return min(1.0, max(0.0, combined_score))
                
            else:
                #print("Возвращен classic score 261")
                return classical_score
                
        except Exception as e:
            print(f"Ошибка при семантическом поиске: {e}")
            return classical_score
        
    def get_all_documents(self) -> List[Dict]:
        """Получение всех документов из базы данных"""
        self.cursor.execute(
            'SELECT id, keywords, label FROM documents ORDER BY id'
        )
        rows = self.cursor.fetchall()
        
        return [{
            'id': row[0],
            'keywords': json.loads(row[1]),
            'label': row[2]
        } for row in rows]
    
    def update_document(self, doc_id: int, keywords: List[str] = None, 
                       label: str = None) -> bool:
        """
        Обновление документа
        
        Args:
            doc_id: ID документа для обновления
            keywords: новые ключевые слова (если None - не обновлять)
            label: новое название (если None - не обновлять)
            
        Returns:
            True если документ обновлен, False если документ не найден
        """
        updates = []
        params = []
        
        if keywords is not None:
            updates.append('keywords = ?')
            params.append(json.dumps(keywords, ensure_ascii=False))
        
        if label is not None:
            updates.append('label = ?')
            params.append(label)
        
        if not updates:
            return False
        
        params.append(doc_id)
        
        self.cursor.execute(
            f'UPDATE documents SET {", ".join(updates)} WHERE id = ?',
            params
        )
        self.conn.commit()
        
        return self.cursor.rowcount > 0
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Удаление документа по ID
        
        Returns:
            True если документ удален, False если документ не найден
        """
        self.cursor.execute(
            'DELETE FROM documents WHERE id = ?',
            (doc_id,)
        )
        self.conn.commit()
        
        return self.cursor.rowcount > 0
    
    def close(self):
        """Закрытие соединения с базой данных"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()