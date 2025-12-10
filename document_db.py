import sqlite3
import json
from typing import List, Dict, Optional, Tuple
import itertools
from collections import Counter

class DocumentDB:
    def __init__(self, db_path: str = 'documents.db'):
        """Инициализация базы данных и создание таблицы если её нет"""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
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
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Получение документа по ID"""
        self.cursor.execute(
            'SELECT id, keywords, label FROM documents WHERE id = ?',
            (doc_id,)
        )
        row = self.cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'keywords': json.loads(row[1]),
                'label': row[2]
            }
        return None
    
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """
        Поиск документов по точному совпадению ключевого слова
        
        Args:
            keyword: ключевое слово для поиска
            
        Returns:
            Список документов, содержащих ключевое слово
        """
        self.cursor.execute(
            'SELECT id, keywords, label FROM documents'
        )
        rows = self.cursor.fetchall()
        
        results = []
        for row in rows:
            doc_keywords = json.loads(row[1])
            if keyword in doc_keywords:
                results.append({
                    'id': row[0],
                    'keywords': doc_keywords,
                    'label': row[2]
                })
        
        return results
    
    def search_by_similar_keywords(self, query_keywords: List[str], 
                                  threshold: float = 0.3) -> List[Dict]:
        """
        Поиск документов по похожим ключевым словам с использованием
        коэффициента Жаккара для измерения сходства
        
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
            
            # Рассчитываем коэффициент Жаккара
            if query_set and doc_set:
                intersection = len(query_set.intersection(doc_set))
                union = len(query_set.union(doc_set))
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 0
            
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
    
    def search_by_fuzzy_keywords(self, query_keywords: List[str],
                                min_matches: int = 1) -> List[Dict]:
        """
        Поиск документов, содержащих хотя бы N указанных ключевых слов
        
        Args:
            query_keywords: список ключевых слов для поиска
            min_matches: минимальное количество совпадений
            
        Returns:
            Список документов с количеством совпадений
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
            
            matches = len(query_set.intersection(doc_set))
            
            if matches >= min_matches:
                results.append({
                    'id': row[0],
                    'keywords': doc_keywords,
                    'label': row[2],
                    'matches': matches
                })
        
        # Сортируем по количеству совпадений
        results.sort(key=lambda x: x['matches'], reverse=True)
        return results
    
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
    
    def search_by_label(self, label: str, exact: bool = True) -> List[Dict]:
        """
        Поиск документов по названию
        
        Args:
            label: название для поиска
            exact: True - точное совпадение, False - частичное совпадение
            
        Returns:
            Список найденных документов
        """
        if exact:
            self.cursor.execute(
                'SELECT id, keywords, label FROM documents WHERE label = ?',
                (label,)
            )
        else:
            self.cursor.execute(
                'SELECT id, keywords, label FROM documents WHERE label LIKE ?',
                (f'%{label}%',)
            )
        
        rows = self.cursor.fetchall()
        
        return [{
            'id': row[0],
            'keywords': json.loads(row[1]),
            'label': row[2]
        } for row in rows]
    
    def get_keyword_statistics(self) -> Dict[str, int]:
        """
        Получение статистики по ключевым словам
        
        Returns:
            Словарь с частотой использования ключевых слов
        """
        self.cursor.execute('SELECT keywords FROM documents')
        rows = self.cursor.fetchall()
        
        all_keywords = []
        for row in rows:
            all_keywords.extend(json.loads(row[0]))
        
        return dict(Counter(all_keywords).most_common())
    
    def close(self):
        """Закрытие соединения с базой данных"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Пример использования
if __name__ == "__main__":
    # Создаем и заполняем базу данных примерами
    with DocumentDB() as db:
        # Добавляем несколько документов
        documents = [
            (["python", "программирование", "алгоритмы"], "Введение в Python"),
            (["базы данных", "sql", "postgresql"], "Основы SQL"),
            (["машинное обучение", "python", "нейронные сети"], "ML для начинающих"),
            (["алгоритмы", "структуры данных", "python"], "Алгоритмы и структуры данных"),
            (["веб-разработка", "javascript", "html", "css"], "Frontend разработка"),
        ]
        
        for keywords, label in documents:
            db.add_document(keywords, label)
        
        print("Все документы в базе:")
        for doc in db.get_all_documents():
            print(f"ID: {doc['id']}, Название: {doc['label']}, Ключевые слова: {doc['keywords']}")
        
        print("\n" + "="*50 + "\n")
        
        # Поиск по точному совпадению ключевого слова
        print("Поиск документов с ключевым словом 'python':")
        for doc in db.search_by_keyword("python"):
            print(f"  - {doc['label']}")
        
        print("\n" + "="*50 + "\n")
        
        # Поиск по похожим ключевым словам
        print("Поиск по похожим ключевым словам ['python', 'алгоритмы']:")
        similar_docs = db.search_by_similar_keywords(["python", "алгоритмы"], threshold=0.2)
        for doc in similar_docs:
            print(f"  - {doc['label']} (сходство: {doc['similarity']})")
        
        print("\n" + "="*50 + "\n")
        
        # Поиск документов с хотя бы одним совпадением
        print("Поиск документов с хотя бы одним из ['sql', 'базы данных']:")
        fuzzy_docs = db.search_by_fuzzy_keywords(["sql", "базы данных"], min_matches=1)
        for doc in fuzzy_docs:
            print(f"  - {doc['label']} (совпадений: {doc['matches']})")
        
        print("\n" + "="*50 + "\n")
        
        # Получение статистики по ключевым словам
        print("Статистика по ключевым словам:")
        stats = db.get_keyword_statistics()
        for keyword, count in list(stats.items())[:10]:  # Показываем топ-10
            print(f"  - {keyword}: {count}")