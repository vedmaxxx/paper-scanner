# Система поиска релевантной литературы по ключевым словам, извлекаемым из входного текста

 Решаемые задачи: Классификация текста, Поиск похожих документов

Используемая модель: deeppavlov topics_distilbert_base_uncased


Датасет: https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset?resource=download&select=train.csv



# Update
Перешли к задаче выделения ключевых слов из текста, т.к. классификация не подходит под нашу задачу
В связи с этим используется модель deeppavlov bert_embedder для извлечения эмбеддинга входного текста и кандидатов
