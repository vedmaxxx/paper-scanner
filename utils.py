# Функция для чтения текста из TXT-файла(ов)
def load_texts_from_txt(txt_file_path):
    """
    Читает весь текст из TXT-файла(ов) как один единый текст.

    Args:
        txt_file_path: строка с путем к файлу или список путей к файлам

    Returns:
        список текстов - каждый файл читается как один текст
    """
    # Если передан список файлов
    if isinstance(txt_file_path, list):
        all_texts = []
        for file_path in txt_file_path:
            texts = _read_single_file(file_path)
            all_texts.extend(texts)
        return all_texts

    # Если передан один файл (строка)
    return _read_single_file(txt_file_path)


def _read_single_file(file_path):
    """
    Вспомогательная функция для чтения одного файла.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Убираем лишние пробелы в начале и конце
        text = text.strip()

        if not text:
            print(f"Файл {file_path} пуст. Пропускается.")
            return []

        # Возвращаем как список с одним элементом
        return [text]

    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Пропускается.")
        return []
    except Exception as e:
        print(f"Ошибка при загрузке текста из {file_path}: {e}")
        return []
