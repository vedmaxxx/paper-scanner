import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from typing import List
import threading
import sqlite3
import sys
import tempfile

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наши модули
from document_db import DocumentDB
from document_processor import DocumentProcessor
from file_reader import FileReader


class DocumentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система управления научными документами")
        self.root.geometry("900x750")

        # Инициализация БД и процессора документов
        self.db = DocumentDB()  # Главное соединение для GUI
        self.processor = DocumentProcessor()
        self.file_reader = FileReader()

        # Переменные
        self.selected_file_path = tk.StringVar()
        self.selected_file_path.set("Файл не выбран")

        # Временные файлы для конвертации
        self.temp_files = []

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настройка растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="📚 Система управления научными документами",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 12))

        # Разделитель
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 12)
        )

        # Панель выбора файла
        file_panel = ttk.LabelFrame(main_frame, text="Выбор документа", padding="8")
        file_panel.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        file_panel.columnconfigure(1, weight=1)

        ttk.Label(file_panel, text="Документ:", font=("Arial", 10)).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 8)
        )

        # Поле с путем к файлу
        file_frame = ttk.Frame(file_panel)
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)

        self.file_label = ttk.Label(
            file_frame,
            textvariable=self.selected_file_path,
            relief=tk.SUNKEN,
            padding=(10, 8),
            background="white",
            anchor=tk.W,
            font=("Arial", 9),
        )
        self.file_label.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 10)
        )

        browse_button = ttk.Button(
            file_frame, text="📂 Обзор...", command=self.browse_file, width=15
        )
        browse_button.grid(row=0, column=2, sticky=tk.E)

        # Информация о файле и его названии
        self.file_info_frame = ttk.Frame(file_panel)
        self.file_info_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0)
        )

        # Название файла (будет использоваться в БД)
        self.filename_label = ttk.Label(
            self.file_info_frame, text="", font=("Arial", 9, "bold"), foreground="green"
        )
        self.filename_label.grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5)
        )

        # Статистика файла
        self.file_info_labels = {}
        for i, (label, var) in enumerate(
            [
                ("Формат:", "format_var"),
                ("Размер:", "size_var"),
                ("Символов:", "chars_var"),
                ("Слов:", "words_var"),
            ]
        ):
            ttk.Label(self.file_info_frame, text=label, font=("Arial", 9)).grid(
                row=1, column=i * 2, sticky=tk.W, padx=(0, 5)
            )
            var_obj = tk.StringVar(value="---")
            self.file_info_labels[var] = ttk.Label(
                self.file_info_frame, textvariable=var_obj, font=("Arial", 9)
            )
            self.file_info_labels[var].grid(
                row=1, column=i * 2 + 1, sticky=tk.W, padx=(0, 20)
            )
            setattr(self, var, var_obj)

        # Панель действий
        action_panel = ttk.LabelFrame(
            main_frame, text="Действия с документом", padding="8"
        )
        action_panel.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        buttons_frame = ttk.Frame(action_panel)
        buttons_frame.grid(row=0, column=0, pady=(0, 8))

        self.add_button = ttk.Button(
            buttons_frame,
            text="📥 Добавить в базу данных",
            command=self.add_to_db,
            state=tk.DISABLED,
            width=25,
        )
        self.add_button.pack(side=tk.LEFT, padx=(0, 10))

        self.search_button = ttk.Button(
            buttons_frame,
            text="🔍 Найти релевантные статьи",
            command=self.search_relevant,
            state=tk.DISABLED,
            width=25,
        )
        self.search_button.pack(side=tk.LEFT)

        # Индикатор загрузки
        self.progress = ttk.Progressbar(action_panel, mode="indeterminate", length=350)
        self.progress.grid(row=1, column=0, pady=(8, 0))
        self.progress.grid_remove()

        # Статистика базы данных
        stats_panel = ttk.LabelFrame(
            main_frame, text="📊 Статистика базы данных", padding="5"
        )
        stats_panel.grid(
            row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.stats_label = ttk.Label(
            stats_panel, text="Документов в базе: 0", font=("Arial", 9)
        )
        self.stats_label.grid(row=0, column=0, sticky=tk.W)

        refresh_stats_button = ttk.Button(
            stats_panel,
            text="🔄 Обновить статистику",
            command=self.update_stats,
            width=25,
        )
        refresh_stats_button.grid(row=0, column=1, sticky=tk.E, padx=(20, 0))

        # Список документов
        docs_panel = ttk.LabelFrame(
            main_frame, text="📄 Документы в базе данных", padding="8"
        )
        docs_panel.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        docs_panel.columnconfigure(0, weight=1)
        docs_panel.rowconfigure(0, weight=1)

        # Treeview для отображения документов
        columns = ("ID", "Название документа", "Ключевые слова")
        self.doc_tree = ttk.Treeview(
            docs_panel, columns=columns, show="headings", height=10
        )

        # Определение заголовков
        self.doc_tree.heading("ID", text="ID")
        self.doc_tree.heading(
            "Название документа", text="Название документа (имя файла)"
        )
        self.doc_tree.heading("Ключевые слова", text="Ключевые слова")

        # Настройка колонок
        self.doc_tree.column("ID", width=50, anchor=tk.CENTER)
        self.doc_tree.column("Название документа", width=250)
        self.doc_tree.column("Ключевые слова", width=350)

        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(
            docs_panel, orient=tk.VERTICAL, command=self.doc_tree.yview
        )
        self.doc_tree.configure(yscrollcommand=scrollbar.set)

        self.doc_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Кнопки управления документами
        docs_buttons_frame = ttk.Frame(docs_panel)
        docs_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky=tk.E)

        refresh_button = ttk.Button(
            docs_buttons_frame,
            text="🔄 Обновить список",
            command=self.refresh_document_list,
            width=25,
        )
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        delete_button = ttk.Button(
            docs_buttons_frame,
            text="🗑️ Удалить выделенное",
            command=self.delete_selected_document,
            width=25,
        )
        delete_button.pack(side=tk.LEFT, padx=(0, 5))

        show_keywords_btn = ttk.Button(
            docs_buttons_frame,
            text="Показать ключевые слова",
            command=self.show_selected_keywords,
            width=25,
        )
        show_keywords_btn.pack(side=tk.LEFT)

        # Обновление статистики и списка документов при запуске
        self.update_stats()
        self.refresh_document_list()

    def browse_file(self):
        """Выбор файла через диалоговое окно"""
        file_path = filedialog.askopenfilename(
            title="Выберите документ",
            filetypes=[
                ("Все поддерживаемые форматы", "*.txt;*.pdf;*.doc;*.docx"),
                ("Текстовые файлы", "*.txt"),
                ("PDF файлы", "*.pdf"),
                ("Word файлы (DOC)", "*.doc"),
                ("Word файлы (DOCX)", "*.docx"),
                ("Все файлы", "*.*"),
            ],
        )

        if file_path:
            self.selected_file_path.set(file_path)
            self.update_file_info(file_path)
            self.add_button.config(state=tk.NORMAL)
            self.search_button.config(state=tk.NORMAL)

    def update_file_info(self, file_path):
        """Обновление информации о выбранном файле"""
        try:
            # Получаем информацию о файле
            file_info = self.processor.get_file_info(file_path)

            if "error" in file_info:
                self.show_file_error(file_info["error"])
                return

            # Отображаем имя файла, которое будет использоваться в БД
            filename = file_info.get(
                "filename_without_ext", os.path.splitext(os.path.basename(file_path))[0]
            )
            self.filename_label.config(text=f"📄 Название в БД: {filename}")

            # Обновляем статистику файла
            self.format_var.set(file_info["extension"].upper())
            self.size_var.set(f"{file_info['size_kb']:.1f} КБ")
            self.chars_var.set(f"{file_info['text_length']:,}")
            self.words_var.set(f"{file_info['words_count']:,}")

        except Exception as e:
            self.show_file_error(str(e))

    def show_file_error(self, error_message: str):
        """Отображение ошибки при чтении файла"""
        self.filename_label.config(text="")
        self.format_var.set("Ошибка")
        self.size_var.set("---")
        self.chars_var.set("---")
        self.words_var.set("---")

        messagebox.showerror(
            "Ошибка чтения файла",
            f"Не удалось прочитать файл.\n\n"
            f"Ошибка: {error_message}\n\n"
            f"Убедитесь, что:\n"
            f"1. Файл не поврежден\n"
            f"2. Установлены необходимые библиотеки\n"
            f"3. Файл имеет правильный формат",
        )

    def show_loading(self, show=True):
        """Показать/скрыть индикатор загрузки"""
        if show:
            self.progress.grid()
            self.progress.start()
            self.root.config(cursor="wait")
            self.add_button.config(state=tk.DISABLED)
            self.search_button.config(state=tk.DISABLED)
        else:
            self.progress.stop()
            self.progress.grid_remove()
            self.root.config(cursor="")
            if self.selected_file_path.get() != "Файл не выбран":
                self.add_button.config(state=tk.NORMAL)
                self.search_button.config(state=tk.NORMAL)

    def add_to_db(self):
        """Добавление документа в БД"""
        file_path = self.selected_file_path.get()

        if not file_path or file_path == "Файл не выбран":
            messagebox.showwarning("Предупреждение", "Сначала выберите файл")
            return

        # Получаем имя файла для подтверждения
        filename = os.path.splitext(os.path.basename(file_path))[0]

        # Подтверждение действия
        if not messagebox.askyesno(
            "Подтверждение", f"Добавить документ '{filename}' в базу данных?"
        ):
            return

        # Запуск в отдельном потоке
        thread = threading.Thread(target=self._add_to_db_thread, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _add_to_db_thread(self, file_path):
        """Поток для добавления документа в БД"""
        try:
            self.root.after(0, lambda: self.show_loading(True))

            # Создаем отдельное соединение с БД для этого потока
            thread_db = DocumentDB()  # Новое соединение в этом потоке

            # Обработка документа и добавление в БД
            result = self.processor.add_paper_to_system(thread_db, file_path)

            # Закрываем соединение в этом потоке
            thread_db.close()

            if result["success"]:
                # Обновление интерфейса в главном потоке
                self.root.after(0, self.refresh_document_list)
                self.root.after(0, self.update_stats)

                # Показать детали добавления
                details = (
                    f"📄 Название (имя файла): {result['label']}\n"
                    f"🆔 ID документа: {result['doc_id']}\n"
                    f"🔤 Извлечено ключевых слов: {result['keywords_count']}\n"
                    f"🔑 Пример ключевых слов: {', '.join(result['keywords'])}"
                )

                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "✅ Успешно",
                        f"Документ успешно добавлен в базу данных!\n\n{details}",
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "❌ Ошибка",
                        f"Не удалось добавить документ:\n{result['message']}",
                    ),
                )

        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "❌ Ошибка", f"Не удалось добавить документ:\n{str(e)}"
                ),
            )
        finally:
            self.root.after(0, lambda: self.show_loading(False))

    def search_relevant(self):
        """Поиск релевантных документов"""
        file_path = self.selected_file_path.get()

        if not file_path or file_path == "Файл не выбран":
            messagebox.showwarning("Предупреждение", "Сначала выберите файл")
            return

        # Запуск в отдельном потоке
        thread = threading.Thread(
            target=self._search_relevant_thread, args=(file_path,)
        )
        thread.daemon = True
        thread.start()

    def _search_relevant_thread(self, file_path):
        """Поток для поиска релевантных документов"""
        try:
            self.root.after(0, lambda: self.show_loading(True))

            # Создаем отдельное соединение с БД для этого потока
            thread_db = DocumentDB()  # Новое соединение в этом потоке

            # Поиск релевантных документов с порогом 60%
            relevant_papers = self.processor.get_relevant_papers(
                thread_db, file_path, similarity_threshold=0.6, max_results=10
            )

            # Закрываем соединение в этом потоке
            thread_db.close()

            if relevant_papers:
                # Форматирование результатов
                formatted_results = []
                for label, similarity in relevant_papers:
                    # Форматируем сходство в процентах
                    similarity_percent = similarity * 100
                    if similarity_percent >= 60:
                        formatted_results.append(
                            f"{label} (сходство: {similarity_percent:.1f}%)"
                        )

                # Открытие окна с результатами
                self.root.after(0, self.show_results_window, formatted_results)
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "🔍 Результаты поиска",
                        "К сожалению, релевантные документы не найдены.\n\n"
                        "Возможные причины:\n"
                        "• Сходство меньше 60%\n"
                        "• В базе данных недостаточно документов\n"
                        "• Ключевые слова документа уникальны\n\n"
                        "Попробуйте:\n"
                        "• Добавить больше документов в базу\n"
                        "• Использовать другой документ для поиска",
                    ),
                )

        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "❌ Ошибка", f"Не удалось выполнить поиск:\n{str(e)}"
                ),
            )
        finally:
            self.root.after(0, lambda: self.show_loading(False))

    def show_results_window(self, results: List[str]):
        """Отображение окна с результатами поиска"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Результаты поиска релевантных документов")
        results_window.geometry("700x500")
        results_window.transient(self.root)  # Сделать окно модальным

        # Центрирование окна
        results_window.update_idletasks()
        width = results_window.winfo_width()
        height = results_window.winfo_height()
        x = (results_window.winfo_screenwidth() // 2) - (width // 2)
        y = (results_window.winfo_screenheight() // 2) - (height // 2)
        results_window.geometry(f"{width}x{height}+{x}+{y}")

        # Заголовок
        ttk.Label(
            results_window,
            text="🎯 Найденные релевантные документы",
            font=("Arial", 14, "bold"),
        ).pack(pady=(15, 10))

        # Подзаголовок
        ttk.Label(
            results_window,
            text=f"Найдено документов: {len(results)} (сходство ≥60%)",
            font=("Arial", 10),
        ).pack(pady=(0, 15))

        # Текстовое поле с результатами
        text_frame = ttk.Frame(results_window)
        text_frame.pack(padx=15, pady=(0, 15), fill=tk.BOTH, expand=True)

        text_area = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, width=80, height=20, font=("Arial", 10)
        )
        text_area.pack(fill=tk.BOTH, expand=True)

        # Вставка результатов
        if results:
            text_area.insert(tk.END, "Список релевантных документов:\n\n")
            for i, result in enumerate(results, 1):
                text_area.insert(tk.END, f"{i}. {result}\n")
        else:
            text_area.insert(tk.END, "Релевантные документы не найдены.")

        text_area.config(state=tk.DISABLED)

        # Кнопки
        button_frame = ttk.Frame(results_window)
        button_frame.pack(pady=(0, 15))

        copy_button = ttk.Button(
            button_frame,
            text="📋 Копировать список",
            command=lambda: self.copy_to_clipboard(results, results_window),
            width=20,
        )
        copy_button.pack(side=tk.LEFT, padx=5)

        save_button = ttk.Button(
            button_frame,
            text="💾 Сохранить в файл",
            command=lambda: self.save_results_to_file(results, results_window),
            width=20,
        )
        save_button.pack(side=tk.LEFT, padx=5)

        back_button = ttk.Button(
            button_frame, text="↩️ Назад", command=results_window.destroy, width=15
        )
        back_button.pack(side=tk.LEFT, padx=5)

    def copy_to_clipboard(self, results: List[str], window: tk.Toplevel):
        """Копирование результатов в буфер обмена"""
        if results:
            text_to_copy = "Релевантные документы:\n\n"
            text_to_copy += "\n".join(
                [f"{i+1}. {result}" for i, result in enumerate(results)]
            )
            self.root.clipboard_clear()
            self.root.clipboard_append(text_to_copy)
            messagebox.showinfo(
                "✅ Успешно", "Список скопирован в буфер обмена", parent=window
            )
        else:
            messagebox.showwarning(
                "⚠️ Внимание", "Нет результатов для копирования", parent=window
            )

    def save_results_to_file(self, results: List[str], window: tk.Toplevel):
        """Сохранение результатов в файл"""
        if not results:
            messagebox.showwarning(
                "⚠️ Внимание", "Нет результатов для сохранения", parent=window
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить результаты поиска",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("РЕЗУЛЬТАТЫ ПОИСКА РЕЛЕВАНТНЫХ ДОКУМЕНТОВ\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Найдено документов: {len(results)}\n\n")

                    for i, result in enumerate(results, 1):
                        f.write(f"{i}. {result}\n")

                messagebox.showinfo(
                    "✅ Успешно",
                    f"Результаты сохранены в файл:\n{file_path}",
                    parent=window,
                )
            except Exception as e:
                messagebox.showerror(
                    "❌ Ошибка", f"Не удалось сохранить файл:\n{str(e)}", parent=window
                )

    def refresh_document_list(self):
        """Обновление списка документов в Treeview"""
        # Очистка текущего списка
        for item in self.doc_tree.get_children():
            self.doc_tree.delete(item)

        try:
            # Получение всех документов из БД (используем главное соединение)
            all_docs = self.db.get_all_documents()

            # Добавление в Treeview (последние 15 документов)
            for doc in all_docs[-15:]:
                keywords_str = ", ".join(
                    doc["keywords"][:3]
                )  # Показываем первые 3 ключевых слова
                if len(doc["keywords"]) > 3:
                    keywords_str += "..."

                self.doc_tree.insert(
                    "", tk.END, values=(doc["id"], doc["label"], keywords_str)
                )

        except Exception as e:
            print(f"Ошибка обновления списка документов: {e}")

    def update_stats(self):
        """Обновление статистики базы данных"""
        try:
            all_docs = self.db.get_all_documents()
            doc_count = len(all_docs)

            # Подсчет общего количества ключевых слов
            total_keywords = 0
            unique_keywords = set()
            for doc in all_docs:
                total_keywords += len(doc["keywords"])
                unique_keywords.update(doc["keywords"])

            self.stats_label.config(
                text=f"📊 Документов: {doc_count} | "
                f"Ключевых слов: {total_keywords} | "
                f"Уникальных: {len(unique_keywords)}"
            )

        except Exception as e:
            print(f"Ошибка обновления статистики: {e}")

    def delete_selected_document(self):
        """Удаление выделенного документа"""
        selected_item = self.doc_tree.selection()
        if not selected_item:
            messagebox.showwarning("⚠️ Внимание", "Выберите документ для удаления")
            return

        # Получаем ID документа
        item_values = self.doc_tree.item(selected_item[0], "values")
        if not item_values:
            return

        doc_id = item_values[0]
        doc_name = item_values[1]

        # Подтверждение удаления
        if messagebox.askyesno(
            "⚠️ Подтверждение удаления",
            f"Вы уверены, что хотите удалить документ?\n\n"
            f"ID: {doc_id}\n"
            f"Название: {doc_name}",
            icon=messagebox.WARNING,
        ):
            try:
                # Удаление документа из БД (используем главное соединение)
                if self.db.delete_document(doc_id):
                    self.refresh_document_list()
                    self.update_stats()
                    messagebox.showinfo(
                        "✅ Успешно", f"Документ '{doc_name}' удален из базы данных"
                    )
                else:
                    messagebox.showerror("❌ Ошибка", "Не удалось удалить документ")

            except Exception as e:
                messagebox.showerror("❌ Ошибка", f"Ошибка при удалении: {str(e)}")

    def show_selected_keywords(self):
        """Отображение ключевых слов выделенного документа"""
        selected_item = self.doc_tree.selection()
        if not selected_item:
            messagebox.showwarning(
                "⚠️ Внимание", "Выберите документ для просмотра ключевых слов"
            )
            return

        # Получаем ID документа

        item_values = self.doc_tree.item(selected_item[0], "values")

        doc_id = item_values[0]
        doc_name = item_values[1]

        try:
            # Получаем все документы из БД
            all_docs = self.db.get_all_documents()

            # Находим нужный документ по ID
            selected_doc = None
            for doc in all_docs:
                if str(doc["id"]) == str(doc_id):
                    selected_doc = doc
                    break

            if not selected_doc:
                messagebox.showerror(
                    "Ошибка", f"Документ с ID {doc_id} не найден в базе данных"
                )
                return

            # Создаем новое окно
            keywords_window = tk.Toplevel(self.root)
            keywords_window.title(f"Ключевые слова документа")
            keywords_window.geometry("600x500")
            keywords_window.transient(self.root)  # Сделать окно модальным
            keywords_window.grab_set()  # Блокировать взаимодействие с основным окном

            # Центрирование окна
            keywords_window.update_idletasks()
            width = keywords_window.winfo_width()
            height = keywords_window.winfo_height()
            x = (keywords_window.winfo_screenwidth() // 2) - (width // 2)
            y = (keywords_window.winfo_screenheight() // 2) - (height // 2)
            keywords_window.geometry(f"{width}x{height}+{x}+{y}")

            # Заголовок
            title_label = ttk.Label(
                keywords_window,
                text="🔑 Ключевые слова документа",
                font=("Arial", 14, "bold"),
            )
            title_label.pack(pady=(15, 10))

            # Информация о документе
            doc_info_frame = ttk.Frame(keywords_window)
            doc_info_frame.pack(pady=(0, 15), padx=20, fill=tk.X)

            ttk.Label(
                doc_info_frame,
                text=f"📄 Документ: {selected_doc['label']}",
                font=("Arial", 11, "bold"),
                foreground="green",
            ).pack(anchor=tk.W, pady=(0, 5))

            ttk.Label(
                doc_info_frame, text=f"🆔 ID: {selected_doc['id']}", font=("Arial", 10)
            ).pack(anchor=tk.W, pady=(0, 5))

            ttk.Label(
                doc_info_frame,
                text=f"🔢 Всего ключевых слов: {len(selected_doc['keywords'])}",
                font=("Arial", 10),
            ).pack(anchor=tk.W)

            # Разделитель
            ttk.Separator(keywords_window, orient=tk.HORIZONTAL).pack(
                fill=tk.X, padx=20, pady=(0, 10)
            )

            # Прокручиваемый список ключевых слов
            list_frame = ttk.Frame(keywords_window)
            list_frame.pack(padx=20, pady=(0, 15), fill=tk.BOTH, expand=True)

            # Создаем Listbox с полосой прокрутки
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            keywords_listbox = tk.Listbox(
                list_frame,
                font=("Arial", 10),
                yscrollcommand=scrollbar.set,
                selectmode=tk.EXTENDED,
                height=15,
            )
            keywords_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar.config(command=keywords_listbox.yview)

            # Добавляем ключевые слова в список
            for i, keyword in enumerate(selected_doc["keywords"], 1):
                keywords_listbox.insert(tk.END, f"{i}. {keyword}")

            # Статистика по ключевым словам
            stats_frame = ttk.Frame(keywords_window)
            stats_frame.pack(pady=(0, 15), padx=20, fill=tk.X)

            # Подсчитываем длину ключевых слов
            total_chars = sum(len(kw) for kw in selected_doc["keywords"])
            avg_length = (
                total_chars / len(selected_doc["keywords"])
                if selected_doc["keywords"]
                else 0
            )

            ttk.Label(
                stats_frame,
                text=f"📊 Статистика: Средняя длина слова: {avg_length:.1f} символов | Общая длина: {total_chars} символов",
                font=("Arial", 9),
                foreground="blue",
            ).pack(anchor=tk.W)

            # Кнопки
            buttons_frame = ttk.Frame(keywords_window)
            buttons_frame.pack(pady=(0, 15))

            # Кнопка копирования всех ключевых слов
            copy_button = ttk.Button(
                buttons_frame,
                text="📋 Копировать все ключевые слова",
                command=lambda: self.copy_keywords_to_clipboard(
                    selected_doc["keywords"], keywords_window
                ),
                width=25,
            )
            copy_button.pack(side=tk.LEFT, padx=5)

            # Кнопка копирования выделенных ключевых слов
            copy_selected_button = ttk.Button(
                buttons_frame,
                text="📋 Копировать выделенные",
                command=lambda: self.copy_selected_keywords_to_clipboard(
                    keywords_listbox, keywords_window
                ),
                width=25,
            )
            copy_selected_button.pack(side=tk.LEFT, padx=5)

            # Кнопка закрытия
            close_button = ttk.Button(
                buttons_frame, text="Закрыть", command=keywords_window.destroy, width=15
            )
            close_button.pack(side=tk.LEFT, padx=5)

        except Exception as e:
            messagebox.showerror(
                "❌ Ошибка", f"Не удалось загрузить ключевые слова:\n{str(e)}"
            )

    def copy_keywords_to_clipboard(self, keywords, window):
        """Копирование всех ключевых слов в буфер обмена"""
        if keywords:
            text_to_copy = "Ключевые слова документа:\n\n"
            text_to_copy += "\n".join([f"{i+1}. {kw}" for i, kw in enumerate(keywords)])
            self.root.clipboard_clear()
            self.root.clipboard_append(text_to_copy)
            messagebox.showinfo(
                "✅ Успешно",
                f"Все ключевые слова ({len(keywords)} шт.) скопированы в буфер обмена",
                parent=window,
            )
        else:
            messagebox.showwarning(
                "⚠️ Внимание",
                "Нет ключевых слов для копирования",
                parent=window,
            )

    def copy_selected_keywords_to_clipboard(self, listbox, window):
        """Копирование выделенных ключевых слов в буфер обмена"""
        selected_indices = listbox.curselection()
        if selected_indices:
            selected_keywords = []
            for index in selected_indices:
                item_text = listbox.get(index)
                # Убираем номер из текста (например, "1. ключевое слово" → "ключевое слово")
                keyword = (
                    item_text.split(". ", 1)[1] if ". " in item_text else item_text
                )
                selected_keywords.append(keyword)

            text_to_copy = "Выбранные ключевые слова:\n\n"
            text_to_copy += "\n".join(
                [f"{i+1}. {kw}" for i, kw in enumerate(selected_keywords)]
            )
            self.root.clipboard_clear()
            self.root.clipboard_append(text_to_copy)
            messagebox.showinfo(
                "✅ Успешно",
                f"Выделенные ключевые слова ({len(selected_keywords)} шт.) скопированы в буфер обмена",
                parent=window,
            )
        else:
            messagebox.showwarning(
                "⚠️ Внимание",
                "Выберите ключевые слова для копирования (удерживайте Ctrl для выбора нескольких)",
                parent=window,
            )

    def cleanup_temp_files(self):
        """Очистка временных файлов"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files = []

    def on_closing(self):
        """Обработка закрытия приложения"""
        try:
            # Очищаем временные файлы
            self.cleanup_temp_files()

            # Закрываем главное соединение с БД
            if hasattr(self, "db"):
                self.db.close()
        except:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DocumentApp(root)

    # Обработка закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Центрирование окна
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
