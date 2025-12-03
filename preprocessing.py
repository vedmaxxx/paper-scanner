from data_formatter import format_dataset
from data_split import split_dataset
from data_filter import filter_dataset


def dataset_preprocessing():
    print("\n======= ПРЕДОБРАБОТКА ДАННЫХ ======\n")
    formatted_dataset = format_dataset()
    filtered_dataset = filter_dataset(formatted_dataset)
    split_dataset(filtered_dataset)
    print("\n======= ПРЕДОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА======\n")
