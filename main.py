# 
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Использовать первую GPU

# from deeppavlov import build_model, configs
# from deeppavlov.core.commands.utils import parse_config
# import numpy as np

# from train import train
# from utils import load_texts_from_txt

from deeppavlov import train_model
from deeppavlov.core.commands.utils import parse_config

config = parse_config('config.json')
config['dataset_reader']['data_path'] = './dataset'

model = train_model(config)

# НУЖНО ПОМЕНЯТЬ ПУТЬ К КОНФИГУ ДАТАСЕТА
# https://docs.deeppavlov.ai/en/master/features/models/classification.html#6.-Train-the-model-on-your-data
# config = configs.classifiers.topics_distilbert_base_uncased
# print(parse_config("topics_distilbert_base_uncased")["dataset_reader"]["data_path"])


# # model = build_model(config, download=False, install=False)

# train()

# Чтение текстов из TXT-файла(ов)
# Можно передать один файл:
# txt_file = "medicine-abstract.txt"
# texts = load_texts_from_txt(txt_file)

# Или список файлов:
# txt_files = [
#     "medicine-abstract.txt",
#     "science-and-technology.txt",
# ]
# texts = load_texts_from_txt(txt_files)


# predictions = model(texts)
# print(predictions)

# print("probabilites ", predictions[0])
# print("labels ", predictions[1])
