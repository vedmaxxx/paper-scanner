import torch
import os


model_path = "~/.deeppavlov/models/classifiers/multilabel_topics_v1/model.pth.tar"
model_path = os.path.expanduser(model_path)

if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("Файл модели валиден!")
        print("Ключи в чекпоинте:", list(checkpoint.keys()))
    except Exception as e:
        print(f"Файл поврежден: {e}")
else:
    print("Файл модели не существует")
