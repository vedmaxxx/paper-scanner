import torch
from deeppavlov import train_model
import gc
from deeppavlov.core.commands.utils import parse_config
from preprocessing import dataset_preprocessing


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU память очищена")


def train_with_gpu(config_path, data_path):
    """Обучает модель с автоматической настройкой GPU"""

    config = parse_config(config_path)
    config["dataset_reader"]["data_path"] = data_path

    if torch.cuda.is_available():
        device_id = 0

        # Автоматическая настройка batch_size
        total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        print(f"GPU память: {total_memory:.2f} GB")

        if total_memory >= 24:
            batch_size = 64
        elif total_memory >= 12:
            batch_size = 32
        elif total_memory >= 8:
            batch_size = 16
        elif total_memory >= 4:
            batch_size = 8
        else:
            batch_size = 4

        config["train"]["batch_size"] = batch_size
        config["chainer"]["pipe"][3]["device"] = f"cuda:{device_id}"

        print(f"Используем GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        print(f"Batch size: {batch_size}")
        cleanup_gpu()
        # Используем mixed precision для ускорения
        try:
            from torch.cuda.amp import autocast, GradScaler

            config["chainer"]["pipe"][3]["fp16"] = True
            print("Используем mixed precision (fp16)")
        except:
            print("Mixed precision не доступен, используем fp32")

    else:
        print("CUDA не доступна, используем CPU")
        config["train"]["batch_size"] = 8
        config["chainer"]["pipe"][3]["device"] = "cpu"

    # Обучаем
    return train_model(config)


# dataset_preprocessing()

# Использование
model = train_with_gpu("config.json", "./dataset/")
