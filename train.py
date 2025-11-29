from deeppavlov import configs
from deeppavlov.core.commands.utils import parse_config


model_config_name = "topics_distilbert_base_uncased"


config = parse_config(model_config_name)["dataset_reader"]["data_path"]
config["dataset_reader"]["data_path"] = "./dataset/"


def train():
    model = train_model(config)
    predictions = model(["You are kind of stupid", "You are a wonderful person!"])
    print("probabilites ", predictions[0])
    print("labels ", predictions[1])
