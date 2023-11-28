"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import os
import yaml
import pandas as pd
import numpy as np
import joblib

from ..transform.transform_data import preprocess_data
from ..transform.transform_dummies import preprocess_dummies
from ..data.get_data import get_dataset


def pipeline_evaluate(
    config_path: str, dataset: pd.DataFrame = None, data_path: str = None
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(
            dataset_path=data_path, sep=preprocessing_config["data_separator"]
        )

    dummies = preprocess_dummies(dataset, **preprocessing_config)
    data = preprocess_data(dataset, **preprocessing_config)

    data = pd.concat([data, dummies], axis=1)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(data)

    if preprocessing_config["log_target"]:
        prediction = np.expm1(prediction)

    return prediction.tolist()
