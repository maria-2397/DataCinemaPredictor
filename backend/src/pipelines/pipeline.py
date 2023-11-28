"""
Программа: Сборный конвейер для тренировки модели
Версия: 1.0
"""

import os
import yaml
import pandas as pd
import joblib

from ..data.get_data import get_dataset
from ..data.split_dataset import split_train_test
from ..transform.transform_data import preprocess_target, preprocess_data
from ..transform.transform_dummies import preprocess_dummies
from ..train.train import find_optimal_params, train_model


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, предобработки и тренировки модели
    :param config_path: путь до файла с конфигурациями
    :return: None
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preproc_config = config["preprocessing"]
    train_config = config["train"]

    # get data
    data = get_dataset(
        dataset_path=preproc_config["original_data_path"],
        sep=preproc_config["data_separator"],
    )

    # preprocessing
    target = preprocess_target(data, **preproc_config)
    dummies = preprocess_dummies(data, flg_evaluate=False, **preproc_config)
    data = preprocess_data(data, flg_evaluate=False, **preproc_config)
    data = pd.concat([target, data, dummies], axis=1)

    # split data
    df_train, df_test = split_train_test(dataset=data, **preproc_config)

    # find optimal params
    study = find_optimal_params(data_train=df_train, data_test=df_test, **train_config)

    # train with optimal params
    reg = train_model(
        data_train=df_train,
        data_test=df_test,
        study=study,
        **train_config,
    )

    # save result (study, model)
    joblib.dump(reg, os.path.join(train_config["model_path"]))
    joblib.dump(study, os.path.join(train_config["study_path"]))
