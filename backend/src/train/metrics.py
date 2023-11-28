"""
Программа: Получение метрик
Версия: 1.0
"""

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from ..data.json_tools import open_json, save_json


def r2_adjusted(
    data_y: np.ndarray, data_y_predict: np.ndarray, data_x: np.ndarray
) -> float:
    """
    Коэффициент детерминации (множественная регрессия)
    :param data_y: реальные данные
    :param data_y_predict: предсказанные значения
    :param data_x: матрица признаков
    :return: скорректированный коэффициент R^2
    """
    n_objects = len(data_y)
    n_features = data_x.shape[1]
    r2 = r2_score(data_y, data_y_predict)
    return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)


def wape(y_true: np.ndarray, y_predict: np.ndarray) -> float:
    """
    Weighted Absolute Percent Error
    :param y_true: реальные данные
    :param y_predict: предсказанные значения
    :return: взвешенная абсолютная процентная ошибка
    """
    return np.sum(np.abs(y_predict - y_true)) / np.sum(y_true) * 100


def create_dict_metrics(
    data_y: pd.Series,
    data_y_predict: pd.Series,
    data_x: pd.DataFrame,
    target_is_log: bool,
) -> dict:
    """
    Получение словаря с метриками для задачи регрессии и запись в словарь
    :param data_y: реальные данные
    :param data_y_predict: предсказанные значения
    :param data_x: матрица признаков для r2_adjusted
    :param target_is_log: логарифмирование целевой переменной, если необходимо
    :return: словарь с метриками
    """
    if target_is_log:
        data_y = np.expm1(data_y)
        data_y_predict = np.expm1(data_y_predict)

    dict_metrics = {
        "mae": round(mean_absolute_error(data_y, data_y_predict), 3),
        "r2_adjusted": round(r2_adjusted(data_y, data_y_predict, data_x), 3),
        "wape": round(wape(data_y, data_y_predict), 3),
    }
    return dict_metrics


def save_metrics(
    x_test: pd.DataFrame,
    x_train: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
    model: object,
    metric_path: str,
    target_is_log: bool,
) -> None:
    """
    Получение и сохранение метрик
    :param x_test: объект-признаки test
    :param y_test: целевая переменная test
    :param x_train: объект-признаки train
    :param y_train: целевая переменная train
    :param model: модель
    :param metric_path: путь для сохранения метрик
    :param target_is_log: условие логарифмированной целевой переменной
    """
    metrics = {
        "test": create_dict_metrics(
            data_y=y_test,
            data_y_predict=model.predict(x_test),
            data_x=x_test,
            target_is_log=target_is_log,
        ),
        "train": create_dict_metrics(
            data_y=y_train,
            data_y_predict=model.predict(x_train),
            data_x=x_train,
            target_is_log=target_is_log,
        ),
    }
    save_json(metric_path, metrics)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    metrics = open_json(config["train"]["metrics_path"])

    return metrics
