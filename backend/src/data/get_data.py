"""
Программа: Получение данных из файла
Версия: 1.0
"""

import pandas as pd


def get_dataset(dataset_path: str, sep: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param sep: разделитель в файле
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path, sep=sep)
