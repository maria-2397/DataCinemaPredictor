"""
Программа: Предобработка данных без бинаризованных колонок
Версия: 1.0
"""

import pandas as pd
import numpy as np
from ..data.json_tools import open_json, save_json


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание их согласно train
    :param data: датасет test
    :param unique_values_path: путь до списка с признаками train для сравнения
    :return: датасет test
    """

    unique_values = open_json(unique_values_path)
    column_sequence = unique_values.keys()

    assert set(column_sequence) == set(data.columns), "Разные признаки"
    return data[column_sequence]


def save_unique_train_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param data: датасет
    :param drop_columns: список с признаками для удаления
    :param target_column: целевая переменная
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    data = data.drop(columns=drop_columns + [target_column], axis=1, errors="ignore")

    # создаем словарь с уникальными значениями для вывода в UI
    dict_unique = {key: data[key].dropna().unique().tolist() for key in data.columns}

    save_json(unique_values_path, dict_unique)


def preprocess_target(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Преобразование целевой переменной
    :params df: датасет с колонкой целевой переменной
    """
    if kwargs["log_target"]:
        df[kwargs["target_column"]] = np.log1p(df[kwargs["target_column"]])

    return df[kwargs["target_column"]]


def preprocess_data(
    df: pd.DataFrame, flg_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Пайплайн по предобработке данных без бинаризации
    :param df: датасет
    :param flg_evaluate: флаг для evaluate
    :return: датасет
    """
    df = df.drop(
        kwargs["drop_columns"] + [kwargs["target_column"]], axis=1, errors="ignore"
    )
    # проверка dataset на совпадение с признаками из train
    # либо сохранение уникальных данных с признаками из train
    if flg_evaluate:
        df = check_columns_evaluate(
            data=df, unique_values_path=kwargs["unique_values_path"]
        )
    else:
        save_unique_train_data(
            data=df,
            drop_columns=kwargs["drop_columns"],
            target_column=kwargs["target_column"],
            unique_values_path=kwargs["unique_values_path"],
        )

    df["movie_age"] = kwargs["current_year"] - df[kwargs["year_column"]]
    df = df.drop(kwargs["year_column"], axis=1)

    # Преобразование типа колонок
    df = transform_types(data=df, change_type_columns=kwargs["change_type_columns"])

    return df
