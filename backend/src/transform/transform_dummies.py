"""
Программа: Предобработка данных с бинаризованными колонками
Версия: 1.0
"""

import ast
import re
import pandas as pd
from transliterate import translit

from ..data.json_tools import open_json, save_json


def series_to_type_list(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Преобразование колонок в тип список
    :param df: датасет
    :param columns: колонки для преобразования
    :return: обновленный датасет
    """
    for col in columns:
        df[col] = df[col].fillna("[]")
        df[col] = df[col].apply(
            lambda x: x if isinstance(x, list) else ast.literal_eval(x)
        )

    return df


def make_dummies(df: pd.DataFrame, col: str, p: float = None) -> pd.DataFrame:
    """
    Создает бинаризованные колонки для указанного признака в dataframe,
    фильтрует колонки по заданному порогу встречаемости
    :param df: датасет
    :param col: колонка для бинаризации
    :param p: пороговое значение частоты встречаемости признака (от 0 до 1)
    :return: датасет с бинаризованными колонками
    """
    dummies = df[col].str.join("|").str.get_dummies(sep="|")

    if p is not None:
        # Расчет частоты каждой категории
        freq = dummies.sum() / len(dummies)

        # Фильтрация категорий, которые встречаются чаще, чем заданный порог
        filtered_categories = freq[freq > p].index
        dummies = dummies[filtered_categories]

    return dummies


def transform_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Пайплайн преобразования названий колонок
    :params df: датасет
    :return: обновленный датасет с новыми названиями колонок
    """

    # Перевод названий колонок в латинские буквы
    df.columns = [translit(x, language_code="ru", reversed=True) for x in df.columns]
    # Замена пробелов в названиях колонок на нижние подчеркивания
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Фильтрация названий колонок.
    # Остаются только нижние подчеркивания, англ. буквы и цифры
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    return df


def check_dummies_columns_evaluate(
    data: pd.DataFrame, train_sequence_path: str
) -> pd.DataFrame:
    """
    Добавление недостающих признаков и упорядочивание согласно train
    :param data: датасет test
    :param train_sequence_path: путь до списка с признаками train для сравнения
    :return: датасет test
    """

    train_sequence = open_json(train_sequence_path)
    data_sequence = data.columns

    # Поиск недостающих бинаризованных колонок
    missing_columns = set(train_sequence) - set(data_sequence)

    for col in missing_columns:
        data[col] = 0

    return data[train_sequence]


def preprocess_dummies(
    df: pd.DataFrame, flg_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Создание и проверка бинаризованных колонок
    :param df: датасет
    :param flg_evaluate: флаг evaluate
    :return: датасет с бинаризованными колонками
    """

    dummy_columns = kwargs["dummies"]["columns"]

    assert all(
        col in df.columns for col in dummy_columns
    ), "Не все требуемые колонки для бинаризации присутствуют в данных"

    # Преобразование колонок в тип список для бинаризации
    df = series_to_type_list(df, dummy_columns)

    # Формирование бинаризованных колонок
    all_dummies = pd.DataFrame()

    for col in dummy_columns:
        dummies = make_dummies(df, col, kwargs["variance_threshold"])

        if not flg_evaluate:
            # Сохранение списка оригинальных названий колонок для UI
            sorted_counts = dummies.sum().sort_values(ascending=False)
            dummies = dummies[sorted_counts.index]

            original_dummies_dict = {col: dummies.columns.tolist()}
            save_json(kwargs["dummies"][col], original_dummies_dict)

        all_dummies = pd.concat([all_dummies, dummies], axis=1)

    # Преобразование названий колонок для LightGBM
    all_dummies = transform_columns_names(all_dummies)

    if flg_evaluate:
        # Проверка на соответствие бинаризованных колонок
        all_dummies = check_dummies_columns_evaluate(
            all_dummies, kwargs["dummies"]["sequence_path"]
        )
    else:
        # Сохранение последовательности бинаризованных колонок для обучения
        save_json(kwargs["dummies"]["sequence_path"], all_dummies.columns.tolist())

    return all_dummies
