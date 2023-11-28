"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

from typing import Tuple, Dict
from io import BytesIO
import io
import streamlit as st
import pandas as pd


def get_dataset(dataset_path: str, sep: str) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param sep: разделитель в файле
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_csv(dataset_path, sep=sep)


def load_data(
    data_path: str,
    sep: str,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data_path: путь к данным
    :param sep: разделитель в файле
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_csv(data_path, sep=sep)
    st.write("Dataset load")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # Запись в BytesIO буфер
    dataset.to_csv(dataset_bytes_obj, sep=sep, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    # Нужно поместить все в такой словарик, чтобы наш backend сервис все считал
    files = {"file": ("uploaded_dataset.csv", dataset_bytes_obj, "multipart/form-data")}
    return dataset, files
