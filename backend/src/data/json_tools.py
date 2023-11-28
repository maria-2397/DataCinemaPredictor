"""
Программа: Чтение и запись json файлов
"""

import json
from typing import Any


def open_json(name: str) -> Any:
    """
    Открытие и чтение содержимого JSON файла
    :param name: путь к JSON файлу для чтения
    :return: содержимое JSON файла
    """
    with open(name, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(name: str, json_data: Any) -> None:
    """
    Сохранение данных в формате JSON в файл
    :param name: путь к файлу для сохранения данных
    :param json_data: данные для сохранения в формате JSON
    """
    with open(name, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4, ensure_ascii=False)
