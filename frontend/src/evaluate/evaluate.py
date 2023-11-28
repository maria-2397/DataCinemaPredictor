"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
import random
from io import BytesIO
from typing import Union, List

import pandas as pd
import requests
import streamlit as st

from ..data.json_tools import open_json


def generate_quartiles(n_parts: int) -> list:
    """
    Генерирует квартили для равномерного разделения на n_parts частей
    :param n_parts: количество частей для разбиения
    :return: список квартилей
    """
    step = 1.0 / n_parts
    return [step * i for i in range(1, n_parts)]


def transform_gradings(
    grade_index: int, grade_values: List[str], data_values: List[Union[int, float]]
) -> Union[int, float]:
    """
    Преобразование градации в числовое значение с элементом случайности
    :param grade_index: индекс элемента в списке с названиями градации
    :param grade_values: список с названиями градаций
    :param data_values: числовые данные для разбиения на квартили
    :return: случайное значение (больше среднего) из заданного интервала
    """
    series = pd.Series(data_values)
    # Генерация квартилей, с учетом количества градаций
    quartiles = generate_quartiles(len(grade_values))
    bins = series.quantile([0] + quartiles + [1]).tolist()

    # Определение границ интервала
    lower_bound = bins[grade_index]
    upper_bound = bins[grade_index + 1]

    # Вычисление середины интервала
    middle = (lower_bound + upper_bound) / 2

    # Фильтрация списка для получения значений в заданном интервале
    filtered_values = [x for x in data_values if middle <= x <= upper_bound]

    # Выбор случайного значения из отфильтрованного списка
    if filtered_values:
        return random.choice(filtered_values)

    return series.median()


def evaluate_input(
    unique_data_path: str, dummies_info: dict, gradings: dict, endpoint: object
) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    :param dummies_info: словарь с информацией о бинаризованных данных
    :param gradings: словарь с информацией о градациях признаков
    :return: None
    """
    with open(unique_data_path) as file:
        uniq_df = json.load(file)

    col1, col2 = st.columns(2)
    with col1:
        year = st.slider(
            "Год выпуска фильма",
            min_value=min(uniq_df["year"]),
            max_value=max(uniq_df["year"]),
        )
        movie_type = st.selectbox("Тип картины (фильм/мультфильм)", (uniq_df["type"]))
        votes_await = st.selectbox(
            "Рейтинг ожидания фильма на Кинопоиске", gradings["await"]
        )
    with col2:
        movie_length = st.slider(
            "Длительность фильма (в минутах)",
            min_value=int(min(uniq_df["movieLength"])),
            max_value=int(max(uniq_df["movieLength"])),
            step=1,
        )
        age_rating = st.selectbox(
            "Возрастной рейтинг фильма", sorted([int(x) for x in uniq_df["ageRating"]])
        )
        videos_trailers_number = st.slider(
            "Количество трейлеров фильма",
            min_value=int(min(uniq_df["videos_trailers_number"])),
            max_value=int(max(uniq_df["videos_trailers_number"])),
        )
    votes, ratings = st.columns(2)
    with votes:
        votes_kp = st.selectbox("Популярность фильма на Кинопоиске", gradings["votes"])
        votes_imdb = st.selectbox("Популярность фильма на IMDb", gradings["votes"])
        votes_film_critics = st.slider(
            "Количество оценок кинокритиков в мире",
            min_value=int(min(uniq_df["votes_filmCritics"])),
            max_value=int(max(uniq_df["votes_filmCritics"])),
            step=1,
        )
    with ratings:
        rating_kp = st.slider(
            "Рейтинг фильма на Кинопоиске",
            min_value=min(uniq_df["rating_kp"]),
            max_value=max(uniq_df["rating_kp"]),
        )
        rating_imdb = st.slider(
            "Рейтинг фильма на IMDb",
            min_value=min(uniq_df["rating_imdb"]),
            max_value=max(uniq_df["rating_imdb"]),
            step=0.1,
        )
        rating_film_critics = st.slider(
            "Рейтинг кинокритиков мире",
            min_value=min(uniq_df["rating_filmCritics"]),
            max_value=max(uniq_df["rating_filmCritics"]),
        )
    budget_values = [int(x) for x in uniq_df["budget"]]
    budget = st.slider(
        "Бюджет фильма (в долларах)",
        min_value=min(budget_values),
        max_value=max(budget_values),
        step=1,
    )
    # Метрики персон
    actors, directors, writers = st.columns(3)
    with actors:
        actor_metric = st.selectbox("Актерский состав", gradings["persons"])
    with directors:
        director_metric = st.selectbox("Режиссеры", gradings["persons"])
    with writers:
        writer_metric = st.selectbox("Сценаристы", gradings["persons"])
    # Жанры и страны
    genres_col, countries_col = st.columns(2)
    with genres_col:
        genres_values = open_json(dummies_info["genres"])["genres"]
        genres = st.multiselect("Жанры фильма", genres_values)
    with countries_col:
        countries_values = open_json(dummies_info["countries"])["countries"]
        countries = st.multiselect("Страны фильма", countries_values)
    # Студии
    production_col, special_effects_col = st.columns(2)
    with production_col:
        production_values = open_json(dummies_info["Production"])["Production"]
        production = st.multiselect("Студии произдодства", production_values)
    with special_effects_col:
        special_effects_values = open_json(dummies_info["Special_effects"])[
            "Special_effects"
        ]
        special_effects = st.multiselect("Студии спецэффектов", special_effects_values)
    # Преобразование градаций кол-ва оценок в числовые значения
    gradings_data = {
        "votes_await": gradings["await"].index(votes_await),
        "votes_kp": gradings["votes"].index(votes_kp),
        "votes_imdb": gradings["votes"].index(votes_imdb),
        "actor_metric": gradings["persons"].index(actor_metric),
        "director_metric": gradings["persons"].index(director_metric),
        "writer_metric": gradings["persons"].index(writer_metric),
    }
    for key, index in gradings_data.items():
        gradings_data[key] = transform_gradings(
            grade_index=index, data_values=uniq_df[key], grade_values=gradings["votes"]
        )
    # Сделаем количество оценок целым числом
    for key in gradings_data:
        if "votes" in key:
            gradings_data[key] = int(gradings_data[key])

    data = {
        "year": year,
        "movieLength": movie_length,
        "ageRating": age_rating,
        "type": movie_type,
        "genres": genres,
        "countries": countries,
        "Production": production,
        "Special_effects": special_effects,
        "budget": budget,
        "videos_trailers_number": videos_trailers_number,
        "votes_filmCritics": votes_film_critics,
        "rating_kp": rating_kp,
        "rating_imdb": rating_imdb,
        "rating_filmCritics": rating_film_critics,
    }
    # Заполнение возрастного рейтинга MPAA
    data["ratingMpaa"] = gradings["age_rating_corr"][data["ageRating"]]
    data.update(gradings_data)

    # markdown_text = "### Введенные данные:\n"
    # for key, value in data.items():
    #     markdown_text += f"- **{key}**: {value}\n"
    # st.markdown(markdown_text)

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"### {output[0]}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO) -> None:
    """
    Получение входных данных в качестве файла и вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files: объект BytesIO, содержащий данные для отправки на сервер
    :return: None
    """
    button_ok = st.button("Predict")
    if button_ok:
        data_ = data[:5]
        response = requests.post(endpoint, files=files, timeout=8000)

        if "movie_name" in data_.columns:
            data_ = data_.loc[:, ["movie_name"]].copy()

        data_["predicted_fees_usa"] = response.json()["prediction"]
        data_["predicted_fees_usa"] = data_["predicted_fees_usa"].astype(int)

        st.write(data_.head())
