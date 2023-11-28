"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os

import yaml
import streamlit as st

from src.data.get_data import get_dataset
from src.plotting.charts import (
    plot_count_year_fees,
    plot_distributions,
    plot_category_distribution,
    plot_fees_dummies,
)

from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input
from src.data.get_data import load_data
from src.evaluate.evaluate import evaluate_from_file

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    movie_logo_path = config["movie_logo_path"]

    st.image(movie_logo_path, width=600)

    st.markdown("# Описание проекта")
    st.title("MLOps: Прогнозирование кассовых сборов фильмов в США 🎬 💰")
    st.write(
        """
        Проект разрабатывает модель машинного обучения для 
        предсказания кассовых сборов фильмов в США, 
        основываясь на данных, собранных с сайта Kinopoisk.
        Включает страницы с визуализациями, интерактивный модуль для обучения 
         модели и функциональность для предсказания сборов."""
    )

    markdown_text = """
        ### Описание полей
        - **movie_id**: уникальный идентификатор фильма
        - **movie_name**: название фильма
        - **year**: год выпуска фильма
        - **movieLength**: длительность фильма
        - **ageRating, ratingMpaa**: возрастные рейтинги фильма
        - **type**: тип картины (фильм/мультфильм)
        - **genres**: жанры фильма
        - **countries**: страны фильма
        - **Production**: студии производства
        - **Special_effects**: студии спецэффектов
        - **budget**: бюджет фильма
        - **videos_trailers_number**: количество трейлеров
        - **rating_kp, rating_imdb**: рейтинги фильма на Кинопоиске и IMDb
        - **rating_filmCritics**: рейтинг кинокритиков в мире
        - **votes_kp, votes_imdb**: количество оценок фильма на Кинопоиске и IMDb
        - **votes_filmCritics**: количество оценок кинокритиков в мире
        - **votes_await**: количество ожидающих фильм на Кинопоиске
        - **actor_metric**: метрика актеров
        - **director_metric**: метрика режиссеров
        - **writer_metric**: метрика сценаристов
        - **fees_usa**: сборы фильма в США
    """
    st.markdown(markdown_text)


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]

    data = get_dataset(
        dataset_path=preprocessing_config["original_data_path"],
        sep=preprocessing_config["data_separator"],
    )

    st.write(data.head())

    year_fees_movie_count = st.expander("Год релиза - сборы и количество фильмов")
    age_rating_fees = st.expander("Возрастной рейтинг фильма и сборы")
    movie_type_fees = st.expander("Тип картины и сборы")
    fees_genres = st.expander("Жанр фильма и сборы")

    with year_fees_movie_count:
        st.markdown(
            "**Гипотеза:** С течением лет, сборы фильмов в США увеличиваются, "
            "также увеличивается количество выпущенных фильмов."
        )
        st.pyplot(
            plot_count_year_fees(
                df=data,
                target_col=preprocessing_config["target_column"],
                year_col=preprocessing_config["year_column"],
            )
        )
        st.markdown("#### Выводы:")
        st.markdown("**Количество выпущенных фильмов:**")
        st.write(
            """
        В период с 2000 по 2008 наблюдается общая тенденция к увеличению
        количества фильмов. Этот тренд может отражать общий рост и динамичность
        кинопромышленности в начале 21-го века. После 2008 года количество фильмов
        в каждом году колеблется, не показывая четкой тенденции к росту или уменьшению. 
        """
        )
        st.markdown("**Медианные сборы:**")
        st.write(
            """
         С 2000 по 2010 год наблюдается тенденция к уменьшению медианных сборов
          фильмов на фоне роста их общего количества. Причиной этому может быть
          большее число фильмов с низкими сборами, что снижает медиану.
        """
        )
        st.markdown("**Максимальные сборы:**")
        st.write(
            """
        Максимальные сборы фильмов в США имеют тенденцию к увеличению со временем.
         Это может быть обусловлено инфляцией, ростом кинопроизводства и маркетинга, 
         технологическими инновациями, а также изменениями потребительских предпочтений.
        """
        )

    with age_rating_fees:
        st.markdown(
            """
        **Гипотеза:** Фильмы, предназначенные для более широкой аудитории 
        (например, 6+, 12+ и 16+), имеют тенденцию привлекать больше зрителей 
        и иметь более высокие сборы по сравнению с фильмами для взрослой аудитории (18+).
        """
        )
        st.pyplot(
            plot_distributions(
                df=data,
                category_col="ageRating",
                value_col=preprocessing_config["target_column"],
                title_kde="Распределение сборов в США по категориям возрастного рейтинга",
                title_boxplot="Boxplot сборов в США в по категориям возрастного рейтинга",
            )
        )
        st.markdown("#### Выводы:")
        st.markdown("**Максимальные сборы:**")
        st.write(
            """
        Фильмы с рейтингом 6+ и 12+ имеют высшие максимальные сборы. 
        Фильмы с такими рейтингами имеют потенциал стать блокбастерами.
        """
        )
        st.markdown("**Медианные значения:**")
        st.write(
            """
        Медианные сборы для фильмов с рейтингами 6+, 12+ и 16+ 
        примерно сопоставимы и значительно выше, чем для 18+.
        """
        )
        st.markdown("**Выбросы:**")
        st.write(
            """
        Фильмы с рейтингами 12+, 16+ и 6+ имеют значительное 
        количество высоких выбросов в сборах. 
        Значит некоторые фильмы в этих категориях имели огромный успех.
        """
        )
        st.markdown("**Общие наблюдения:**")
        st.write(
            """
        Фильмы с рейтингом 18+ склонны иметь менее стабильные сборы и 
        меньший потенциал для высоких доходов по сравнению с фильмами 
        других рейтингов, хотя исключения, конечно, существуют.
        """
        )

    with movie_type_fees:
        st.markdown(
            """
        **Гипотеза:** Мультфильмы в среднем собирают больше денег в США, чем кинофильмы.
        """
        )
        st.pyplot(
            plot_category_distribution(
                df=data,
                category_col="type",
                title="Распределение фильмов по типу картины",
            )
        )
        st.pyplot(
            plot_distributions(
                df=data,
                category_col="type",
                value_col=preprocessing_config["target_column"],
                title_kde="Распределение сборов в США в разрезе типа картины",
                title_boxplot="Boxplot сборов в США в разрезе типа картины",
            )
        )
        st.markdown("#### Выводы:")
        st.markdown(
            """
        - Большая часть данных относится к фильмам (не мультфильмам)
        - Медианный сбор мультфильмов составляет примерно 60 млн долларов,
         что значительно выше, чем для кинофильмов (9 млн долларов). 
         Также есть мультфильмы с выдающимися сборами, превышающими 400 млн долларов.
        - Несмотря на то что мультфильмов гораздо меньше, их медиана сборов выше. 
          Это может указывать на то, что мультфильмы в целом более прибыльны, 
          чем кинофильмы, или имеют большую аудиторию в США.
        """
        )

    with fees_genres:
        st.markdown(
            """
        **Гипотеза:** Фильмы с популярными жанрами собирают больше денег.
        """
        )
        st.pyplot(
            plot_fees_dummies(
                df=data,
                dummies_col="genres",
                target_col=preprocessing_config["target_column"],
                title_barplot="Распределение количества фильмов по жанрам",
                title_boxplot="Boxplot сборов фильмов в США по жанрам",
            )
        )
        st.markdown("#### Выводы:")
        st.markdown("**Сборы по Жанрам:**")
        st.markdown(
            """
          - **Драма:** Популярный жанр со средними сборами и несколькими кассовыми хитами.
          - **Комедия:** Аналогично драме, второй по популярности, с некоторыми высокими выбросами в сборах.
          - **Приключения:** Меньшая популярность, но более высокие средние сборы.
          - **Фантастика и мультфильмы:** Не самые популярные, но часто с высокими сборами.
        """
        )
        st.markdown("**Выбросы:**")
        st.write(
            """
            Показывают, что хотя большинство фильмов в определенных
            жанрах может не быть кассовыми хитами, исключения бывают.
        """
        )
        st.markdown("**Общий вывод:**")
        st.write(
            """
         Успех фильма в кассовых сборах не всегда коррелирует с популярностью его жанра.
         Менее популярные жанры могут иметь высокие средние сборы, тогда как более популярные
         жанры часто имеют выбросы, становясь кассовыми хитами, несмотря на низкие медианные сборы.
        """
        )


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model LightGBM")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]
    sep = config["preprocessing"]["data_separator"]

    upload_file = st.file_uploader(
        "Выберите файл с тестовыми данными",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )
    if upload_file:
        dataset_csv_df, files = load_data(data_path=upload_file, sep=sep)
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]
    dummies_info = config["preprocessing"]["dummies"]
    gradings = config["preprocessing"]["grading_values"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(
            unique_data_path=unique_data_path,
            dummies_info=dummies_info,
            gradings=gradings,
            endpoint=endpoint,
        )
    else:
        st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """

    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction from file": prediction_from_file,
        "Prediction": prediction,
    }

    st.sidebar.markdown("## Меню")

    selected_page = st.sidebar.radio("", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
