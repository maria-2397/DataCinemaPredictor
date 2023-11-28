"""
Программа: Модель для прогнозирования сборов фильмов в США
Версия: 1.0
"""

import warnings

import optuna
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.evaluate.evaluate import pipeline_evaluate
from src.pipelines.pipeline import pipeline_training
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATH = "../config/params.yml"
app = FastAPI()


class MovieInfo(BaseModel):
    """
    Признаки для получения результатов модели
    """

    year: int
    votes_kp: float
    votes_imdb: float
    votes_filmCritics: float
    votes_await: float
    rating_kp: float
    rating_imdb: float
    rating_filmCritics: float
    movieLength: float
    ageRating: float
    ratingMpaa: str
    type: str
    genres: list
    countries: list
    videos_trailers_number: float
    budget: float
    actor_metric: float
    writer_metric: float
    director_metric: float
    Special_effects: list
    Production: list


@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {"message": "Hello Data Scientist!"}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return metrics


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(movie: MovieInfo):
    """
    Предсказание модели по введенным данным
    """
    features = [
        [
            movie.year,
            movie.votes_kp,
            movie.votes_imdb,
            movie.votes_filmCritics,
            movie.votes_await,
            movie.rating_kp,
            movie.rating_imdb,
            movie.rating_filmCritics,
            movie.movieLength,
            movie.ageRating,
            movie.ratingMpaa,
            movie.type,
            movie.genres,
            movie.countries,
            movie.videos_trailers_number,
            movie.budget,
            movie.actor_metric,
            movie.writer_metric,
            movie.director_metric,
            movie.Special_effects,
            movie.Production,
        ]
    ]

    cols = [
        "year",
        "votes_kp",
        "votes_imdb",
        "votes_filmCritics",
        "votes_await",
        "rating_kp",
        "rating_imdb",
        "rating_filmCritics",
        "movieLength",
        "ageRating",
        "ratingMpaa",
        "type",
        "genres",
        "countries",
        "videos_trailers_number",
        "budget",
        "actor_metric",
        "writer_metric",
        "director_metric",
        "Special_effects",
        "Production",
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    result = {f"Сборы фильма в США составят ${predictions:,.0f}!"}

    return result
