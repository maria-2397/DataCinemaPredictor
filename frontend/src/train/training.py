"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import os
import requests
import joblib
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history

from ..data.json_tools import open_json


def check_overfitting(metrics: dict) -> dict:
    """
    Вычисляет степень переобучения модели.

    :param metrics: Словарь с метриками для обучающих и тестовых данных.
                    Структура словаря:
                    {'train': {'metric1': value, 'metric2': value, ...},
                     'test': {'metric1': value, 'metric2': value, ...}}

    :return: Словарь, отражающий процент переобучения для каждой метрики.
             Структура словаря:
             {'metric1': overfitting_percentage,
              'metric2': overfitting_percentage, ...}
    """

    overfitting = {}
    metrics_train = metrics["train"]
    metrics_test = metrics["test"]

    for metric in metrics_test:
        if metrics_test[metric] != 0:
            train = metrics_train[metric]
            test = metrics_test[metric]
            overfitting[metric] = round(abs((train - test) / test * 100), 2)
        else:
            overfitting[metric] = 100

    return overfitting


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    :return: None
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        old_metrics = open_json(config["train"]["metrics_path"])
    else:
        # если до этого не обучали модель и нет прошлых значений метрик
        old_metrics = {
            "test": {"mae": 0, "r2_adjusted": 0, "wape": 0},
            "train": {"mae": 0, "r2_adjusted": 0, "wape": 0},
        }

    # расчет переобучения прошлых метрик
    old_overfitting = check_overfitting(old_metrics)

    # Train
    with st.spinner("Модель подбирает параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()

    # diff metrics
    mae, r2_adjusted, wape = st.columns(3)
    mae.metric(
        "MAE",
        new_metrics["test"]["mae"],
        f"{new_metrics['test']['mae']-old_metrics['test']['mae']:.3f}",
    )

    r2_adjusted.metric(
        "R2_adjusted",
        new_metrics["test"]["r2_adjusted"],
        f"{new_metrics['test']['r2_adjusted']-old_metrics['test']['r2_adjusted']:.3f}",
    )

    wape.metric(
        "WAPE%",
        new_metrics["test"]["wape"],
        f"{new_metrics['test']['wape']-old_metrics['test']['wape']:.3f}",
    )

    # Overfitting
    new_overfitting = check_overfitting(new_metrics)
    mae_overfitting, r2_adjusted_overfitting, wape_overfitting = st.columns(3)

    mae_overfitting.metric(
        "MAE_overfitting%",
        round(new_overfitting["mae"], 2),
        f"{new_overfitting['mae']-old_overfitting['mae']:.2f}",
    )

    r2_adjusted_overfitting.metric(
        "R2_adjusted_overfitting%",
        round(new_overfitting["r2_adjusted"], 2),
        f"{new_overfitting['r2_adjusted']-old_overfitting['r2_adjusted']:.2f}",
    )

    wape_overfitting.metric(
        "WAPE_overfitting%",
        round(new_overfitting["wape"], 2),
        f"{new_overfitting['wape']-old_overfitting['wape']:.2f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
