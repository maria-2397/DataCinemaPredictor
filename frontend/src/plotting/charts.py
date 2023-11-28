"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import ast
import pandas as pd
import numpy as np

import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
ENG_FORMAT = ticker.EngFormatter()


def plot_count_year_fees(
    df: pd.DataFrame, target_col: str, year_col: str
) -> matplotlib.figure.Figure:
    """
    Построение трёх графиков для анализа сборов и количества фильмов по годам
    :param df: датасет
    :param target_col: имя колонки с информацей о сборах фильмов
    :param year_col: имя колонки, содержащей информацию о годе выхода фильмов
    :return: поле рисунка
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))  # 3 графика
    sns.set(font_scale=1.0)

    max_year = int(df[year_col].max())
    min_year = int(df[year_col].min())

    sns.lineplot(data=df, x=year_col, y=target_col, ax=axs[0], estimator=np.median)
    axs[0].set_title("Распределение медианных сборов  в США по годам", fontsize=16)
    axs[0].set_ylabel("Сборы")

    sns.boxplot(data=df, x=year_col, y=target_col, showfliers=True, ax=axs[1])
    axs[1].set_title("Boxplot сборов фильмов в США в каждый год", fontsize=16)
    axs[1].set_ylabel("Сборы")
    axs[1].tick_params(axis="x", rotation=45)

    df["movie_count"] = df.groupby(year_col).transform("size")
    sns.lineplot(data=df, x=year_col, y="movie_count", ax=axs[2], color="green")
    axs[2].set_title("Распределение количества фильмов в США по годам", fontsize=16)
    axs[2].set_ylabel("Количество фильмов")

    for ax in axs:
        ax.yaxis.set_major_formatter(ENG_FORMAT)

    axs[0].set_xticks(list(range(min_year, max_year + 1, 1)))
    axs[2].set_xticks(list(range(min_year, max_year + 1, 1)))

    plt.tight_layout()
    return fig


def plot_distributions(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title_kde: str,
    title_boxplot: str,
) -> matplotlib.figure.Figure:
    """
    Построение KDE и Boxplot графиков для распределений
    числовых значений по категориям в DataFrame.

    :param df: DataFrame с числовыми и категориальными данными
    :param category_col: имя категориальной колонки для группировки
    :param value_col: имя числовой колонки для распределений
    :param title_kde: заголовок для KDE-графика
    :param title_boxplot: заголовок для boxplot
    :return: поле рисунка
    """
    # Удаляем пропуски и получаем уникальные значения для категориальной колонки
    unique_categories = df[category_col].dropna().unique()
    unique_categories.sort()

    # Создаем словарь для данных
    data = {
        str(category): df[df[category_col] == category][value_col]
        for category in unique_categories
    }

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # KDE plot
    sns.kdeplot(data=data, common_norm=False, ax=axes[0])
    # Boxplot
    sns.boxplot(
        data=df, y=category_col, x=value_col, orient="h", showfliers=True, ax=axes[1]
    )

    axes[0].set_title(title_kde, fontsize=16)
    axes[1].set_title(title_boxplot, fontsize=16)

    axes[1].xaxis.set_major_formatter(ENG_FORMAT)
    axes[0].xaxis.set_major_formatter(ENG_FORMAT)

    # Установка одинаковых пределов для оси X для обоих графиков
    xlimits = axes[0].get_xlim()
    axes[1].set_xlim(xlimits)

    plt.tight_layout()
    return fig


def plot_category_distribution(
    df: pd.DataFrame, category_col: str, title: str = None
) -> matplotlib.figure.Figure:
    """
    Построение столбчатой диаграммы распределения значений
    в указанной категориальной колонке DataFrame
    :param df: датасет с категориальными данными
    :param category_col: имя колонки для построения диаграммы распределения
    :param title: заголовок для графика
    :return: поле рисунка
    """
    # Получаем подсчеты количества для категории
    category_counts = df[category_col].value_counts().reset_index()
    category_counts.columns = [category_col, "Количество"]

    fig = plt.figure(figsize=(12, 5))
    sns.barplot(
        x=category_col,
        y="Количество",
        data=category_counts,
        order=category_counts[category_col],
    )

    # Настройка заголовка
    if title is None:
        title = f"Распределение по категории {category_col}"
    plt.title(title, fontsize=16)

    return fig


def parse_series_to_lists(series: pd.Series) -> pd.Series:
    """
    Преобразование элементов Series в списки
    :param series: объект Series для преобразования
    :return: обновленный объект Series с типом list
    """
    # Заполняем отсутствующие значения пустыми списками
    series = series.fillna("[]")
    # Преобразуем каждый элемент в список
    series = series.apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x))
    return series


def make_dummies(df: pd.DataFrame, col: str, top_dummies_num: int = 15) -> pd.DataFrame:
    """
    Создает бинаризованные колонки для указанного признака в dataframe
    :param df: датасет
    :param col: колонка для бинаризации
    :param top_dummies_num: топ количество самых часто встречающихся колонок
    :return: датасет с бинаризованными колонками
    """
    df[col] = parse_series_to_lists(df[col])

    dummies = df[col].str.join("|").str.get_dummies(sep="|")
    sorted_counts = dummies.sum().sort_values(ascending=False)
    top_cols = sorted_counts.iloc[:top_dummies_num].index
    return dummies[top_cols]


def plot_fees_dummies(
    df: pd.DataFrame,
    target_col: str,
    dummies_col: str,
    title_barplot: str,
    title_boxplot: str,
) -> matplotlib.figure.Figure:
    """
    Строит столбчатую диаграмму и boxplot, используя данные
    целевой переменной и категории
    :param df: датасет
    :param target_col: имя колонки, содержащей целевую переменную
    :param dummies_col: имя колонки для бинаризации
    :param title_barplot: заголовок для столбчатой диаграммы
    :param title_boxplot: заголовок для boxplot
    :return: поле рисунка
    """
    dummies = make_dummies(df, dummies_col)
    counts = dummies.sum()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    sns.barplot(x=counts.index, y=counts.values, palette="turbo", ax=axes[0])
    axes[0].set_title(title_barplot, fontsize=16)

    # Временный DataFrame для построения графиков
    df_temp = pd.concat([df[[target_col]], dummies], axis=1)
    data = {}
    for col in dummies.columns:
        data[col] = df_temp[df_temp[col] == 1][target_col]

    sns.boxplot(data=pd.DataFrame(data), palette="turbo", ax=axes[1])
    axes[1].set_title(title_boxplot, fontsize=16)

    # Настройка формата осей
    for ax in axes:
        ax.yaxis.set_major_formatter(ENG_FORMAT)
        ax.set_xticklabels(dummies.columns, rotation=90, fontsize=14)
        ax.tick_params(axis="x", rotation=65)

    plt.tight_layout()
    return fig
