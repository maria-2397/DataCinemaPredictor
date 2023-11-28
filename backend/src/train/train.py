"""
Программа: Тренировка данных
Версия: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor
import optuna
from optuna import Study

from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective(
    trial,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_estimators: int,
    n_folds: int,
    log_target: bool,
    random_state: int,
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_estimators: количество базовых алгоритмов
    :param n_folds: количество фолдов
    :param: log_target: условие логарифмированной целевой переменной
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """

    lgb_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [n_estimators]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "max_bin": trial.suggest_int("max_bin", 20, 3300, step=10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 3000, step=10),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "objective": trial.suggest_categorical("objective", ["mae"]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
    }

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_mae_scores = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l1")

        model = LGBMRegressor(**lgb_params, early_stopping_rounds=100, verbose=-1)

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="mae",
            callbacks=[pruning_callback],
        )

        predictions = model.predict(x_test)
        # Вычисление MAE на исходной шкале данных
        if log_target:
            mae_score = mean_absolute_error(np.expm1(y_test), np.expm1(predictions))
        else:
            mae_score = mean_absolute_error(y_test, predictions)

        cv_mae_scores[idx] = mae_score

    return np.mean(cv_mae_scores)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMClassifier tuning, Study]
    """
    # возвращаем study чтобы потом использовать для визуализации
    # и чтобы потом использовать лучшие параметры из этого Study

    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="minimize", study_name="LGB")

    def function(trial):
        return objective(
            trial,
            x_train,
            y_train,
            kwargs["n_estimators"],
            kwargs["n_folds"],
            kwargs["log_target"],
            kwargs["random_state"],
        )

    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame, data_test: pd.DataFrame, study: Study, **kwargs
) -> LGBMRegressor:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :return: LGBMClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    # training optimal params
    reg = LGBMRegressor(**study.best_params, silent=True, verbose=-1)
    reg.fit(x_train, y_train)

    # save train and test metrics
    save_metrics(
        x_test=x_test,
        y_test=y_test,
        x_train=x_train,
        y_train=y_train,
        model=reg,
        metric_path=kwargs["metrics_path"],
        target_is_log=kwargs["log_target"],
    )

    return reg
