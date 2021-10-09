"""Docstring"""
from typing import Tuple, Type, Iterable
from pandas import DataFrame, date_range, read_csv
from dataclasses import dataclass, InitVar
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from pickle import load as p_load
from src.utils import *

import xgboost as xgb
import matplotlib.pyplot as plt


@dataclass
class predictor:
    """docstring for predictor"""

    path: InitVar[str]
    main_column: str

    def __post_init__(self, path):
        df = read_csv(path)
        df = set_df_index(df, "Datetime")
        self.train_set, self.validation_set, self.test_set = split_dataset(
            df, 0.7
        )

    def prepare_to_train(self, lags: Iterable[int], rolls: Iterable[int]):
        create_date_features(self.train_set)
        create_date_features(self.validation_set)
        create_date_features(self.test_set)

        # date_features = self.train_set.filter(regex="^date_").columns

        create_circular_features(self.train_set)
        create_circular_features(self.validation_set)
        create_circular_features(self.test_set)

        create_lag_features(self.train_set, self.main_column, lags)
        create_lag_features(self.validation_set, self.main_column, lags)
        create_lag_features(self.test_set, self.main_column, lags)

        create_rolling_features(self.train_set, self.main_column, rolls)
        create_rolling_features(self.validation_set, self.main_column, rolls)
        create_rolling_features(self.test_set, self.main_column, rolls)

    def train_xgboost(
        self,
        col_regex: str = "^lag|^date",
        plot_importance: bool = True,
    ):
        """Fit the xgboost regressor"""
        X_train = self.train_set.filter(regex=col_regex)
        X_validation = self.validation_set.filter(regex=col_regex)

        y_train = self.train_set[self.main_column]
        y_validation = self.validation_set[self.main_column]

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=2000, max_depth=10, learning_rate=0.01
        )

        tic()
        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_validation, y_validation)],
            early_stopping_rounds=50,
            verbose=200,
        )
        toc()

        # Plot the feature importance of the xgboost model
        if plot_importance:
            plt.style.use("fivethirtyeight")
            xgb.plot_importance(
                self.xgb_model, height=0.9, max_num_features=10
            )
            plt.show()

    def test_xgboost(
        self, col_name: str = "xgb_prediction", regex: str = "^lag|^date"
    ):
        """Use the test set in the xgboost model"""
        X_test = self.test_set.filter(regex=regex)
        tic()
        self.test_set[col_name] = self.xgb_model.predict(X_test)
        toc()

    def load_xgb_model(self, path: str):
        """Load the model from the models folder"""
        self.xgb_model.load_model(path)

    def train_xgb_multioutput(self, **kargs):
        """Fit the multioutput xgboost regressor"""
        X_train, y_train = self._prepare_multioutput_features(**kargs)

        self.multiout_xgb = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=2000, max_depth=10, learning_rate=0.01
            )
        )

        tic()
        self.multiout_xgb.fit(X_train, y_train)
        toc()

    def test_moxgb(
        self,
        lags: Iterable,
        col_name: str = "xgbmo_prediction",
        regex: str = "^lag|^date",
    ):
        """Use the test set in the xgboost model"""
        create_multioutput_features(self.test_set, self.main_column, lags=lags)
        X_test = self.test_set.filter(regex=regex)
        tic()
        self.test_set[col_name] = self.xgb_model.predict(X_test)
        toc()

    def _prepare_multioutput_features(self, col_regex: str, lags: Iterable):
        """Docstring"""
        create_multioutput_features(
            self.train_set, self.main_column, lags=lags
        )
        X_train = self.train_set.filter(regex=col_regex).fillna(0)

        y_train = self.train_set.filter(
            regex=self.main_column + "|^y\d"
        ).fillna(0)

        return X_train, y_train

    def train_svr_multioutput(self, **kargs):
        """Fit the multioutput SVR model"""
        X_train, y_train = self._prepare_multioutput_features(**kargs)

        self.multiout_svr = MultiOutputRegressor(
            make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=1e3, gamma=0.1, cache_size=1000),
            )
        )

        tic()
        self.multiout_svr.fit(X_train, y_train)
        toc()

    def load_mosvr_model(self, path: str):
        """Load the model from the models folder"""
        self.multiout_svr = p_load(open(path, "rb"))

    def train_knn_model(self, **kargs):
        """Docstring"""
        X_train, y_train = self._prepare_multioutput_features(**kargs)

        self.multiout_knn = MultiOutputRegressor(
            make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(
                    n_neighbors=5,
                    weights="distance",
                    algorithm="auto",
                    leaf_size=30,
                    p=2,
                ),
            )
        )

        tic()
        self.multiout_knn.fit(X_train, y_train)
        toc()

    def predict(self, model, regex, col_name: str, limit: int = 26):
        """Docstring"""
        forecast = DataFrame(columns=[col_name])
        X_test = self.test_set.copy()

        for i in range(11, limit, 6):
            sample = X_test.filter(regex=regex).iloc[i : i + 1]
            prediction = model.predict(sample)

            X_test.iloc[i : i + 6].PJME_MW = prediction.T
            create_lag_features(
                X_test, self.main_column, lags=[1, 3, 5, 7, 9, 11]
            )
            create_rolling_features(X_test, self.main_column, windows=[6, 12])

            forecast = forecast.append(
                DataFrame(
                    prediction.T,
                    columns=[col_name],
                    index=X_test.iloc[i : i + 6].index,
                )
            )

        return forecast
