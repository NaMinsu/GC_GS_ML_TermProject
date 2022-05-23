import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

path_data = './packet_info_data.xlsx'


def encoding_scaling(df, columns):
    encoder = OrdinalEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(pd.DataFrame(df[column]).astype('str'))

    scaler = RobustScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df


if __name__ == "__main__":
    dataset = pd.read_excel(path_data)
    print("Pure Dataset")
    print(dataset.head())
    attrib = dataset.columns.values.tolist()
    categories = ['패킷 구분 명', '패킷 명', '패킷 단위', '패킷 범주', '설명']

    dataset.drop('공개 구분 명', axis=1, inplace=True)
    dataset.drop('데이터 사이즈', axis=1, inplace=True)
    print("Dataset after drop")
    print(dataset.head())
    print(dataset.isna().sum())

    imputer = SimpleImputer(strategy='most_frequent')
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    print("Dataset after imputation")
    print(dataset.head())
    print(dataset.isna().sum())

    dataset = encoding_scaling(dataset, categories)
    print("Dataset after encoding & scaling")
    print(dataset.head())

    matrix_corr = dataset.corr()
    print(matrix_corr["패킷 사이즈"].sort_values(ascending=False))
    print(matrix_corr["패킷 명"].sort_values(ascending=False))
    print(matrix_corr["송신 서버 번호"].sort_values(ascending=False))

    y = pd.DataFrame(dataset['패킷 사이즈'], columns=['패킷 사이즈'])
    x = dataset.drop('패킷 사이즈', axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=123)

    model = XGBRegressor()
    model.fit(x_train, y_train, verbose=False)
    pred = model.predict(x_val)

    print("XGBoost Result")
    acc = model.score(x_val, y_val)
    error = mean_squared_error(y_val, pred)
    print("Accuracy: {0}, Loss: {1}".format(acc, error))

    x_ax = range(len(y_val))
    plt.plot(x_ax, y_val, label="original")
    plt.plot(x_ax, pred, label="predicted")
    plt.title("Prediction Result")
    plt.legend()
    plt.show()

    params = {
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "min_split_loss": [0, 5, 10, 15, 20, 25, 30],
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "max_delta_step": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "subsample": [0, 0.5, 1]
    }

    best_score = 0
    cross_valid = 5
    rand_cv = RandomizedSearchCV(model, param_distributions=params, cv=cross_valid,
                                 scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1, random_state=123)
    rand_cv.fit(x_train, y_train.values.ravel())
    pred_cv = rand_cv.predict(x_val)
    print("Best loss: {}".format(mean_squared_error(y_val, pred_cv)))
    print("Best parameter: {}".format(rand_cv.best_params_))
