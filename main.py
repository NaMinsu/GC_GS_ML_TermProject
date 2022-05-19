import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
    attrib = dataset.columns.values.tolist()
    categories = ['패킷 구분 명', '패킷 명', '패킷 단위', '패킷 범주', '설명']

    dataset.drop('공개 구분 명', axis=1, inplace=True)
    dataset.drop('데이터 사이즈', axis=1, inplace=True)
    dataset = encoding_scaling(dataset, categories)

    imputer = SimpleImputer()
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

    matrix_corr = dataset.corr()
    print(matrix_corr["패킷 사이즈"].sort_values(ascending=False))
    print(matrix_corr["패킷 명"].sort_values(ascending=False))
    print(matrix_corr["송신 서버 번호"].sort_values(ascending=False))

    y = pd.DataFrame(dataset['패킷 사이즈'], columns=['패킷 사이즈'])
    x = dataset.drop('패킷 사이즈', axis=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=123)

    model = XGBRegressor()

    print("Training Start")
    start = time.time()
    model.fit(x_train, y_train, verbose=False)
    end = time.time()
    print("Training End")

    print("Prediction Start")
    pred = model.predict(x_val)
    print("Prediction End")

    acc = model.score(x_val, y_val)
    error = mean_squared_error(y_val, pred)
    print("Learning time: {0}, Accuracy: {1}, Loss: {2}".format(end - start, acc, error))
