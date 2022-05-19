import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random

path_data = './packet_info_data.xlsx'
pd.set_option('mode.chained_assignment',  None)


def augmentation(source):
    data_copy = pd.DataFrame.copy(source)
    attr_ps = '패킷 사이즈'
    attr_ds = '데이터 사이즈'

    threshold = 0.2

    for i in range(len(data_copy[attr_ps])):
        min_ps = int(data_copy[attr_ps][i] * (1 - threshold))
        max_ps = int(data_copy[attr_ps][i] * (1 + threshold))
        data_copy[attr_ps][i] = random.randint(min_ps, max_ps)

    for i in range(len(data_copy[attr_ds])):
        data_copy[attr_ds][i] = random.randint(int(data_copy[attr_ps][i] * 0.8), int(data_copy[attr_ps][i] * 0.9))

    output = pd.concat([source, data_copy])

    return output


dataset = pd.read_excel(path_data)

round_aug = 7
for idx in range(round_aug):
    print("Round {0} Start".format(idx + 1))
    dataset = augmentation(dataset)
    dataset.reset_index(inplace=True, drop=True)

dataset.to_excel('packet_info_data.xlsx', index=False)
