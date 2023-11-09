import os
import glob
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def get_folder_list(directory_path):
    """Get a list of folders in the specified directory."""
    folder_list = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folder_list.append(item_path)
    return folder_list


def import_SIT_params(text_file_path):
    df = pd.read_excel(text_file_path)

    # 1열을 키로, 2열을 값으로 사용하여 딕셔너리 생성
    my_dict = dict(zip(df['mouse_name'], df['SIratio-postSD']))

    # 결과 출력
    return my_dict