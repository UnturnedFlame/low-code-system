import numpy as np
import pandas as pd
from scipy.io import loadmat
import os


def load_data(datafile, multiple_data=False):
    """
    :param multiple_data: 是否为多段数据
    :param datafile: 要读取的文件路径
    :return: (读取的数据, 读取数据的文件名)
    """
    file_type = os.path.basename(datafile).split('.')[-1]
    file_name = os.path.basename(datafile).split('.')[0]
    global data
    if multiple_data:
        if file_type == 'mat':
            data = loadmat(datafile)
        else:
            return 'Invalid file type'
    else:
        if file_type == 'mat':
            file_data = loadmat(datafile)
            for key, value in file_data.items():
                if isinstance(value, np.ndarray):
                    data = value
        elif file_type == 'csv':
            data = pd.read_csv(datafile)
        elif file_type == 'npy':
            data = np.load(datafile)
        else:
            return 'Invalid file type'

    return data, file_name
