import sys
from pathlib import Path
import pandas as pd
import numpy as np
_data_dir = Path(__file__).parent

docs_path = {'original_data': _data_dir / 'final_{}.csv',
             'embeding_matrix': _data_dir / 'embedding_x_{}.npy',
             'process_y': _data_dir / 'process_y.npy',
             'idcard_json': _data_dir / 'idcard_json.json',
             'model_path': _data_dir / 'model_{}.h5'}


def load_file(data_name, tail=False):
    file_path = str(docs_path[data_name])
    if tail:
        file_path = file_path.format(tail)
    if file_path.endswith('npy'):
        return np.load(file_path)
    elif file_path.endswith('csv'):
        return pd.read_csv(file_path)
    else:
        print('please check the {} name'.format(data_name))

if __name__ == '__main__':
    tail = 'LOAN'
    X = load_file('embeding_matrix', tail=tail)
    print(X.shape)
