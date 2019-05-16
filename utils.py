from sklearn import preprocessing
import networkx as nx
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.pipeline import Pipeline
from networkx.algorithms import centrality
import numpy as np
import pandas as pd
from files.file_path import load_file, docs_path


def normal(train, x_test):
    normalizer = preprocessing.StandardScaler().fit(train)
    X = normalizer.transform(train)
    x_test = normalizer.transform(x_test)
    return X, x_test


def customer_attrs(tail, idcard_dict):
    # return node attribute like address age ...
    attributes = load_file('original_data', tail='attribute')
    # sort with idcard_dict
    # todo one—hot
    return attributes


def idcard2dict(df):
    # param: datafram data have the id_card col
    # return idcard to id for keep the sequence of nodes
    unique_p_num = len(df['loan_id:ID'].unique())
    id_card = df['loan_id:ID'].unique().tolist()
    p_id = [i for i in range(unique_p_num)]
    # 身份证号dict
    idcard_dict = dict(zip(id_card, p_id))
    return idcard_dict


def embedding_method(digraph, p_id):
    # return graph features like degree short_path...
    # 连接数
    indegree = centrality.degree_centrality(digraph)
    # 中心性
    node_centrality = centrality.eigenvector_centrality_numpy(digraph)
    # 社区数
    # clique = nx.algorithms.clique.number_of_cliques(digraph.to_undirected())
    dict2matrix = lambda x: pd.DataFrame.from_dict(
        x, orient='index').values[p_id]
    concats = list(map(dict2matrix, [indegree, node_centrality]))
    design_matrix = np.concatenate(concats, axis=1)
    return design_matrix


def balance_data(X, Y):
    X_resampled, y_resampled = SMOTE().fit_resample(X, Y)
    return X_resampled, y_resampled


def up_sample(X, Y, num):
    listx = [X]
    listy = [Y]
    insert_data = X[Y == 0]
    insert_y = Y[Y == 0]
    for i in range(num):
        listx.append(insert_data)
        listy.append(insert_y)
    pred_X = np.concatenate(listx, axis=0)
    pred_y = np.concatenate(listy, axis=0)
    return pred_X, pred_y


def query_from_file(tail):
    query_df = load_file('original_data', tail=tail)
    attributes=set(query_df.columns)-set(['start_id','end_id'])
    # if tail == 'loan':
    #     attributes = ['days90_overdue', 'loan_count',
    #                   'overdue', 'years5_overdue']
    # else:
    #     attributes = ['count']
    return query_df, list(attributes)

if __name__ == "__main__":
    _,cols=query_from_file('chaxun')
    print(cols)
    
