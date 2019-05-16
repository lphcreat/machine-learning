from networkx import DiGraph
import networkx as nx
import pandas as pd
import numpy as np
# from py2neo import Graph
from matplotlib import pyplot
from sklearn.decomposition import TruncatedSVD
from utils import idcard2dict, embedding_method
import json
import sys
from files.file_path import load_file, docs_path

# 链接数据库

# def query_data(relation):
#     if relation == 'KNOW':
#         cquery = f'MATCH(p) - [r:{relation}]->(n) RETURN p.ID_CARD, p.flag, n.ID_CARD, r.CNT, r.DURATION_TIME, p.id, n.id limit 50000'
#     else:
#         cquery = f'MATCH(n) - [r:{relation}]->(p) RETURN p.ID_CARD, p.flag, n.ID_CARD, r.CNT, r.DURATION_TIME, p.id, n.id limit 100000'
#     query_df = graph.run(cquery).to_data_frame()
#     relation_df = query_df[['p.id', 'n.id', 'p.ID_CARD']]
#     attribute_df = query_df[['p.ID_CARD', 'n.ID_CARD',
#                              'r.CNT', 'r.DURATION_TIME', 'p.flag']]
#     return relation_df, attribute_df


def query_from_file(tail):
    query_df = load_file('original_data', tail=tail)
    relation_df = query_df[['start_id', 'end_id']]
    if tail == 'loan':
        attribute_df = query_df[
            ['days90_overdue', 'loan_count', 'overdue', 'years5_overdue']]
    else:
        attribute_df = query_df[['call_time', 'count']]
    return relation_df, attribute_df


def generate_graph(df, idcard_dict):
    nodes = df['start_id'].unique().tolist() + df['end_id'].unique().tolist()
    node_list = list(set(nodes))
    # 保证客户位置固定并返回idcard_dict，保持两个graph顺序一致
    # node_list.sort(key=nodes.index)
    c_id = [i for i in range(len(node_list))]
    transform = dict(zip(node_list, c_id))
    p_id = set(idcard_dict.values())
    c_id = list(set(c_id).difference(p_id))
    # 对datafram 排序
    df['index'] = df['start_id'].map(transform)
    df.sort_values(by=['index'], inplace=True)
    df['eds'] = df.apply(lambda x: (transform[x['start_id']],
                                    transform[x['end_id']]), axis=1)
    digraph = DiGraph()
    digraph.add_edges_from(df['eds'].tolist())
    graph_matrix = nx.adjacency_matrix(digraph)
    # graph_matrix = graph_matrix[~np.eye(graph_matrix.shape[0], dtype=bool)].reshape(graph_matrix.shape[0], -1)
    # 去除node对角线元素
    p_id = list(p_id)
    graph_matrix = graph_matrix[p_id]
    graph_matrix = graph_matrix[:, c_id]
    return digraph, graph_matrix, p_id, c_id


def embedding_graph(graph_matrix):
    print('embedding graph')
    n_components = 100
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(graph_matrix)
    svd_ratio = svd.explained_variance_ratio_.sum()
    while svd_ratio < 0.9:
        n_components += 10
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(graph_matrix)
        svd_ratio = svd.explained_variance_ratio_.sum()
    embedding_matrix = svd.transform(graph_matrix)
    return embedding_matrix


def concat_matrix(matrixs):
    A, B = matrixs
    A = A.astype('float64').toarray()
    B = B.astype('float64')
    matrixs = list(map(lambda x: np.multiply(A, B[:, x]),
                       [x for x in range(B.shape[1])]))
    final_matrix = np.concatenate(matrixs, axis=1)
    return final_matrix


def get_xy(relation_type, idcard_dict, save=True):
    relation_df, attribute_df = query_from_file(tail=relation_type)
    digraph, graph_matrix, p_id, c_id = generate_graph(
        relation_df, idcard_dict)
    matrixs = (graph_matrix, attribute_df.values[c_id])
    final_matrix = concat_matrix(matrixs)
    final_matrix = embedding_graph(final_matrix)
    nx_matrix = embedding_method(digraph, p_id)
    combine_matrix = np.concatenate([final_matrix, nx_matrix], axis=1)
    if save:
        np.save(str(docs_path['embeding_matrix']).format(
            relation_type), combine_matrix)
    return combine_matrix
if __name__ == '__main__':
    cust_df = load_file('original_data', tail='cust')
    idcard_dict = idcard2dict(cust_df[['loan_id:ID']])
    json_path = docs_path['idcard_json']
    with open(str(json_path), "w") as f:
        json.dump(idcard_dict, f)
    # 标签数据
    p_id = list(idcard_dict.values())
    y = cust_df['classification'].values[p_id]
    # 客户属性
    attr_cust = pd.get_dummies(
        cust_df[['CITY', 'SEX', 'MARITAL', 'EDU', 'INDUSTRY', 'POSITION', 'SALARY']])
    attr_cust = attr_cust.values[p_id]
    np.save('embedding_y.npy', y)
    np.save('embedding_x_attr.npy', attr_cust)
    # 关系矩阵
    get_xy('loan', idcard_dict)
    print('finishing')
    get_xy('call', idcard_dict)
    print('finishing')
