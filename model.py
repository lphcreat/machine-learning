from keras.layers import Input, Dense, Dropout, concatenate
# import graph
from keras import metrics
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical, plot_model
import numpy as np
import math
import pandas as pd
from utils import normal, balance_data, embedding_method, up_sample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from graph import get_xy
from files.file_path import load_file


class Model_Graph(object):
    # if X is a list mutilin_model else creat_model

    def __init__(self, X, Y, **kwargs):
        self.X, self.Y = X, Y
        self.model = None

    def creat_model(self):
        input_matrix = Input(shape=(self.X.shape[1],))
        flatten_1 = Dense(int(math.sqrt(self.X.shape[1] * 5)), init='random_uniform',
                          activation='relu')(input_matrix)
        flatten_2 = Dense(5, init='random_uniform',
                          activation='relu')(flatten_1)
        flatten_3 = Dropout(0.3)(flatten_2)
        result = Dense(2, activation='softmax')(flatten_3)
        self.model = Model(input_matrix, result)
        optimizer = Adam(lr=3e-4)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
                           metrics.binary_accuracy])
        return self

    def mutilin_model(self):
        input_dims = []

        def flatten_matrix(item_matrix):
            input_dim = item_matrix.shape[1]
            input_matrix = Input(shape=(input_dim,))
            input_dims.append(input_dim)
            return input_matrix
        inputs = list(map(lambda item: flatten_matrix(item), self.X))
        flatten_list = []

        def flatten(inputs=inputs, input_dims=input_dims):
            for i, j in zip(inputs, input_dims):
                item = Dense(
                    int(math.sqrt(j * 5)), init='random_uniform', activation='relu')(i)
                flatten_list.append(item)
            return flatten_list
        flatten_1 = flatten()
        concat_1 = concatenate(flatten_1)
        concat = Dropout(0.2)(concat_1)
        score = Dense(5, activation='relu')(concat)
        score_final = Dense(2, activation='softmax')(score)
        # classify = Dropout(0.2)(score_final)
        self.model = Model(inputs=inputs, outputs=score_final)
        optimizer = SGD(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy',
                           metrics=[metrics.binary_accuracy, metrics.mse])
        return self

    def load_data(self):
        # 加载待预测数据
        self.X, self.Y = np.load('embedding_x.npy', 'embedding_y.npy')
        return self

    def model_fit(self, test_x, test_y):
        self.model.fit(self.X, self.Y, nb_epoch=500, batch_size=256)
        self.model.evaluate(test_x, test_y, batch_size=256)
        plot_model(self.model, 'model_shape.png', show_shapes=True)
        return self

    def predict(self, test_x):
        pre_y = self.model.predict(test_x, batch_size=256)
        return pre_y

if __name__ == '__main__':
    # 三个输入：loan、know、attribute共三个矩阵，然后combine后进行训练。
    Y = np.load('embedding_y.npy')
    X_attr = np.load('embedding_x_attr.npy')
    X_loan = load_file('embeding_matrix', tail='loan')
    X_chaxun = load_file('embeding_matrix', tail='chaxun')
    Y = np.where(Y == 'good', 1, 0)
    X_train1, X_test1, X_train2, X_test2, X_train3, X_test3, y_train_or, y_test = train_test_split(
        X_attr, X_loan, X_chaxun, Y, test_size=0.2, random_state=4, shuffle=False)
    X_train1, y_train = balance_data(X_train1, y_train_or)
    X_train2, _ = balance_data(X_train2, y_train_or)
    X_train3, _ = balance_data(X_train3, y_train_or)
    X_train1, X_test1 = normal(X_train1, X_test1)
    X_train2, X_test2 = normal(X_train2, X_test2)
    X_train3, X_test3 = normal(X_train3, X_test3)
    # print(X_train1.shape)
    # X_train1, y_train = up_sample(X_train1, y_train_or, 2)
    # X_train2, _ = up_sample(X_train2, y_train_or, 2)
    # X_train3, _ = up_sample(X_train3, y_train_or, 2)
    # print(X_train1.shape, X_train2.shape, X_train3.shape)
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    # 输入模型
    model = Model_Graph([X_train1, X_train2, X_train3], y_train)
    model.mutilin_model()
    model.model_fit([X_test1, X_test2, X_test3], y_test)
    pre_y = model.predict([X_test1, X_test2, X_test3])
    # 一个输入
    # model = Model_Graph(X_train2, y_train)
    # model.creat_model()
    # model.model_fit(X_test2, y_test)
    # pre_y = model.predict(X_test2)
    ####
    print(pre_y)
    score = classification_report(y_test, pre_y.round())
    print(score)
