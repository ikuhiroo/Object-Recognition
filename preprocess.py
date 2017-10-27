import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys

def input(self):
    # 空クラス作成
    class Preprocess(object):
        pass

    result = Preprocess()

    """ データの取得
        train_images[0].shape → (784,)
        train_labels[0].shape → (2,)
        train_images.shape[0] →　4781
        train_labels.shape[0] →　4781
        train_images.shape → (4781, 784)
        train_labels.shape → (4781, 2)
    """
    with open('train_image.dump', 'rb') as f:
        images = pickle.load(f)
        images = np.array(images)

    with open('train_label.dump', 'rb') as f:
        labels = pickle.load(f)
        labels = np.array(labels)

    # snickesのみ取得
    """クラスごとに分割
    snickers_labels(139, ?)
    other_labels(4642, ?)
    """
    snickers_images = images[3970:4109]
    snickers_labels = labels[3970:4109]
    other_images = images[0:3970]
    other_images = np.append(other_images, images[4109:len(images)], axis=0)
    other_labels = labels[0:3970]
    other_labels = np.append(other_labels, labels[4109:len(labels)], axis=0)

    """訓練データとテストデータの分割"""
    split = 10
    # それぞれのindexからランダムにsplit個抽出したリスト
    rand_list_snickers = sorted(random.sample(list(range(len(snickers_labels))), split))
    rand_list_others = sorted(random.sample(list(range(len(other_labels))), split))

    # 空リストの作成
    train_snickers_images = np.empty((len(snickers_labels)-split, 784))
    train_snickers_labels = np.empty((len(snickers_labels)-split, 2))
    train_other_images = np.empty((len(other_labels)-split, 784))
    train_other_labels = np.empty((len(other_labels)-split, 2))
    test_snickers_images = np.empty((10, 784))
    test_snickers_labels = np.empty((10, 2))
    test_other_images = np.empty((10, 784))
    test_other_labels = np.empty((10, 2))


    cnt_snickers_test = 0
    cnt_snickers_train = 0
    for index in range(len(snickers_images)):
        # テストデータセット
        if index in rand_list_snickers:
            test_snickers_images[cnt_snickers_test] = snickers_images[index]
            test_snickers_labels[cnt_snickers_test] = snickers_labels[index]
            cnt_snickers_test += 1
        else: #訓練データセット
            train_snickers_images[cnt_snickers_train] = snickers_images[index]
            train_snickers_labels[cnt_snickers_train] = snickers_labels[index]
            cnt_snickers_train += 1

    cnt_other_test = 0
    cnt_other_train = 0
    for index in range(len(other_images)):
        #     # テストデータセット
        if index in rand_list_others:
            test_other_images[cnt_other_test] = other_images[index]
            test_other_labels[cnt_other_test] = other_labels[index]
            cnt_other_test += 1
        else:# 訓練データセット
            train_other_images[cnt_other_train] = other_images[index]
            train_other_labels[cnt_other_train] = other_labels[index]
            cnt_other_train += 1

    # print(len(test_other_labels)) #10
    # print(len(train_other_labels)) #4632

    result.train_snickers_images = train_snickers_images
    result.train_snickers_labels = train_snickers_labels
    result.train_other_images = train_other_images
    result.train_other_labels = train_other_labels

    result.test_snickers_images = test_snickers_images
    result.test_snickers_labels = test_snickers_labels
    result.test_other_images = test_other_images
    result.test_other_labels = test_other_labels

    # 訓練データセットの更新
    result.train_images = np.append(train_snickers_images, train_other_images, axis=0)
    result.train_labels = np.append(train_snickers_labels, train_other_labels, axis=0)

    # テストデータセットの更新
    result.test_images = np.append(test_snickers_images, test_other_images, axis=0)
    result.test_labels = np.append(test_snickers_labels, test_other_labels, axis=0)

    return result

# """ データの取得
#     train_images[0].shape → (784,)
#     train_labels[0].shape → (2,)
#     train_images.shape[0] →　4781
#     train_labels.shape[0] →　4781
#     train_images.shape → (4781, 784)
#     train_labels.shape → (4781, 2)
# """
# with open('train_image.dump', 'rb') as f:
#     train_images = pickle.load(f)
#     train_images = np.array(train_images)
#
# with open('train_label.dump', 'rb') as f:
#     train_labels = pickle.load(f)
#     train_labels = np.array(train_labels)
#
# # snickesのみ取得
# """クラスごとに分割
# snicker_labels(139, ?)
# other_labels(4642, ?)
# """
# snicker_images = test_images[3970:4109]
# snicker_labels = test_labels[3970:4109]
# other_images = test_images[0:3970]
# other_images = np.append(other_images, test_images[4109:len(test_images)], axis=0)
# other_labels = test_labels[0:3970]
# other_labels = np.append(other_labels, test_labels[4109:len(test_labels)], axis=0)
#
# """訓練データとテストデータの分割"""
# split = 10
# # それぞれのindexからランダムにsplit個抽出したリスト
# rand_list_snickers = sorted(random.sample(list(range(len(snicker_labels))), split))
# rand_list_others = sorted(random.sample(list(range(len(other_labels))), split))
#
# # 空リストの作成
# train_snickers_images = np.empty((split, 784))
# train_snickers_labels = np.empty((split, 2))
# train_other_images = np.empty((split, 784))
# train_other_labels = np.empty((split, 2))
# test_snickers_images = np.empty((split, 784))
# test_snickers_labels = np.empty((split, 2))
# test_other_images = np.empty((split, 784))
# test_other_labels = np.empty((split, 2))
#
# for index in range(len(train_images)):
#     # テストデータセット
#     if index in rand_list_snickers:
#         test_snickers_images[index] = snicker_images[rand_list_snickers[index]]
#         test_snickers_labels[index] = snickers_labels[rand_list_snickers[index]]
#     # テストデータセット
#     elif index in rand_list_others:
#         test_other_images[index] = other_images[rand_list_others[index]]
#         test_other_labels[index] = other_labels[rand_list_others[index]]
#     else:# 訓練データセット
#         train_snickers_images[index] = snicker_images[rand_list_snickers[index]]
#         train_snickers_labels[index] = snickers_labels[rand_list_snickers[index]]
#         train_other_images[index] = other_images[rand_list_others[index]]
#         train_other_labels[index] = other_labels[rand_list_others[index]]
#
# # 訓練データセットの更新
# _train_images = np.append(train_snickers_images, train_other_images, axis=0)
# _train_labels = np.append(train_snickers_labels, train_other_labels, axis=0)
#
# # テストデータセットの更新
# _test_images = np.append(test_snickers_images, test_other_images, axis=0)
# _test_labels = np.append(test_snickers_labels, test_other_labels, axis=0)
