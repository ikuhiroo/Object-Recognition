# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys

def inference_deep(images_placeholder, keep_prob, image_size, num_classes):
    # リサイズ
    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])
    # 重みベクトル
    def weight_variable(shape):
        inital = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
        return tf.Variable(inital, name='w')
    # バイアス
    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital, name='b')
    # CV層
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name='output')
    # プーリング層
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    """畳み込み層1
    入力前 : [-1, image_size, image_size, 3], shape=(120, 56, 56, 3)
    カーネルサイズ : 16
    入力後 : [-1, image_size, image_size, 16], shape=(120, 56, 56, 16)
    """
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([3, 3, 3, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        tf.summary.histogram("weight", W_conv1)
        tf.summary.histogram("bias", W_conv1)
        tf.summary.histogram("h_conv1", h_conv1)

    """畳み込み層2
    入力前 : [-1, image_size, image_size, 16], shape=(120, 56, 56, 16)
    カーネルサイズ : 16
    入力後 : [-1, image_size, image_size, 16], shape=(120, 56, 56, 16)
    """
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([3, 3, 16, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        tf.summary.histogram("weight", W_conv2)
        tf.summary.histogram("bias", W_conv2)
        tf.summary.histogram("h_conv2", h_conv2)

    """プーリング層1
    入力前 : [-1, image_size, image_size, 16], shape=(120, 56, 56, 16)
    入力後 : [-1, image_size/2, image_size/2, 16], shape=(120, 28, 28, 16)
    """
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv2)

    """畳み込み層3
    入力前 : [-1, image_size/2, image_size/2, 16], shape=(120, 28, 28, 16)
    カーネルサイズ : 32
    入力後 : [-1, image_size/2, image_size/2, 32], shape=(120, 28, 28, 32)
    """
    with tf.name_scope("conv3") as scope:
        W_conv3 = weight_variable([3, 3, 16, 32])
        b_conv3 = bias_variable([32])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
        tf.summary.histogram("weight", W_conv3)
        tf.summary.histogram("bias", W_conv3)
        tf.summary.histogram("h_conv3", h_conv3)

    """畳み込み層3
    入力前 : [-1, image_size/2, image_size/2, 32], shape=(120, 28, 28, 32)
    カーネルサイズ : 32
    入力後 : [-1, image_size/2, image_size/2, 32], shape=(120, 28, 28, 32)
    """
    with tf.name_scope("conv4") as scope:
        W_conv4 = weight_variable([3,3,32,32])
        b_conv4 = bias_variable([32])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

        tf.summary.histogram("weight", W_conv4)
        tf.summary.histogram("bias", W_conv4)
        tf.summary.histogram("h_conv4", h_conv4)

    """プーリング層2
    入力前 : [-1, image_size/2, image_size/2, 32], shape=(120, 28, 28, 32)
    入力後 : [-1, image_size/(2*2), image_size/(2*2), 32], shape=(120, 14, 14, 32)
    """
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv4)

    """畳み込み層5
    入力前 : [-1, image_size/(2*2), image_size/(2*2), 32], shape=(120, 14, 14, 32)
    カーネルサイズ : 64
    入力後 : [-1, image_size/(2*2), image_size/(2*2), 64], shape=(120, 14, 14, 64)
    """
    with tf.name_scope("conv5") as scope:
        W_conv5 = weight_variable([3,3,32,64])
        b_conv5 = bias_variable([64])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

        tf.summary.histogram("weight", W_conv5)
        tf.summary.histogram("bias", W_conv5)
        tf.summary.histogram("h_conv5", h_conv5)

    """畳み込み層6
    入力前 : [-1, image_size/(2*2), image_size/(2*2), 64], shape=(120, 14, 14, 64)
    カーネルサイズ : 64
    入力後 : [-1, image_size/(2*2), image_size/(2*2), 64], shape=(120, 14, 14, 64)
    """
    with tf.name_scope("conv6") as scope:
        W_conv6 = weight_variable([3,3,64,64])
        b_conv6 = bias_variable([64])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

        tf.summary.histogram("weight", W_conv6)
        tf.summary.histogram("bias", W_conv6)
        tf.summary.histogram("h_conv6", h_conv6)

    """プーリング層3
    入力前 : [-1, image_size/(2*2), image_size/(2*2), 64], shape=(120, 14, 14, 64)
    入力後 : [-1, image_size/(2*2*2), image_size/(2*2*2), 64], shape=(120, 7, 7, 64)
    """
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2(h_conv6)

    """全結合層1
    入力前 : [-1, image_size/(2*2*2), image_size/(2*2*2), 64], shape=(120, 7, 7, 64)
    入力前の変換(flat) :  [image_size/(2*2*2) * image_size/(2*2*2) * 64, 1024], shape=(7*7*64, 1024)
    ノード数 : 1024
    入力後 : [-1, 1024]、 shape=(120, 1024)
    """
    with tf.name_scope("fc1") as scope:
        w = int(image_size/(2*2*2))
        W_fc1 = weight_variable([w*w*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool3_flat = tf.reshape(h_pool3, [-1, w*w*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        tf.summary.histogram("weight", W_fc1)
        tf.summary.histogram("bias", b_fc1)
        tf.summary.histogram("h_fc1_drop", h_fc1_drop)

    """全結合層2
    入力前 : [-1, 1024]、shape=(120, 1024)
    入力後 : [120, num_classes]、shape=(120, 2)
    """
    with tf.name_scope("fc2") as scope:
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        tf.summary.histogram("weight", W_fc2)
        tf.summary.histogram("bias", b_fc2)
        tf.summary.histogram("h_fc2", h_fc2)

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(h_fc2)
        tf.summary.histogram("logits", y_conv)

    return y_conv

def loss(logits, labels):
    """ lossブロック
    引数:
      logits: ロジットのtensor, float - [FLAGS.batch_size, FLAGS.num_classes]
      labels: ラベルのtensor, int32 - [FLAGS.batch_size, FLAGS.num_classes]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float

      ログオペレーション追加
    """
    with tf.name_scope('loss') as scope:
        # 交差エントロピー誤差の値の計算（学習が進むにつれて小さくなるはず）
        delta = 1e-7
        # cross_entropy_sum = -tf.reduce_sum(labels * tf.log(logits + delta))
        cross_entropy_sum = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

        # cross_entropy_mean = -tf.reduce_mean(labels * tf.log(logits))
        # cross_entropy_mean_L2 = cross_entropy_mean

        """正則化項、重み減衰を加えた損失（平均）"""
        # L2 regularization for the fully connected parameters.
        # regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
        # Add the regularization term to the loss.
        # cross_entropy_mean_L2 = cross_entropy_mean_L2 + (penalty * regularizers)

        """TensorBoardで表示するためにログを取得するオペレーションを定義"""
        tf.summary.histogram("cross_entropy_sum", cross_entropy_sum)
        # tf.summary.histogram("cross_entropy_mean", cross_entropy_mean)
        # tf.summary.histogram("cross_entropy_mean_L2", cross_entropy_mean_L2)
        tf.summary.scalar("cross_entropy_sum", cross_entropy_sum)
        # tf.summary.scalar("cross_entropy_mean", cross_entropy_mean)
        # tf.summary.scalar("cross_entropy_mean_L2", cross_entropy_mean_L2)

    return cross_entropy_sum

def training(loss, learning_rate):
    """ trainingブロック
    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のop

      最適化学習法：誤差逆伝播法を用いた勾配法（ADAM, GCD）
    """
    with tf.name_scope('training') as scope:

        #train_step = tf.train.GradientDescentOptimizer(0.01)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return train_op


def accuracy(logits, labels):
    """評価ブロック
    引数:
        logits: ブロックの結果
        labels: ラベルのtensor, int32 - [FLAGS.batch_size, FLAGS.num_classes]
    返り値:
        accuracy: 正解率(float)

         ログオペレーション追加
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # TensorBoardで表示するためにログを取得するオペレーションを定義
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

""" inference（学習モデルの生成）"""
def inference(images_placeholder, keep_prob, image_size, num_classes):

    """x_image
    [-1(flatten), height, width、チャネル数]
    images_placeholderがまず1D(784次元)に変換され、28*28に変換される
    """
    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])

    def weight_variable(shape):
        inital = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(inital, name='w')

    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital, name='b')

    def conv2d(x, W):
        """画像サイズに変化なし"""
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name='output')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    """畳み込み層1
    入力前 : [-1, image_size, image_size, 3]、shape=(120, 56, 56, 3)
    カーネルサイズ : 64
    入力後 : [-1, image_size, image_size, 64]、shape=(120, 56, 56, 64)
    """
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([5,5,3,64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        print(h_conv1)

    """プーリング層1
    入力前 : [-1, image_size, image_size, 64]
    入力後 : [-1, image_size/2, image_size/2, 64]、shape=(120, 28, 28, 64)
    """
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    """LocalResponseNormalization(LRN)
    入力前 : [-1, image_size/2, image_size/2, 64]
    入力後 : [-1, image_size/2, image_size/2, 64]
    """
    with tf.name_scope("norm1") as scope:
        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    """畳み込み層2
    入力前 : [-1, image_size/2, image_size/2, 64]
    カーネルサイズ : 64
    入力後 : [-1, image_size/2, image_size/2, 64]
    """
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([5,5,64,64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)

    """LocalResponseNormalization(LRN)
    入力前 : [-1, image_size/2, image_size/2, 64]
    入力後 : [-1, image_size/2, image_size/2, 64]
    """
    with tf.name_scope("norm2") as scope:
        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    """プーリング層2
    入力前 : [-1, image_size/2, image_size/2, 64]
    入力後 : [-1, image_size/2*2, image_size/2*2, 64]
    """
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(norm2)

    """全結合層1
    入力前 : [-1, image_size/(2*2), image_size/(2*2), 64]
    入力前の変換(flat) :  [-1, image_size/(2*2) * image_size/(2*2) * 64]
    ノード数 : 1024
    入力後 : [-1, 1024]、 shape=(120, 1024)
    """
    with tf.name_scope("fc1") as scope:
        w = int(image_size / 4)
        W_fc1 = weight_variable([w*w*64, 1024])
        b_fc1 = bias_variable([1024])
        # tf.summary.histogram("weight", W_fc1)
        # tf.summary.histogram("bias", b_fc1)

        h_pool2_flat = tf.reshape(h_pool2, [-1, w*w*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropout1の設定
        # 擬似的に平均化してくれる
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        print(h_fc1_drop)

    """全結合層1
    入力前 : [-1, image_size/(2*2), image_size/(2*2), 64]
    入力前の変換(flat) :  [-1, image_size/(2*2) * image_size/(2*2) * 64]
    ノード数 : 1024
    入力後 : [-1, 1024]、 shape=(120, 1024)
    """
    with tf.name_scope("fc2") as scope:
        W_fc2 = weight_variable([1024, num_classes])
        b_fc2 = bias_variable([num_classes])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        print(h_fc2)
        tf.summary.histogram("weight", W_fc2)
        tf.summary.histogram("bias", b_fc2)
        tf.summary.histogram("h_fc2", h_fc2)

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv
