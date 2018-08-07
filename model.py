# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys

# fine_tuning用モデル
def VGG(images_placeholder, keep_prob, image_size, num_classes, weights):

    # パラメータ保存用
    parameters = []

    # 出力層保存用
    value = []

    # 入力画像
    x_image = tf.reshape(images_placeholder, [-1, image_size, image_size, 3])

    """basic"""
    def weight_variable(shape):
        inital = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1, seed=None)
        return tf.Variable(inital, name='w', trainable=True)

    def bias_variable(shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital, name='b', trainable=True)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name='output')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    """zero-mean input"""
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        x_image = x_image - mean

    """畳み込み層1_1"""
    with tf.name_scope("conv1_1") as scope:
        conv1_1_W = tf.Variable(weights['conv1_1_W'], name='w_', trainable=False)
        conv1_1_b = tf.Variable(weights['conv1_1_b'], name='w_', trainable=False)
        h_conv1_1 = tf.nn.relu(conv2d(x_image, conv1_1_W) + conv1_1_b)

        tf.summary.histogram("weight", conv1_1_W)
        tf.summary.histogram("bias", conv1_1_b)
        tf.summary.histogram("h_conv1_1", h_conv1_1)

        parameters += [conv1_1_W, conv1_1_b]

    """畳み込み層1_2"""
    with tf.name_scope("conv1_2") as scope:
        conv1_2_W = tf.Variable(weights['conv1_2_W'], name='w_', trainable=False)
        conv1_2_b = tf.Variable(weights['conv1_2_b'], name='w_', trainable=False)
        h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, conv1_2_W) + conv1_2_b)

        tf.summary.histogram("weight", conv1_2_W)
        tf.summary.histogram("bias", conv1_2_b)
        tf.summary.histogram("h_conv1_2", h_conv1_2)

        parameters += [conv1_2_W, conv1_2_b]

    """プーリング層1"""
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv1_2)
        tf.summary.histogram("h_pool1", h_pool1)

    """畳み込み層2_1"""
    with tf.name_scope("conv2_1") as scope:
        conv2_1_W = tf.Variable(weights['conv2_1_W'], name='w_', trainable=False)
        conv2_1_b = tf.Variable(weights['conv2_1_b'], name='w_', trainable=False)
        h_conv2_1 = tf.nn.relu(conv2d(h_pool1, conv2_1_W) + conv2_1_b)

        tf.summary.histogram("weight", conv2_1_W)
        tf.summary.histogram("bias", conv2_1_b)
        tf.summary.histogram("h_conv2_1", h_conv2_1)

        parameters += [conv2_1_W, conv2_1_b]

    """畳み込み層2_2"""
    with tf.name_scope("conv2_2") as scope:
        conv2_2_W = tf.Variable(weights['conv2_2_W'], name='w_', trainable=False)
        conv2_2_b = tf.Variable(weights['conv2_2_b'], name='w_', trainable=False)
        h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, conv2_2_W) + conv2_2_b)

        tf.summary.histogram("weight", conv2_2_W)
        tf.summary.histogram("bias", conv2_2_b)
        tf.summary.histogram("h_conv2_2", h_conv2_2)

        parameters += [conv2_2_W, conv2_2_b]

    """プーリング層2"""
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv2_2)
        tf.summary.histogram("h_pool2", h_pool2)

    """畳み込み層3_1"""
    with tf.name_scope("conv3_1") as scope:
        conv3_1_W = tf.Variable(weights['conv3_1_W'], name='w_', trainable=False)
        conv3_1_b = tf.Variable(weights['conv3_1_b'], name='w_', trainable=False)
        h_conv3_1 = tf.nn.relu(conv2d(h_pool2, conv3_1_W) + conv3_1_b)

        tf.summary.histogram("weight", conv3_1_W)
        tf.summary.histogram("bias", conv3_1_b)
        tf.summary.histogram("h_conv3_1", h_conv3_1)

        parameters += [conv3_1_W, conv3_1_b]

    """畳み込み層3_2"""
    with tf.name_scope("conv3_2") as scope:
        conv3_2_W = tf.Variable(weights['conv3_2_W'], name='w_', trainable=False)
        conv3_2_b = tf.Variable(weights['conv3_2_b'], name='w_', trainable=False)
        h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, conv3_2_W) + conv3_2_b)

        tf.summary.histogram("weight", conv3_2_W)
        tf.summary.histogram("bias", conv3_2_b)
        tf.summary.histogram("h_conv3_2", h_conv3_2)

        parameters += [conv3_2_W, conv3_2_b]

    """畳み込み層3_3"""
    with tf.name_scope("conv3_3") as scope:
        conv3_3_W = tf.Variable(weights['conv3_3_W'], name='w_', trainable=False)
        conv3_3_b = tf.Variable(weights['conv3_3_b'], name='w_', trainable=False)
        h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, conv3_3_W) + conv3_3_b)

        tf.summary.histogram("weight", conv3_3_W)
        tf.summary.histogram("bias", conv3_3_b)
        tf.summary.histogram("h_conv3_3", h_conv3_3)

        parameters += [conv3_3_W, conv3_3_b]

    """プーリング層3"""
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2(h_conv3_3)
        tf.summary.histogram("h_pool3", h_pool3)

    """畳み込み層4_1"""
    with tf.name_scope("conv4_1") as scope:
        conv4_1_W = tf.Variable(weights['conv4_1_W'], name='w_', trainable=False)
        conv4_1_b = tf.Variable(weights['conv4_1_b'], name='w_', trainable=False)
        h_conv4_1 = tf.nn.relu(conv2d(h_pool3, conv4_1_W) + conv4_1_b)

        tf.summary.histogram("weight", conv4_1_W)
        tf.summary.histogram("bias", conv4_1_b)
        tf.summary.histogram("h_conv4_1", h_conv4_1)

        parameters += [conv4_1_W, conv4_1_b]

    """畳み込み層4_2"""
    with tf.name_scope("conv4_2") as scope:
        conv4_2_W = tf.Variable(weights['conv4_2_W'], name='w_', trainable=False)
        conv4_2_b = tf.Variable(weights['conv4_2_b'], name='w_', trainable=False)
        h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, conv4_2_W) + conv4_2_b)

        tf.summary.histogram("weight", conv4_2_W)
        tf.summary.histogram("bias", conv4_2_b)
        tf.summary.histogram("h_conv4_2", h_conv4_2)

        parameters += [conv4_2_W, conv4_2_b]

    """畳み込み層4_3"""
    with tf.name_scope("conv4_3") as scope:
        conv4_3_W = tf.Variable(weights['conv4_3_W'], name='w_', trainable=False)
        conv4_3_b = tf.Variable(weights['conv4_3_b'], name='w_', trainable=False)
        h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, conv4_3_W) + conv4_3_b)

        tf.summary.histogram("weight", conv4_3_W)
        tf.summary.histogram("bias", conv4_3_b)
        tf.summary.histogram("h_conv4_3", h_conv4_3)

        parameters += [conv4_3_W, conv4_3_b]

    """プーリング層4"""
    with tf.name_scope("pool4") as scope:
        h_pool4 = max_pool_2x2(h_conv4_3)
        tf.summary.histogram("h_pool4", h_pool4)

    """畳み込み層5_1"""
    with tf.name_scope("conv5_1") as scope:
        conv5_1_W = tf.Variable(weights['conv5_1_W'], name='w_', trainable=False)
        conv5_1_b = tf.Variable(weights['conv5_1_b'], name='w_', trainable=False)
        h_conv5_1 = tf.nn.relu(conv2d(h_pool4, conv5_1_W) + conv5_1_b)

        tf.summary.histogram("weight", conv5_1_W)
        tf.summary.histogram("bias", conv5_1_b)
        tf.summary.histogram("h_conv5_1", h_conv5_1)

        parameters += [conv5_1_W, conv5_1_b]

    """畳み込み層5_2"""
    with tf.name_scope("conv5_2") as scope:
        conv5_2_W = tf.Variable(weights['conv5_2_W'], name='w_', trainable=False)
        conv5_2_b = tf.Variable(weights['conv5_2_b'], name='w_', trainable=False)
        h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, conv5_2_W) + conv5_2_b)

        tf.summary.histogram("weight", conv5_2_W)
        tf.summary.histogram("bias", conv5_2_b)
        tf.summary.histogram("h_conv5_2", h_conv5_2)

        parameters += [conv5_2_W, conv5_2_b]

    """畳み込み層5_3"""
    with tf.name_scope("conv5_3") as scope:
        conv5_3_W = tf.Variable(weights['conv5_3_W'], name='w_', trainable=False)
        conv5_3_b = tf.Variable(weights['conv5_3_b'], name='w_', trainable=False)
        h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, conv5_3_W) + conv5_3_b)

        tf.summary.histogram("weight", conv5_3_W)
        tf.summary.histogram("bias", conv5_3_b)
        tf.summary.histogram("h_conv5_3", h_conv5_3)

        parameters += [conv5_3_W, conv5_3_b]

    """プーリング層5"""
    with tf.name_scope("pool5") as scope:
        h_pool5 = max_pool_2x2(h_conv5_3)
        tf.summary.histogram("h_pool5", h_pool5)

    """全結合層1"""
    with tf.name_scope("fc6") as scope:
        fc6_W = tf.Variable(weights['fc6_W'], name='w_', trainable=True)
        fc6_b = tf.Variable(weights['fc6_b'], name='w_', trainable=True)

        # 形状把握
        shape = int(np.prod(h_pool5.get_shape()[1:]))

        #１次元
        h_pool5_flat = tf.reshape(h_pool5, [-1, shape])

        # 非線形変換_relu_sigmoid_tanh
        h_fc6 = tf.nn.sigmoid(tf.matmul(h_pool5_flat, fc6_W) + fc6_b)
        # h_fc6 = tf.nn.bias_add(tf.matmul(pool5_flat, fc6_W), fc6_b)

        # dropout層
        h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

        tf.summary.histogram("weight", fc6_W)
        tf.summary.histogram("bias", fc6_b)
        tf.summary.histogram("h_fc6", h_fc6)
        tf.summary.histogram("h_fc6_drop", h_fc6_drop)

        parameters += [fc6_W, fc6_b]

    """全結合層2"""
    with tf.name_scope("fc7") as scope:
        fc7_W = weight_variable([4096, 4096])
        fc7_b = bias_variable([4096])

        # 非線形変換_relu_sigmoid_tanh
        h_fc7 = tf.nn.sigmoid(tf.matmul(h_fc6_drop, fc7_W) + fc7_b)

        tf.summary.histogram("weight", fc7_W)
        tf.summary.histogram("bias", fc7_b)
        tf.summary.histogram("h_fc7", h_fc7)

        parameters += [fc7_W, fc7_b]

    """全結合層2"""
    with tf.name_scope("fc8") as scope:
        fc8_W = weight_variable([4096, num_classes])
        fc8_b = bias_variable([num_classes])
        h_fc8 = tf.matmul(h_fc7, fc8_W) + fc8_b

        tf.summary.histogram("weight", fc8_W)
        tf.summary.histogram("bias", fc8_b)
        tf.summary.histogram("h_fc8", h_fc8)

        parameters += [fc8_W, fc8_b]

    # """clipping
    # fc層の値の範囲に広がりがあるので、clippingする
    # """
    # with tf.name_scope("fc8_clipping") as scope:
    #     # cliping
    #     fc8_clipping = tf.clip_by_value(h_fc8, clip_value_min=-1000, clip_value_max=1000)
    #     tf.summary.histogram("fc8_clipping", fc8_clipping)
    
    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(h_fc8)
        tf.summary.histogram("logits", y_conv)

    return y_conv

def loss(logits, labels, delta):
    """ lossブロック
    args:
      logits: ロジットのtensor, float - [FLAGS.batch_size, FLAGS.num_classes]
      labels: ラベルのtensor, int32 - [FLAGS.batch_size, FLAGS.num_classes]

    returns:
      cross_entropy: 交差エントロピーのtensor, float

      ログオペレーション追加
    """
    with tf.name_scope('loss') as scope:
        """交差エントロピー誤差の値の計算（学習が進むにつれて小さくなるはず）"""
        # cross_entropy_sum = -tf.reduce_sum(labels * tf.log(logits + delta))
        cross_entropy_sum = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, delta, 1.0)))

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
    args:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    returns:
      train_step: 訓練のop

      最適化学習法：誤差逆伝播法を用いた勾配法（ADAM, GCD）
    """
    with tf.name_scope('training') as scope:
        #train_step = tf.train.GradientDescentOptimizer(0.01)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return train_op

def accuracy(logits, labels):
    """評価ブロック
    ・Accuracy : 正解率，(cnt(pred == target)) / batch_size
    """
    # True or Falseに変換
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # 1 or 0に変換
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # TensorBoardで表示するためにログを取得するオペレーションを定義
    tf.summary.scalar("accuracy", accuracy)
    return accuracy
