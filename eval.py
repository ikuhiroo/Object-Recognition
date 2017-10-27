#coding:utf-8

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys


# 定義用のflagsを作成
flags = tf.app.flags
# 値取得用のFLAGSを作成
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, "学習率")
flags.DEFINE_float('lamb', 5e-4, "正則化項のペナルティ度")
flags.DEFINE_integer('num_classes', 2, "分類クラス数")
flags.DEFINE_integer('batch_size', 50, "ミニバッチ数.")
flags.DEFINE_integer('num_epochs', 10, "エポック数")
flags.DEFINE_integer('max_steps', 5, 'Number of steps to run trainer.')
flags.DEFINE_string('model_dir', './model', '学習モデルの保存場所')
flags.DEFINE_string('log_dir', './data', 'Directory to store checkpoints and summary logs')
flags.DEFINE_string('result_dir', './result', '評価の結果の保存場所')


# with open('train_image.dump', 'rb') as f:
#     test_images = pickle.load(f)
#     test_images = np.array(test_images)
#
# with open('train_label.dump', 'rb') as f:
#     test_labels = pickle.load(f)
#     test_labels = np.array(test_labels)
#
#
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

# 前処理のインスタンス作成
result = preprocess.Preprocess()
result.input()

test_images = result._test_images
test_labels = result._test_labels


""" inference（学習モデルの生成）
Args:
  images_placeholder: モデルへの入力（訓練標本）
  keep_prob: モデルへの入力（dropout率）

Returns:
  y_conv: 各クラスの確率
"""
def interence(images_placeholder, keep_prob):
    """空クラスを生成"""
    class Parameter_set(object):
        pass

    save_parameter = Parameter_set()#インスタンス作成

    """　x_image
    [-1(flatten), height, width、チャネル数]
    images_placeholderがまず1D(784次元)に変換され、28*28に変換される
    """
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 1])

    def weight_variable(shape):
        """ 重みパラメータの初期化
        Args:
          shape: [バッチサイズ, バッチサイズ, 入力チャネル数、出力のチャネル数]

        Returns:
          重みパラメータの初期値はランダム → 事前学習ないNN
          mean: 平均（default=0.0）
          stddev：標準偏差（default=1.0）
          tf.truncated_normalは標準偏差２倍以上の値をランダムに取ってくる
        """
        inital = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(inital, name='w')

    def bias_variable(shape):
        """ 重みのバイアスの初期化
        Args:
            shape:[出力チャネル数]
        Returns:
            バイアスパラメータの初期化は固定
        """
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital, name='b')

    def conv2d(x, W):
        """ 畳み込み層
        Args:
        Returns:[バッチサイズ、高さ、横幅、チャンネル数]
          strides: [1, dy, dx, 1]
            縦方向にdyピクセル毎、横方向にdxピクセル毎にフィルタを適用する
          padding: 画像領域が足りない時の処理
            SAMEは自動的に元画像に０を足す
        """
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", name='output')

    def max_pool_2x2(x):
        """ Maxプーリング層（入力値の2*2の範囲に置ける最大値をプールする）
        Args:
          x:学習するデータなし

        Returns:
          ksize: [1,dy,dx,1]
          strides: [1, dy, dx, 1]
            縦方向にdyピクセル毎、横方向にdxピクセル毎にフィルタを適用する
          padding: 画像領域が足りない時の処理
            SAMEは自動的に元画像に０を足す
        """
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    """処理ごとnamespaceを切る"""
    # 畳み込み層1の作成（最初のレイヤ）
    with tf.name_scope("conv1") as scope:
        W_conv1 = weight_variable([3,3,1,16])
        b_conv1 = bias_variable([16])
        tf.summary.histogram("weight", W_conv1)
        tf.summary.histogram("bias", W_conv1)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 畳み込み層2の作成
    with tf.name_scope("conv2") as scope:
        W_conv2 = weight_variable([3,3,16,16])
        b_conv2 = bias_variable([16])
        tf.summary.histogram("weight", W_conv2)
        tf.summary.histogram("bias", W_conv2)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # プーリング層1の作成
    with tf.name_scope("pool1") as scope:
        h_pool1 = max_pool_2x2(h_conv2)

    # 畳み込み層3の作成
    with tf.name_scope("conv3") as scope:
        W_conv3 = weight_variable([3,3,16,32])
        b_conv3 = bias_variable([32])
        tf.summary.histogram("weight", W_conv3)
        tf.summary.histogram("bias", W_conv3)
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # 畳み込み層4の作成
    with tf.name_scope("conv4") as scope:
        W_conv4 = weight_variable([3,3,32,32])
        b_conv4 = bias_variable([32])
        tf.summary.histogram("weight", W_conv4)
        tf.summary.histogram("bias", W_conv4)
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # プーリング層2の作成
    with tf.name_scope("pool2") as scope:
        h_pool2 = max_pool_2x2(h_conv4)

    # 畳み込み層5の作成
    with tf.name_scope("conv5") as scope:
        W_conv5 = weight_variable([3,3,32,64])
        b_conv5 = bias_variable([64])
        tf.summary.histogram("weight", W_conv5)
        tf.summary.histogram("bias", W_conv5)
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # 畳み込み層6の作成
    with tf.name_scope("conv6") as scope:
        W_conv6 = weight_variable([3,3,64,64])
        b_conv6 = bias_variable([64])
        tf.summary.histogram("weight", W_conv6)
        tf.summary.histogram("bias", W_conv6)
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    # プーリング層3の作成
    with tf.name_scope("pool3") as scope:
        h_pool3 = max_pool_2x2(h_conv6)

    # 結合層1の作成
    with tf.name_scope("fc1") as scope:
        save_parameter.W_fc1 = weight_variable([4*4*64, 1024])
        save_parameter.b_fc1 = bias_variable([1024])
        tf.summary.histogram("weight", save_parameter.W_fc1)
        tf.summary.histogram("bias", save_parameter.b_fc1)

        h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, save_parameter.W_fc1) + save_parameter.b_fc1)
        # dropout1の設定
        # 擬似的に平均化してくれる
        h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 結合層2の作成
    with tf.name_scope("fc2") as scope:
        save_parameter.W_fc2 = weight_variable([1024, FLAGS.num_classes])
        save_parameter.b_fc2 = bias_variable([FLAGS.num_classes])
        tf.summary.histogram("weight", save_parameter.W_fc2)
        tf.summary.histogram("bias", save_parameter.b_fc2)

    # ソフトマックス関数による正規化
    with tf.name_scope("softmax") as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, save_parameter.W_fc2) + save_parameter.b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv, save_parameter

# グラフ実行
def main(argv):
    with tf.Graph().as_default():
        """データ定義"""
        # 訓練データのplaceholder
        x_image = tf.placeholder("float", shape=[None, 784])
        y_label = tf.placeholder("float", shape=[None, 2])
        # パラメータ
        W = tf.Variable(tf.zeros([784,2]))
        b = tf.Variable(tf.zeros([2]))
        # dropout率
        # 中間層のニューロンをランダムに削除する
        keep_prob = tf.placeholder("float")

        """学習ブロック"""
        logits, a = interence(x_image, keep_prob)

        """評価ブロック"""
        classificate = tf.equal(tf.argmax(logits, 1), tf.arg_max(y_label, 1))

        """モデルの復元"""
        #saverオブジェクトの作成
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 変数の初期化
            sess.run(tf.global_variables_initializer())
            # ckptオブジェクト生成
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir+'/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                saver.restore(sess, last_model)
            else:
                print('No checkpoint file found')
                # return

            # 評価ブロック
            # 順伝搬の計算
            _logits_snickers = sess.run(logits, feed_dict={x_image:snicker_images, keep_prob:1.0})
            _logits_others = sess.run(logits, feed_dict={x_image:other_images, keep_prob:1.0})

            _classificate_snickers = sess.run(classificate, feed_dict={logits:_logits_snickers, y_label:snicker_labels})
            _classificate_others = sess.run(classificate, feed_dict={logits:_logits_others, y_label:other_labels})

            # args = sys.argv
            # if str(args[1]) == 'snickers':
            for index in range(len(_logits_snickers)):
                print('snickers_image   logit：{} 分類先:{}'.format(_logits_snickers[index], _classificate_snickers[index]))
            # else:
            for index in range(len(_logits_others)):
                print('other_image   logit：{} 分類先:{}'.format(_logits_others[index], _classificate_others[index]))

            # logitの平均
            print('（正解）snickers:{}（モデル出力）snickers:{}'.format(len(snicker_labels), len(np.where(_logits_snickers==True)[0])))
            print('（正解）snickers以外:{}（モデル出力）snickers以外：{}'.format(len(other_labels), len(np.where(_logits_snickers==True)[0])))


            # # ファイル出力
            # if os.path.isdir(os.path.join(FLAGS.result_dir, 'snickers')):
            #     pass
            # else:
            #     os.mkdir(os.path.join(FLAGS.result_dir, 'snickers'))
            #
            # if os.path.isdir(os.path.join(FLAGS.result_dir, 'others')):
            #     pass
            # else:
            #     os.mkdir(os.path.join(FLAGS.result_dir, 'others'))
            # with open(os.path.join(FLAGS.result_dir, 'snickers.txt') , 'w') as f:
            #     for index in range(len(_logits_snickers)):
            #         f.write('snickers_image   logit：{} 分類先:{}'.format(_logits_snickers[index], _classificate_snickers[index]))
            #
            # with open(os.path.join(FLAGS.result_dir, 'others.txt'), 'w') as f:
            #     for index in range(len(_logits_others)):
            #         f.write('other_image   logit：{} 分類先:{}'.format(_logits_others[index], _classificate_others[index]))

if __name__=="__main__":
    tf.app.run()
