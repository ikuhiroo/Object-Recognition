#coding:utf-8

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys
import preprocess

#定義用のflagsを作成
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

result = preprocess.input()

# 訓練データ
train_snickers_images = result.train_snickers_images
train_snickers_labels = result.train_snickers_labels
train_other_images = result.train_other_images
train_other_labels = result.train_other_labels

train_images = result.train_images
train_labels = result.train_labels

# テストデータ
test_snickers_images = result.test_snickers_images
test_snickers_labels = result.test_snickers_labels
test_other_images = result.test_other_images
test_other_labels = result.test_other_labels

test_images = result.test_images
test_labels = result.test_labels


"""アップサンプリング"""
args = sys.argv
print('アップサンプリング数{}倍'.format(args[1]))
for _ in range(int(args[1])):
    train_images = np.append(train_images, train_snickers_images, axis=0)
    train_labels = np.append(train_labels, train_snickers_labels, axis=0)

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

def loss(logits, labels, save_parameter):
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
        cross_entropy_sum = -tf.reduce_sum(labels * tf.log(logits))
        cross_entropy_mean = -tf.reduce_mean(labels * tf.log(logits))
        cross_entropy_mean_L2 = cross_entropy_mean

        """正則化項、重み減衰を加えた損失（平均）"""
        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(save_parameter.W_fc1) + tf.nn.l2_loss(save_parameter.b_fc1) + tf.nn.l2_loss(save_parameter.W_fc2) + tf.nn.l2_loss(save_parameter.b_fc2))
        # Add the regularization term to the loss.
        cross_entropy_mean_L2 = cross_entropy_mean_L2 + (FLAGS.lamb * regularizers)

        """TensorBoardで表示するためにログを取得するオペレーションを定義"""
        tf.summary.histogram("cross_entropy_sum", cross_entropy_sum)
        tf.summary.histogram("cross_entropy_mean", cross_entropy_mean)
        tf.summary.histogram("cross_entropy_mean_L2", cross_entropy_mean_L2)
        tf.summary.scalar("cross_entropy_sum", cross_entropy_sum)
        tf.summary.scalar("cross_entropy_mean", cross_entropy_mean)
        tf.summary.scalar("cross_entropy_mean_L2", cross_entropy_mean_L2)

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
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return train_step


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

        """学習率のスケジュール"""
        #num_batches_per_epoch =  FLAGS.train_size / FLAGS.batch_size
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        #batch = tf.Variable(0, "float")#, dtype=tf.float32
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        # learning_rate = tf.train.exponential_decay(
        #     0.01,                         # 初期値
        #     batch * FLAGS.batch_size,     # Current index into the dataset.
        #     FLAGS.train_size,             # Decay step.
        #     0.95,                         # Decay rate.
        #     staircase=True)
        #tf.summary.scalar("learning_rate", FLAGS.learning_rate)

        """学習の定義"""
        # inferenceブロック
        logits, save_parameter = interence(x_image, keep_prob)
        # lossブロック
        loss_value = loss(logits, y_label, save_parameter)
        # trainingブロック
        train_op = training(loss_value, FLAGS.learning_rate)
        # 評価ブロック
        accur = accuracy(logits, y_label)

        """学習の実行"""
        with tf.Session() as sess:

            #変数初期化の実行
            sess.run(tf.global_variables_initializer())
            print('Initialized!')

            # 保存するためのオブジェクトを生成
            saver = tf.train.Saver()

            # TensorBoardで表示するため、全てのログをマージするオペレーション
            summary_op = tf.summary.merge_all()
            # Writerを呼び出して対象のグラフと出力先を指定
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)



            """バッチ取得
            int(NUM_EPOCHS * TRAIN_SIZE) →　学習で使う標本数
            BATCH_SIZE → １回の更新で使う標本数
            int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE →　ループ回数（int型）
            """
            # 訓練の実行
            TRAIN_SIZE = train_images.shape[0] #訓練標本数
            BATCH_SIZE = FLAGS.batch_size #１回の更新で使う標本数
            NUM_EPOCHS = FLAGS.num_epochs #全ての訓練データを使って学習して１回

            for step in range(int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
                #実行数とstep数を合わせる
                #step = step + 1

                offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
                batch_images = train_images[offset:(offset + BATCH_SIZE)]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

                _loss_value = sess.run(loss_value, feed_dict={x_image: batch_images, y_label: batch_labels, keep_prob:0.5})

                print('step：{}回目 Minibatch loss:{}'.format(step, _loss_value))

                # 学習
                sess.run(train_op, feed_dict={x_image: batch_images, y_label: batch_labels, keep_prob:0.5})
                # 評価
                _accury = sess.run(accur, feed_dict={x_image: batch_images, y_label: batch_labels, keep_prob: 1.0})

                # 100エポックごとの処理
                if step % 100 == 0:
                    # 認識率
                    print('エポック数：{}回目 正解率:{}'.format(step, _accury))

                    model_dir = os.path.join(FLAGS.model_dir, 'step')
                    # 変数データ保存
                    saver.save(sess, model_dir, global_step=step)

                # 1step終わるたびにTensorBoardに表示する値を追加する
                #ログ取得の実行
                _summary_op = sess.run(summary_op, feed_dict={x_image: batch_images, y_label: batch_labels, keep_prob: 1.0})
                #取得した結果をWriterで書き込み
                summary_writer.add_summary(_summary_op, step)
                #初期化する
                summary_writer.flush()

            #結果表示
            print('訓練データの最終認識率：{}'.format(_accury))

if __name__=="__main__":
    tf.app.run()
