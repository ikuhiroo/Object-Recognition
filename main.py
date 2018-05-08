# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys

import utils
import model

from time import sleep

"""global変数"""
#定義用のflagsを作成
flags = tf.app.flags

# 値取得用のFLAGSを作成
FLAGS = flags.FLAGS

# 画像の水増し_ハイパーパラメータ
flags.DEFINE_float('lr', 1e-4, "学習率")
flags.DEFINE_integer('bs', 100, "ミニバッチ数.")
flags.DEFINE_integer('t_num', 4727, 'Number of training_samples.default=4735')
flags.DEFINE_boolean("distored", True, "distored, default=True")
flags.DEFINE_boolean("shuffle", True, "shuffle, default=True")

# ディレクトリ
flags.DEFINE_string('t_dir', './train.txt', './train.txt')
flags.DEFINE_string('model_dir', './model', '学習モデルの保存場所')
flags.DEFINE_string('log_dir', './log', 'checkpoints and summary_logsの格納場所')
flags.DEFINE_string('data_dir', './data', '画像フォルダ格納場所')

# クラス名
MEMBER_NAMES = {
    0: "snickers",
    1: "others"
}

# ネットワークの実行
def main(argv):
    graph = tf.Graph()
    with graph.as_default():
        """データ定義"""
        # dropout率
        keep_prob = tf.placeholder("float")

        """データ生成"""
        images, labels, filename = utils.load_data([FLAGS.t_dir], FLAGS.bs, shuffle=FLAGS.shuffle, distored=FLAGS.distored)

        tf.summary.image('image', tf.reshape(images, [-1, utils.DST_INPUT_SIZE, utils.DST_INPUT_SIZE, 3]), 50)

        """学習の定義"""
        # inferenceブロック
        logits = model.inference_deep(images, keep_prob, utils.DST_INPUT_SIZE, utils.NUM_CLASS)

        # lossブロック
        loss_value = model.loss(logits, labels)
        # trainingブロック
        train_op = model.training(loss_value, FLAGS.lr)
        # 評価ブロック
        acc = model.accuracy(logits, labels)

        # 保存するためのオブジェクトを生成
        saver = tf.train.Saver(max_to_keep=0)

        init_op = tf.global_variables_initializer()
        """学習の実行"""
        with tf.Session() as sess:
            # sessの初期化
            sess.run(init_op)
            print('Initialized!')

            # coodinetorの作成（複数スレッドを束ね、停止などの同期をとるクラス）
            coord = tf.train.Coordinator()
            # threadsに開始の要求
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 最大ループ回数の計算
            MAX_STEPS = int(FLAGS.t_num/FLAGS.bs)
            try:
                # ミニバッチ処理ループ
                while not coord.should_stop():
                    # TensorBoardで表示するため、全てのログをマージするオペレーション
                    summary_op = tf.summary.merge_all()
                    # Writerを呼び出して対象のグラフと出力先を指定
                    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)

                    print('distored : {}'.format(FLAGS.distored))
                    print('shuffle : {}'.format(FLAGS.shuffle))
                    print('batch_size : {}'.format(FLAGS.bs))
                    print('max_steps : {}'.format(MAX_STEPS))
                    MAX_STEPS = 1000

                    for step in range(MAX_STEPS):
                        _logits = sess.run(logits, feed_dict={keep_prob: 0.99})
                        _, loss_result, acc_res = sess.run([train_op, loss_value, acc], feed_dict={keep_prob: 0.99})

                        print('step：{}回目 Minibatch loss:{}'.format(step, loss_result))
                            # print('step：{}回目 logit:{}'.format(step, _logits))
                        # sleep(3)

                        # 100エポックごとの処理
                        # if step % 100 == 0:
                        # 100step終わるたびにTensorBoardに表示する値を追加する
                        summary_str = sess.run(summary_op, feed_dict={keep_prob: 1.0})
                        # 取得した結果をWriterで書き込む
                        summary_writer.add_summary(summary_str, step)
                        # 初期化する
                        summary_writer.flush()

                        if step % 400 == 0 or (step + 1) == MAX_STEPS or loss_result == 0:
                            # 変数データ保存
                            checkpoint_path = os.path.join(FLAGS.model_dir, 'step')
                            saver.save(sess, checkpoint_path, global_step=step)

                        if loss_result == 0:
                            print('loss is zero')
                            coord.request_stop()
                            break
                    # break
                # cnt += 1
                # coord.join(threads)
                # print('__________________queue : {}____________________'.format(cnt))
                # sleep(5)

            except tf.errors.OutOfRangeError:
                print("out of range.")

            finally:
                coord.request_stop()


if __name__=="__main__":
    tf.app.run(main)
