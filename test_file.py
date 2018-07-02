# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys
import pickle
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import utils
import model

"""global変数"""
flags = tf.app.flags
FLAGS = flags.FLAGS
#ハイパーパラメータ
flags.DEFINE_integer('numofsamples', 20, 'default=20, テストデータ数')
flags.DEFINE_integer('batchsize', 1, "default=1, バッチサイズ")
#ディレクトリ
flags.DEFINE_string('test', './test.txt', 'default=./test.txt, データ保存dir')
flags.DEFINE_string('model_dir', './model', 'default=./model, 学習済モデルdir')
flags.DEFINE_string('log_dir', './log_test', 'default=./log_test, Directory to store checkpoints and summary logs')
flags.DEFINE_string('model_VGG', './model_fine_tuning/vgg16_weights.npz', 'default=./model_fine_tuning/vgg16_weights.npz, fine_tuning用モデルの保存dir')

#ラベルdumpの解凍
FOLDER_PATH = os.path.dirname(os.path.abspath(__name__))
with open(os.path.join(FOLDER_PATH, 'label.pickle'), 'rb') as f:
    MEMBER_NAMES = pickle.load(f)
MEMBER_NAMES = {x[1]:x[0] for x in MEMBER_NAMES.items()}

# 学習済みモデルのロード
weights = np.load(FLAGS.model_VGG)

# ネットワークの実行
def main(argv):
    graph = tf.Graph()
    with graph.as_default():
        """データ定義"""
        # 訓練データのplaceholder
        images, labels, filename = utils.load_data([FLAGS.test], FLAGS.batchsize , shuffle=False, distored=False, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=FLAGS.numofsamples)
        keep_prob = tf.placeholder("float")

        # tensorboard設定
        tf.summary.image('image', tf.reshape(images, [-1, utils.DST_INPUT_SIZE, utils.DST_INPUT_SIZE, 3]), utils.DST_INPUT_SIZE)

        """学習の定義"""
        logits = model.inference_deep_VGG(images, keep_prob, utils.DST_INPUT_SIZE, utils.NUM_CLASS, weights)
        acc = model.accuracy(logits, labels)

        """モデルの復元"""
        #saverオブジェクトの作成
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # sessの初期化
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print('Initialized!')

            # TensorBoardで表示するため、全てのログをマージするオペレーション
            summary_op = tf.summary.merge_all()
            # Writerを呼び出して対象のグラフと出力先を指定
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)

            # 学習済みモデルの復元
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir+'/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print('checkpoint file found: {}'.format(last_model))
                saver.restore(sess, last_model)
            else:
                print('No checkpoint file found')

            # coodinetorの作成とthreadsに開始の要求
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                # ミニバッチ処理ループ
                while not coord.should_stop():
                    print("'''''''''''''''''''''''''''''''''''''''''''''''''")
                    print('class : {}'.format(utils.NUM_CLASS))
                    print('IMAGE_SIZE : {}'.format(utils.IMAGE_SIZE))
                    print('INPUT_SIZE : {}'.format(utils.INPUT_SIZE))
                    print('DST_INPUT_SIZE : {}'.format(utils.DST_INPUT_SIZE))
                    print('numofsamples : {}'.format(FLAGS.numofsamples))
                    print('distored : false')
                    print('shuffle : false')
                    print('batchsize : {}'.format(FLAGS.batchsize))
                    print("'''''''''''''''''''''''''''''''''''''''''''''''''")
                    # エポックを回す
                    for step in range(FLAGS.numofsamples):
                        _acc, _filename, _logits, _labels = sess.run([acc, filename, logits, labels], feed_dict={keep_prob: 1.0})

                        # tensorboard実行
                        summary_str = sess.run(summary_op, feed_dict={keep_prob: 1.0})
                        summary_writer.add_summary(summary_str)
                        summary_writer.flush()

                        # 評価
                        y_true = np.argmax(_labels, 1)
                        y_pred = np.argmax(_logits, 1)
                        print(_filename[0])

                        for i in range(3,0,-1):
                            print('{}({}): {}'.format(MEMBER_NAMES[np.argsort(_logits)[0][-i]],i , np.sort(_logits)[0][-i]))
                        print("{} → {}".format(MEMBER_NAMES[y_true[0]], MEMBER_NAMES[y_pred[0]]))
                        print('\n')

                        # accuracyの計算
                        True_cnt = 0
                        if y_true[0] == y_pred[0]:
                            True_cnt+=1
                    print('accuracy : {}'.format((True_cnt/FLAGS.numofsamples)*100))
                    break
            except tf.errors.OutOfRangeError:
                print("out of range.")

            finally:
                coord.request_stop()

if __name__=="__main__":
    tf.app.run(main)
