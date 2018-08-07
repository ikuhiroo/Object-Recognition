# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys
from time import sleep

import utils
import model

"""global変数"""
#定義用のflagsを作成
flags = tf.app.flags

# 値取得用のFLAGSを作成
FLAGS = flags.FLAGS

# ハイパーパラメータ設定
flags.DEFINE_float('lr', 1e-4, "default=1e-4, 交差エントロピー誤差の学習率")
flags.DEFINE_float('cross_entropy_min', 1e-10, "交差エントロピー誤差のclipping")
flags.DEFINE_integer('train_batch_size', 32, "バッチサイズ")
flags.DEFINE_integer('num_exsamples', 8136, 'build_train.pyにおける訓練標本数')

# ディレクトリ
flags.DEFINE_string('train_file', './train.txt', '訓練用データのパス')
flags.DEFINE_string('model_dir', './model', 'checkpointと学習済みモデルのdir')
flags.DEFINE_string(
    'weight_file', './model_fine_tuning/vgg16_weights.npz', 'VGG16の学習済みモデルnpzファイル')
flags.DEFINE_string('train_log_dir', './log_train',
                    'Directory to store checkpoints and summary logs')
flags.DEFINE_string('data_dir', './data', 'default=./data, 画像フォルダ格納場所')

# ラベルdumpの解凍
FOLDER_PATH = os.path.dirname(os.path.abspath(__name__))
with open(os.path.join(FOLDER_PATH, 'label.pickle'), 'rb') as f:
    MEMBER_NAMES = pickle.load(f)
MEMBER_NAMES = {x[1]: x[0] for x in MEMBER_NAMES.items()}

# 学習済みモデルのパラメータ
weights = np.load(FLAGS.weight_file)

# ネットワークの実行
def main(argv):
    graph = tf.Graph()
    with graph.as_default():
        """データ定義"""
        # Queueの準備
        images, labels, filename = utils.load_data(
            [FLAGS.train_file], 
            FLAGS.train_batch_size, 
            shuffle=True, 
            distored=True, 
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=FLAGS.num_exsamples)
        # placeholderの定義
        keep_prob = tf.placeholder("float")

        # tensorboard設定_1
        tf.summary.image('image', tf.reshape(
            images, 
            [-1, utils.INPUT_SIZE, utils.INPUT_SIZE, 3]),
            utils.INPUT_SIZE)

        """学習の定義"""
        # inferenceブロック
        logits = model.VGG(
            images, 
            keep_prob,
            utils.INPUT_SIZE,
            utils.NUM_CLASS, 
            weights)

        # lossブロック
        loss_value = model.loss(logits, labels, FLAGS.cross_entropy_min)
        # trainingブロック
        train_op = model.training(loss_value, FLAGS.lr)
        # 評価ブロック
        acc = model.accuracy(logits, labels)

        # 保存するためのオブジェクトを生成
        saver = tf.train.Saver(max_to_keep=10)

        with tf.Session() as sess:
            init_op = [
                tf.global_variables_initializer(),
                tf.local_variables_initializer()]
            sess.run(init_op)
            print('INFO : Initialized!')

            # TensorBoardで表示するため、全てのログをマージするオペレーション
            summary_op = tf.summary.merge_all()
            # Writerを呼び出して対象のグラフと出力先を指定
            summary_writer = tf.summary.FileWriter(
                FLAGS.train_log_dir, graph=sess.graph)

            # 学習済みモデルの復元
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir+'/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print('INFO : checkpoint file found: {}'.format(last_model))
                saver.restore(sess, last_model)
            else:
                print('INFO : No checkpoint file found')

            # coodinetorの作成と開始の要求
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 最大ループ回数の計算
            MAX_STEPS = int(FLAGS.num_exsamples/FLAGS.train_batch_size)

            """学習の実行"""
            print('INFO : class : {}'.format(utils.NUM_CLASS))
            print('INFO : lr : {}'.format(FLAGS.lr))
            print('INFO : batch_size : {}'.format(FLAGS.train_batch_size))
            print('INFO : MAX_STEPS : {}'.format(MAX_STEPS))
            print(' ')
            try:
                for step in range(MAX_STEPS*5):
                    print('step：{}回目'.format(step))
                    # 更新
                    _labels, _train_op, _logits, _loss_value, _acc = sess.run(
                        [labels, train_op, logits, loss_value, acc], 
                        feed_dict={keep_prob: 0.99})

                    print('INFO : loss: {}'.format(_loss_value))
                    print("INFO : Accuracy_ave: {}".format(_acc*100))
                    print(' ')

                    # Tensorboard出力設定
                    # 100step終わるたびにTensorBoardに表示する値を追加する
                    summary_str = sess.run(summary_op, feed_dict={keep_prob: 1.0})
                    # 取得した結果をWriterで書き込む
                    summary_writer.add_summary(summary_str, step)
                    # 初期化する
                    summary_writer.flush()

                    if step % 100 == 0 or (step + 1) == MAX_STEPS or _loss_value == 0:
                        if step != 0:
                            # 変数データ保存
                            checkpoint_path = os.path.join(FLAGS.model_dir, 'step')
                            saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print("out of range.")
            finally:
                coord.request_stop()
            
            coord.join(threads)

if __name__=="__main__":
    tf.app.run(main)
