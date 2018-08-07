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
flags.DEFINE_integer('num_exsamples', 20, '検証用データ数')
flags.DEFINE_integer('eval_batch_size', 20, '検証用バッチサイズ')
#ディレクトリ
flags.DEFINE_string('val_file', './val.txt', '検証用データのパス')
flags.DEFINE_string('model_dir', './model', 'checkpointと学習済みモデルのdir')
flags.DEFINE_string('val_log_dir', './log_val',
                    'Directory to store checkpoints and summary logs')
flags.DEFINE_string('weight_file', './model_fine_tuning/vgg16_weights.npz',
                    'VGG16の学習済みモデルnpzファイル')

# ラベルdumpの解凍
FOLDER_PATH = os.path.dirname(os.path.abspath(__name__))
with open(os.path.join(FOLDER_PATH, 'label.pickle'), 'rb') as f:
    MEMBER_NAMES = pickle.load(f)
MEMBER_NAMES = {x[1]: x[0] for x in MEMBER_NAMES.items()}

# 学習済みモデルのロード
weights = np.load(FLAGS.weight_file)

# ネットワークの実行
def main(argv):
    graph = tf.Graph()
    with graph.as_default():
        """データ定義"""
        # Queueの準備
        images, labels, filename = utils.load_data(
            [FLAGS.val_file],
            FLAGS.eval_batch_size,
            shuffle=False,
            distored=False,
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=FLAGS.num_exsamples)
        # placeholderの定義
        keep_prob = tf.placeholder('float')

        # tensorboard設定_1
        tf.summary.image('image', tf.reshape(
            images, 
            [-1, utils.INPUT_SIZE, utils.INPUT_SIZE, 3]),
            utils.INPUT_SIZE)

        """モデルの出力（確率）"""
        logits = model.VGG(
            images, 
            keep_prob, 
            utils.INPUT_SIZE,
            utils.NUM_CLASS, 
            weights)

        """評価の定義"""
        acc = model.accuracy(logits, labels)

        """グラフの実行"""
        #saverオブジェクトの作成
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # sessの初期化
            init_op = [
                tf.global_variables_initializer(), 
                tf.local_variables_initializer()]
            sess.run(init_op)
            print('INFO : Initialized!')

            # TensorBoardで表示するため、全てのログをマージするオペレーション
            summary_op = tf.summary.merge_all()
            # Writerを呼び出して対象のグラフと出力先を指定
            summary_writer = tf.summary.FileWriter(FLAGS.val_log_dir, graph=sess.graph)

            # 学習済みモデルの復元
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir+'/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print('INFO : checkpoint file found: {}'.format(last_model))
                saver.restore(sess, last_model)
            else:
                print('INFO : No checkpoint file found')

            # coodinetorの作成とthreadsに開始の要求
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            MAX_STEPS = int(FLAGS.num_exsamples / FLAGS.eval_batch_size)
            try:
                for step in range(MAX_STEPS):
                    _acc, _filename, _logits, _labels = sess.run(
                        [acc, filename, logits, labels], feed_dict={keep_prob: 1.0})

                    # tensorboard実行
                    summary_str = sess.run(
                        summary_op, feed_dict={keep_prob: 1.0})
                    summary_writer.add_summary(summary_str)
                    summary_writer.flush()

                    # 評価
                    for batch in range(FLAGS.eval_batch_size):
                        print('step : {}'.format(step+batch))
                        print('input : {}'.format(_filename[batch]))
                        print('target : {}'.format(
                            MEMBER_NAMES[np.argmax(_labels, 1)[batch]]))
                        preds = (np.argsort(_logits[batch])[::-1])[0:5]
                        for p in preds:
                            print('{} : {}'.format(
                                MEMBER_NAMES[p], _logits[batch][p]))
                        print(' ')
                    print('Accuracy_ave : {}'.format(_acc))
            except tf.errors.OutOfRangeError:
                print('out of range.')
            finally:
                coord.request_stop()
            
            coord.join(threads)

if __name__ == "__main__":
    tf.app.run(main)
