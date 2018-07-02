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
#定義用のflagsを作成
flags = tf.app.flags
# 値取得用のFLAGSを作成
FLAGS = flags.FLAGS
flags.DEFINE_integer('numofsamples', 1, 'default=20')
flags.DEFINE_integer('batch_size', 1, 'default=1')
flags.DEFINE_string('imgpath', './imgpath', '画像パス')
flags.DEFINE_string('model_dir', './model', '学習済みモデルdir')
flags.DEFINE_string('log_dir', './log_test', 'Directory to store checkpoints and summary logs')
flags.DEFINE_string('model_VGG', './model_fine_tuning/vgg16_weights.npz', 'default=./model_fine_tuning/vgg16_weights.npz, fine_tuning用モデルの保存dir')

#ラベルdumpの解凍
FOLDER_PATH = os.path.dirname(os.path.abspath(__name__))
with open(os.path.join(FOLDER_PATH, 'label.pickle'), 'rb') as f:
    MEMBER_NAMES = pickle.load(f)
MEMBER_NAMES = {x[1]:x[0] for x in MEMBER_NAMES.items()}

# 学習済みモデルのロード（初期化）
weights = np.load(FLAGS.model_VGG)

def main(argv):
    graph = tf.Graph()
    with graph.as_default():
        """データ定義"""
        imgpath = FLAGS.imgpath
        jpeg = tf.read_file(imgpath)
        image = tf.image.decode_jpeg(jpeg, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [utils.DST_INPUT_SIZE, utils.DST_INPUT_SIZE])
        image = tf.squeeze(image, [0])
        image = tf.image.per_image_standardization(image)

        """学習の定義"""
        logits = model.inference_deep_VGG(image, 1.0, utils.DST_INPUT_SIZE, utils.NUM_CLASS, weights)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            """モデルの復元"""
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            print('Initialized!')

            print("'''''''''''''''''''''''''''''''''''''''''''''''''")
            print('class : {}'.format(utils.NUM_CLASS))
            print('imagepath : {}'.format(FLAGS.imgpath))
            print('IMAGE_SIZE : {}'.format(utils.IMAGE_SIZE))
            print('INPUT_SIZE : {}'.format(utils.INPUT_SIZE))
            print('DST_INPUT_SIZE : {}'.format(utils.DST_INPUT_SIZE))
            print('numofsamples : {}'.format(FLAGS.numofsamples))
            print('distored : false')
            print('shuffle : false')
            print('batch_size : {}'.format(FLAGS.batch_size))
            print("'''''''''''''''''''''''''''''''''''''''''''''''''")

            # ckptオブジェクト生成
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir+'/')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print('checkpoint file found: {}'.format(last_model))
                saver.restore(sess, last_model)
            else:
                print('No checkpoint file found')

            # モデル出力値
            # softmax = logits.eval()
            softmax = sess.run(logits)
            result = softmax[0]
            pred = np.argmax(result)

            # 評価
            for i in range(3,0,-1):
                print('{}({}): {}'.format(MEMBER_NAMES[np.argsort(result)[-i]],i , np.sort(result)[-i]))
            print('分類結果 : {}'.format(MEMBER_NAMES[pred]))

if __name__=="__main__":
    tf.app.run(main)
