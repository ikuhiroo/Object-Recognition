'''
train/val/testファイルの作成

ファイル形式 : [絶対パス\sクラスラベル\n]
出力先 :
'''
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys
import itertools
from numpy.random import *
import random
import re
from pathlib import Path

#定義用のflagsを作成
flags = tf.app.flags
# 値取得用のFLAGSを作成
FLAGS = flags.FLAGS

# @params
flags.DEFINE_integer('s_cnt', 10, "画像の水増し")

# @static
CURRENT_DIR = Path('./').resolve()
WORK_DIR = CURRENT_DIR
IMAGE_DIR = CURRENT_DIR / Path('data/thumbnails')
OUTPUT_DIR = CURRENT_DIR
trainval_txt = OUTPUT_DIR / 'trainval.txt'
train_txt = OUTPUT_DIR / 'train.txt'
val_txt = OUTPUT_DIR / 'val.txt'

# result_list
trainval = []
train = []
val = []
split = 0.9

def main(args):
    # クラス名取得
    class_name = [x for x in list(IMAGE_DIR.iterdir()) if x.is_dir() == True]
    class_dict = {}
    label = 0
    for i in range(len(class_name)):
        class_dict[class_name[i].name] = label
        label+=1

    # trainvalに追加
    for label in class_dict.keys():
        try:
            class_name = list(IMAGE_DIR.glob(label))[0]
            fname = list(class_name.iterdir())
            for i in range(len(fname)):
                trainval.append('{} {}'.format(fname[i], class_dict[label]))
        except:
            pass

    # シャッフル
    random.shuffle(trainval)

    # 分割
    train = trainval[0:int(len(trainval)*split)]
    val = trainval[int(len(trainval)*split):-1]
    print(len(trainval))
    print(len(train))
    print(len(val))

    # 書き込み
    with trainval_txt.open(mode='w', encoding="utf-8") as f:
        f.write("\n".join(val))
    with train_txt.open(mode='w', encoding="utf-8") as f:
        f.write("\n".join(val))
    with val_txt.open(mode='w', encoding="utf-8") as f:
        f.write("\n".join(val))

if __name__=="__main__":
    tf.app.run(main)
