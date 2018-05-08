# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import random
import sys
import itertools
from numpy.random import *
import random
# import cv2

"""global"""
IMAGE_SIZE = 112
INPUT_SIZE = 96
DST_INPUT_SIZE = 56
NUM_CLASS = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4727

def load_data(csv, batch_size, shuffle, distored):
    """
    [csv] : filepath
    shuffle : ファイル行をランダムにとるか
    distored : 画像の水増しを行うか
    """
    """
    ●ファイルパスのリストを渡し、1行単位でデータを読み取る
        ・tf.train.string_input_producer()
            args : [csv]
            Returns : Queueのインスタンス
        ・reader.read()
            args : Queue
            Returns : A tuple of Tensors(key, value).A string scalar tensor
                    key : ファイル名の何行目かという文字列
                    value : 行のデータそのもの
    ●読み取った内容をパースし、CSVとしてデコード（Convert CSV resodes to tensors）
        ・tf.decode_csv
            args : record_defaults、代表値（それぞれのカラムの扱い方を決める）
            returns : list of Tensor objects same type as record_defaults
            Queueが処理される度に更新される.
        ・tf.stack
            配列の生成。np.asaaray([x, y, z])と同様

    ●label → one_hot, tf.float32
    ●image → [IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32
    """
    filename_queue = tf.train.string_input_producer(csv, shuffle=shuffle)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue) #識別キー、スカラ文字列
    filename, label = tf.decode_csv(value, record_defaults=[["path"], [1]])

    # labelのcastとtensorオブジェクトの作成
    label = tf.cast(label, tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASS, on_value=1.0, off_value=0.0, axis=-1)

    # imageのcastとtensorオブジェクトの作成
    jpeg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpeg, channels=3)
    image = tf.cast(image, tf.float32)

    # image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.squeeze(image, [0])
    # image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))


    # 画像の水増し
    if distored:
        cropsize = random.randint(INPUT_SIZE, INPUT_SIZE+(IMAGE_SIZE - INPUT_SIZE)/2)
        framesize = INPUT_SIZE + (cropsize - INPUT_SIZE)*2
        image = tf.image.resize_image_with_crop_or_pad(image, framesize, framesize)
        image = tf.random_crop(image, [cropsize, cropsize, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        # image = tf.image.random_hue(image, max_delta=0.04)
        # image = tf.image.random_saturation(image, lower=0.6, upper=1.4)


    # リサイズ
    # image = tf.image.resize_images(image, (DST_INPUT_SIZE, DST_INPUT_SIZE))
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [DST_INPUT_SIZE, DST_INPUT_SIZE])
    image = tf.squeeze(image, [0])

    # 正規化
    image = tf.image.per_image_standardization(image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(image, label, filename, min_queue_examples, batch_size, shuffle=shuffle)

def _generate_image_and_label_batch(image, label, filename, min_queue_examples, batch_size, shuffle):
    """batchにする
    ・tf.train.shuffle_batch
    args :
        tensors : the list of Tensors.[x, y, z]
        num_threads : dequeueするためのスレッド数
        capacity : Queue内のデータ最大数
        min_after_dequeue : dequeue後のQueueのデータ数
    Returns :  [batch_size, x, y, z]
    """
    num_preprocess_threads = 16
    capacity = min_queue_examples + 3 * batch_size
    if shuffle:
        images, label_batch, filename = tf.train.shuffle_batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, filename = tf.train.batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity)

    # tensorboard設定
    # tf.summary.image(name='image', images, max_outputs=100)

    labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])

    return images, labels, filename
