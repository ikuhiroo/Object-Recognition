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

"""グローバル変数"""
IMAGE_SIZE = 270
INPUT_SIZE = 224
NUM_CLASS = 39 #クラス数
# cropsize = 242
# framesize = 260
# 画像は一度214~260のサイズの画像で水増し
# モデルに入力するサイズ = 224


def load_data(csv, batch_size, shuffle, distored, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
    """
    ファイルからの入力としてQueueを用いてpipelineを構築する（大規模データの入力方法）
    """
    # Set a QueueRuner
    # train -> num_epochs = None
    # val -> num_epochs = 1, etc...
    filename_queue = tf.train.string_input_producer(
        string_tensor = csv, 
        shuffle=shuffle, 
        seed=None, 
        num_epochs=None)
    # Set a Reader
    reader = tf.TextLineReader(skip_header_lines=None)
    # Get a tuple (key, value)
    key, value = reader.read(queue=filename_queue)
    # Convert CSV records to tensors like record_defaults
    filename, label = tf.decode_csv(
        records=value, 
        record_defaults=[['./data/thumbnails/noise/noise10.png'], [1]], 
        field_delim=',')

    # label -> onehot vector
    label = tf.cast(label, tf.int64)
    label = tf.one_hot(
        label, 
        depth=NUM_CLASS, 
        on_value=1.0, 
        off_value=0.0, 
        axis=-1)

    # image -> [IMAGE_SIZE, IMAGE_SIZE]
    jpeg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpeg, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.squeeze(image, [0])

    # 画像の水増し
    # ある程度のスケーリング誤差も吸収するために[crop_size, frame_size]にリサイズ
    # 水増し操作で最終的に[input_size, input_size]にリサイズする
    if distored:
        # 242-260の間で
        cropsize = random.randint(INPUT_SIZE, INPUT_SIZE+(IMAGE_SIZE - INPUT_SIZE)/2)
        framesize = INPUT_SIZE + (cropsize - INPUT_SIZE)*2

        image = tf.image.resize_image_with_crop_or_pad(image, framesize, framesize)
        image = tf.random_crop(image, [cropsize, cropsize, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.random_hue(image, max_delta=0.04)
        image = tf.image.random_saturation(image, lower=0.6, upper=1.4)

    # image -> [INPUT_SIZE, INPUT_SIZE]
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [INPUT_SIZE, INPUT_SIZE])
    image = tf.squeeze(image, [0])

    # imageを正規化
    image = tf.image.per_image_standardization(image)

    # Setting min_after_dequeue
    min_fraction_of_examples_in_queue = 0.4
    min_after_dequeue = int(
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(image, label, filename, min_after_dequeue, batch_size, shuffle=shuffle)


def _generate_image_and_label_batch(image, label, filename, min_after_dequeue, batch_size, shuffle):
    """
    Creates batches by randomly shuffling tensors.
    Args:
        min_after_dequeue: Minimum number elements in the queue after a dequeue
        , used to ensure a level of mixing of elements.  
        seed: Seed for the random shuffling within the queue.  
        enqueue_many: Whether each tensor in `tensor_list` is a single example.  

    capacity > batch_size
    """
    # setting num_preprocess_threads
    # The number of threads enqueuing `tensor_list`.
    num_preprocess_threads = 1
    # setting capacity
    # The maximum number of elements in the queue.
    capacity = min_after_dequeue + 3 * batch_size

    # shuffle_batch(train) or batch(val)
    if shuffle:
        images, label_batch, filename = tf.train.shuffle_batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            seed=None, 
            enqueue_many=False,
            allow_smaller_final_batch=True)
    else:
        # allow_smaller_final_batch: (Optional) Boolean. 
        # If `True`, 
        # allow the final batch to be smaller if there are insufficient items left in the queue.
        images, label_batch, filename = tf.train.batch(
            tensors=[image, label, filename],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=capacity,
            allow_smaller_final_batch=True)

    print('INFO : batch_size:{}'.format(batch_size))
    print('INFO : min_after_dequeue:{}'.format(min_after_dequeue))
    print('INFO : capacity:{}'.format(capacity))
    
    # labels -> [batch_size, NUM_CLASS]
    labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])

    return images, labels, filename
