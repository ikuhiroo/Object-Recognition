import pickle
import numpy as np
import random
import sys

""" データの取得
    train_images[0].shape → (784,)
    train_labels[0].shape → (2,)
    train_images.shape[0] →　4781
    train_labels.shape[0] →　4781
    train_images.shape → (4781, 784)
    train_labels.shape → (4781, 2)
"""
with open('train_image.dump', 'rb') as f:
    train_images = pickle.load(f)
    train_images = np.array(train_images)

with open('train_label.dump', 'rb') as f:
    train_labels = pickle.load(f)
    train_labels = np.array(train_labels)

"""クラスごとに分割"""
snicker_images = train_images[3970:4109]
snicker_labels = train_labels[3970:4109]
other_images = train_images[0:3970]
other_images = np.append(other_images, train_images[4109:len(train_images)], axis=0)
other_labels = train_labels[0:3970]
other_labels = np.append(other_labels, train_labels[4109:len(train_labels)], axis=0)

print(train_labels.shape)
print(snicker_labels.shape)
print(other_labels.shape)

"""アップサンプリング
・[[][][][][]] → n * 784
・[] → (784,)
snicker_images = images[3970:4109] #[[][][][][]] → n * 784
snicker_labels = labels[3970:4109] #[[][][][][]] → n * 2

合計(4781, 2)
snicker_labels(139, 2)
other_labels(4642, 2)
"""

args = sys.argv
print('アップサンプリング数{}倍'.format(args[1]))
for _ in range(int(args[1])):
    train_images = np.append(train_images, snicker_images, axis=0)
    train_labels = np.append(train_labels, snicker_labels, axis=0)



# """ランダム性"""
# length = len(train_labels)
# length_list = range(length)
# length_list = random.shuffle(length_list)
# _train_images = np.empty((length, 784))
# _train_labels = np.empty((length, 2))
#
# for index in range(length):
#     _train_images = np.append(_train_images, train_images[index], axis=0)
#     _train_labels = np.append(_train_labels, train_labels[index], axis=0)
#
# print(_train_images.shape)
# print(_train_labels.shape)
