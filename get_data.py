from PIL import Image
import numpy as np
import os
import pickle

"""
・入力画像: [[RGB][RGB][RGB]...[RGB]]　→　np.float32 → (784, 0)
・ラベル: one_hot → np.float64 <class 'numpy.ndarray'>
・画像のサイズresizeで圧縮 → 同じサイズにする
・np.array()にPIL.Imageを渡す
RGB画像は[行, 列, 色]のndarray、
グレー画像は[行, 列]のndarrayになる。

・ndim → 3
・shape → (28, 28, 3)
"""

URL = './thumbnails'
# フォルダ名取得
data_list = os.listdir(URL)
if '.DS_Store' in data_list:
    data_list = data_list[1:]

# np.arrayで初期化
train_images = []
train_labels = []
# 処理
for i in range(len(data_list)):
    data = os.listdir(os.path.join(URL, data_list[i]))
    if '.DS_Store' in data:
        data = data[1:]

    for j in range(len(data)):
        #絶対パス
        data_path = os.path.join(URL, data_list[i], data[j])
        print(data_path)

        # 画像データの取り込み
        img = Image.open(data_path)

        #サイズの圧縮
        img = img.resize((28,28), Image.ANTIALIAS)

        #グレー画像
        img = img.convert('L')

        #ピクセルを[0, 1]でおさめる
        arrayImg = np.asarray(img).astype(np.float32)/255.
        # 1次元
        arrayImg = arrayImg.reshape(-1)
        #print(arrayImg.ndim)
        # print(arrayImg.shape)

        #ラベル
        if data_list[i] == 'snickers':
            arr = [np.float64(1), np.float64(0)]
        else:
            arr = [np.float64(0), np.float64(1)]
        print(arr)

        # 更新
        train_images.append(arrayImg)
        train_labels.append(arr)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

# モデルの保存
with open('train_image.dump', 'wb') as f:
    pickle.dump(train_images, f)

with open('train_label.dump', 'wb') as f:
    pickle.dump(train_labels, f)
