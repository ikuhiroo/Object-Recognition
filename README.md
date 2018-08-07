# 手順書
Classification :  predict the presence/absence    

## GPUに送るファイル  

## 参考資料
・[アイドルの顔識別](https://memo.sugyan.com/entry/20160128/1453939376)
・[shuffle_batchしたときの偏り具合](http://ykicisk.hatenablog.com/entry/2016/12/18/184840)
・[csvファイルではなくtf-recodesの場合](https://www.cresco.co.jp/blog/entry/3024/)
・[VGG16 学習済みモデル](https://www.cs.toronto.edu/~frossard/post/vgg16/)
・[VGG16 関連](https://blog.heuritech.com/2016/02/29/a-brief-report-of-the-heuritech-deep-learning-meetup-5/)
## tensorflowをクローン  
`$ git clone https://github.com/ikuhiroo/Object-Recognition.git`  

## ディレクトリ構造
```
.
├── adaptive.py
├── build_train.py
├── build_train_3classes.py
├── data
│   └── thumbnails
│       ├── animal
│       ├── ankodon
│       ├── banana
│       ├── bazooka
│       ├── bird
│       ├── boat
│       ├── dog
│       ├── donburi
│       ├── drink
│       ├── erizabes
│       ├── flower
│       ├── food
│       ├── friedegg
│       ├── frog
│       ├── glasses
│       ├── humer
│       ├── ichigomilk
│       ├── insect
│       ├── lizard
│       ├── machine
│       ├── mayo
│       ├── mayodon
│       ├── natto
│       ├── noise
│       ├── otherstick
│       ├── parfait
│       ├── person
│       ├── plant
│       ├── ramen
│       ├── rice
│       ├── riceball
│       ├── seaanimal
│       ├── snickers
│       ├── sukonbu
│       ├── sunglasses
│       ├── sweets
│       ├── true
│       ├── vehicle
│       └── wood
├── eval.py
├── increase_picture.py
├── label.pickle
├── log
│   ├── events.out.tfevents.1533566615.nishiyamanoiMac.local
│   └── events.out.tfevents.1533566810.nishiyamanoiMac.local
├── main.py
├── main_without_VGG.py
├── model
│   ├── checkpoint
│   ├── step-30.data-00000-of-00001
│   ├── step-30.index
│   ├── step-30.meta
│   ├── step-3200.data-00000-of-00001
│   ├── step-3200.index
│   ├── step-3200.meta
│   ├── step-4000.data-00000-of-00001
│   ├── step-4000.index
│   └── step-4000.meta
├── model.py
├── model_fine_tuning
│   └── vgg16_weights.npz
├── snickers.txt
├── test.txt
├── test_file.textClipping
├── test_path.py
├── train.txt
├── tree.txt
├── utils.py
└── utils_without_VGG.py
```
## modelの構成
### ●学習済みモデルの取得
#### ・[VGG16](http://www.cs.toronto.edu/~frossard/post/vgg16/)から，vgg16_weights.npzを取得する．
#### ・VGG16 is a very deep network with a lot of convolution layer followed by max-pooling, reducing the dimensionality.
#### ・The model achieves 92.7% top-5 test accuracy in [ImageNet](http://image-net.org/), which is a dataset of over 14 million images belonging to 1000 classes.
### ●VGG-16モデルの構成
#### ・層の種類と大きさの確認
##### 以下より層の種類を確かめることができる
```
>>> weight_file = './model_fine_tuning/vgg16_weights.npz'
>>> weights = np.load(weight_file)
>>> keys = sorted(weights.keys())
>>> keys
['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b', 'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']
```
##### 以下よりshapeの大きさを確かめることができる
```
>>> vgg16_weights = np.load('./model_fine_tuning/vgg16_weights.npz')
>>>vgg16_weights['conv1_1_W'].shape
[3, 3, 3, 64]
```
### 変更前
```
>>> for i, k in enumerate(keys):
...             print(i, k, np.shape(weights[k]))
...
0 conv1_1_W (3, 3, 3, 64)
1 conv1_1_b (64,)
2 conv1_2_W (3, 3, 64, 64)
3 conv1_2_b (64,)
4 conv2_1_W (3, 3, 64, 128)
5 conv2_1_b (128,)
6 conv2_2_W (3, 3, 128, 128)
7 conv2_2_b (128,)
8 conv3_1_W (3, 3, 128, 256)
9 conv3_1_b (256,)
10 conv3_2_W (3, 3, 256, 256)
11 conv3_2_b (256,)
12 conv3_3_W (3, 3, 256, 256)
13 conv3_3_b (256,)
14 conv4_1_W (3, 3, 256, 512)
15 conv4_1_b (512,)
16 conv4_2_W (3, 3, 512, 512)
17 conv4_2_b (512,)
18 conv4_3_W (3, 3, 512, 512)
19 conv4_3_b (512,)
20 conv5_1_W (3, 3, 512, 512)
21 conv5_1_b (512,)
22 conv5_2_W (3, 3, 512, 512)
23 conv5_2_b (512,)
24 conv5_3_W (3, 3, 512, 512)
25 conv5_3_b (512,)
26 fc6_W (25088, 4096)
27 fc6_b (4096,)
28 fc7_W (4096, 4096)
29 fc7_b (4096,)
30 fc8_W (4096, 1000)
31 fc8_b (1000,)
>>>
```
### 変更後
```
30 fc8_W (4096, class_num)
31 fc8_b (class_num,)
```
#### 基本構成
##### ●kernel（weight_variable）
##### ±2σの切断正規分布からランダムに取り出したテンソルを生成する
##### shape : 戻り値のtensorの次元
##### mean : 生成する切断正規分布の平均値，0.0
##### stddev : 生成する切断正規分布の標準偏差，0.1
##### dtype : 戻り値のtensorのtensorの型
##### seed : 乱数固定，None
##### trainable : 学習するか否か，True
##### ●kernel（bias_variable）
##### shape : 戻り値のtensorの次元
##### bias : 定数
##### trainable : 学習するか否か，True
```
weight_variable(shape: [int, int]) -> Tensor
bias_variable(shape: [int, int]) -> Tensor
```
##### ●convolution層
##### strides : ，[1,1,1,1]
##### padding : ，SAME
```
conv2d(x: [-1, image_size, image_size, 3], W: tensor): -> Tensor
```
##### ●max pooling層
##### ksize : ，[1,2,2,1]
##### strides : ，[1,2,2,1]
##### padding : ，SAME
```
max_pool_2x2(x: [-1, image_size, image_size, 3]):-> Tensor
```
##### ●fully connected
##### ●convolution + ReLU
##### ●fully connected + ReLU
##### ●softmax

## 前処理部分
#### ●ファイルから入力としてQueueを用いてpipelineを構築する
```
1.	QueueRunerの設定
	trainの場合，num_epochs = None（ファイルを無制限）
	valの場合，num_epochs = 1, etc...
2.	Readerの設定
	skip_header_linesで飛ばして読み込める
3. (key, value)の取得
4. value(csv file)をtensorsに変換する
	record_defaultsの形式のtensorsにする
```
#### ●imageに関する前処理
##### CIFARでのコードを真似た水増し操作
##### [crop_size, frame_size]で一度リサイズ
##### 最終的にモデルに入力するサイズにリサイズする
#### ・Queueの生成
```
load_data(csv, batch_size, shuffle, distored, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN): 
-> _generate_image_and_label_batch(image, label, filename, min_after_dequeue, batch_size, shuffle)
-> [images, labels, filename]
```
#### ・Queueの開始
```
# coodinetorの作成とthreadsに開始の要求
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
```
## train
```python train.py```
### ●ハイパーパラメータ設定
#### ・`lr = 1e-4`  
##### 交差エントロピー誤差の学習率.  
#### ・`cross_entropy_min = 1e-10`  
##### 交差エントロピー誤差のclipping.  
#### ・`train_batch_size = 32`  
##### バッチサイズ
#### ・`num_exsamples = 8136`  
##### 訓練標本数.  
##### default : 600

## eval  
```$ python eval_nbatch.py```
### ●ハイパーパラメータ設定
#### ・`num_exsamples = 20`  
##### バッチサイズ
#### ・`eval_batch_size = 20`  
##### 訓練標本数.  
