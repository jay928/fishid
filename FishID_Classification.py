from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트
import tensorflow as tf
import tensorflow_hub as hub
import os
import glob
import PIL.Image as Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
import random

# 레이블
img_dir = "..\\img"
dir_list = os.listdir(img_dir)
fishName_list = []
label = []
cnt = 1
for dir in dir_list :
    fishName_list.append(dir)
    cnt += 1
print(fishName_list)
print("bicolor parrotfish`s index = " + str(fishName_list.index("bicolor parrotfish")))
print("bluegreen chromis`s index = " + str(fishName_list.index("bluegreen chromis")))
print("convict tang`s index = " + str(fishName_list.index("convict tang")))

# Feature Vector
featureVector_url  = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3" #@param {type:"string"}
featureVector_module = hub.Module(featureVector_url, tags=[])

# featureVector에서 요구하는 사이즈 도출
height, width = hub.get_expected_image_size(featureVector_module)
image_shape = (height, width)
print(image_shape)

# 모델 첫번째에 사전 훈련된 레이어추가를 위한 선언
hub_layer = hub.KerasLayer(featureVector_module)

model = tf.keras.Sequential()

# 첫번째층
# 텐서플로우 허브층
# 사전 훈련된 모델을 사용하여 하나의 Feature Vector로 이미지를 매핑한다.
model.add(hub_layer)

# 허브층에서 출력된 벡터는 512개의 은닉유닛을 가진 Dense층 -> 128개의 은닉유닛을 가진 Dense층 -> 32개의 은닉유닛을 가진 Dense층으로 주입됨.
# model.add(tf.keras.layers.Dense(1024, input_shape=(image_shape,3), activation='relu'))
model.add(tf.keras.layers.Dense(512, input_shape=(image_shape,3), activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='softmax'))
#model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 데이터 로드
image_batch = np.empty((0, height, width, 3))
data_all = []
for fish in fishName_list:  # 물고기(레이블) 선택
    print("fish name = " + fish)
    for fishimg in glob.glob(img_dir+"\\"+fish+"\\*.jpg"): # 전체 파일 read
        try:
            img = Image.open(fishimg)
        except:
            print('Exception occurred during opening this file.', fishimg)
            continue

        if img.mode != 'RGB':
            print(img.mode)
            continue

        img_w, img_h = img.size

        desired_size = max(img_w, img_h)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(img, ((desired_size - img_w) // 2, (desired_size - img_h) // 2))
        img = new_im
        img = img.resize(image_shape)
        img = np.array(img)
        img = img / 255.0   #노멀라이징???

        # print(np.mean(img), (img_w, img_h), np.array(img).shape)

        Image_batch = np.concatenate((image_batch, img[np.newaxis, ...].astype(np.float32)), axis=0)
        data_all.append([img.astype(np.float32), fishName_list.index(fish)])

print("data_all length = " + str(len(data_all)))

npData_all = np.array(data_all)
random.shuffle(npData_all)
x_data = npData_all[:,0]
y_data = np_utils.to_categorical(npData_all[:,1])

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=321)
# train / test 셋 생성
x_train = x_data[:int(len(x_data) * 0.8)]
x_test = x_data[int(len(x_data) * 0.8):]

y_train = y_data[:int(len(y_data) * 0.8)]
y_test = y_data[int(len(y_data) * 0.8):]

print(np.mean(x_train[0]), np.array(x_train[0]).shape)
print(len(x_train), len(x_test), len(y_train), len(y_test))

#Debug
plt.figure()
plt.imshow(x_train[100])
plt.colorbar()
plt.grid(False)
plt.show()

history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test,y_test))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()