import tensorflow as tf
import tensorflow_hub as hub
# from tensorboardcolab import *

EPOCHS = 10

PATH = '/data/fishid/source/'
TARGET_SIZE = 224
COUNT = 25
NOISED = [16, 17, 18, 19]

x_data = []
y_data = []
for i in range(0, COUNT):
    print(i)
    if i in NOISED:
        continue

    x_part = np.load(PATH + "x" + str(i) + "_" + str(TARGET_SIZE) + '.npy', allow_pickle=True)
    y_part = np.load(PATH + "y" + str(i) + "_" + str(TARGET_SIZE) + '.npy', allow_pickle=True)
    #     x_part = tf.image.convert_image_dtype(x_part, tf.float32)

    if i == 0:
        x_data = x_part
        y_data = y_part
        continue

    x_data = np.concatenate((x_data, x_part), axis=0)
    y = y_data[-1] + 1
    for j in range(0, len(x_part)):
        y_data = np.concatenate((y_data, [y]), axis=0)

#     y_data = np.concatenate((y_data, y_part), axis=0)

x_data = x_data / 255.0
y_data = np_utils.to_categorical(y_data)

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3")
# module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3")

# tbc = TensorBoardColab()
# cb = TensorBoardColabCallback(tbc)

for variable in variables:
    print(variable)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(module),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(COUNT - len(NOISED), activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(lr=0.001)
    classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #     log_dir = '/data/logs/' + str(variable['batch_size']) + str(variable['lr']) + str(variable['dense_size'])
    #     tlog = TensorBoard(log_dir=log_dir)

    classifier.fit(x_data, y_data, epochs=EPOCHS, batch_size=32, verbose=1, shuffle=True, validation_split=0.05)

