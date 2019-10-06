from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

DIR = '/Volumes/SD/deeplearning/data/fish/integration'
# DIR = "/data/fishid/source/integration"
BATCH_SIZE, EPOCHS, LR, LABELS_SIZE, DECAY = 32, 100, 0.002, 50, 1e-5
PIXEL = 331

training_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=10,
    shear_range=0.5,
    vertical_flip=True,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1)

train_generator = training_datagen.flow_from_directory(
    DIR,
    target_size=(PIXEL, PIXEL),
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='rgb',
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    DIR,
    target_size=(PIXEL, PIXEL),
    batch_size=BATCH_SIZE,
    subset='validation',
    color_mode='rgb',
    class_mode='categorical')

inception_module = hub.Module("https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/3", trainable=False)
nasnet_module = hub.Module("https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/3", trainable=False)

main_input = tf.keras.layers.Input(shape=(PIXEL, PIXEL, 3), dtype='float')

im = hub.KerasLayer(inception_module)(main_input)
# idense = tf.keras.layers.Dense(2048, activation='relu')(im)
# ibn = tf.keras.layers.BatchNormalization()(idense)
# ido = tf.keras.layers.Dropout(0.125)(ibn)

nm = hub.KerasLayer(nasnet_module)(main_input)
# ndense = tf.keras.layers.Dense(2048, activation='relu')(nm)
# nbn = tf.keras.layers.BatchNormalization()(ndense)
# ndo = tf.keras.layers.Dropout(0.125)(nbn)

# merge = tf.keras.layers.concatenate([ido, ndo])
merge = tf.keras.layers.concatenate([im, nm])

mdense1 =  tf.keras.layers.Dense(2048, activation='relu')(merge)
mbn1 = tf.keras.layers.BatchNormalization()(mdense1)
mdo1 = tf.keras.layers.Dropout(0.125)(mbn1)
mdense2 =  tf.keras.layers.Dense(2048, activation='relu')(mdo1)
mbn2 = tf.keras.layers.BatchNormalization()(mdense2)
mdo2 = tf.keras.layers.Dropout(0.125)(mbn2)
mdense3 =  tf.keras.layers.Dense(2048, activation='relu')(mdo2)
mbn3 = tf.keras.layers.BatchNormalization()(mdense3)
mdo3 = tf.keras.layers.Dropout(0.125)(mbn3)
mdense4 =  tf.keras.layers.Dense(2048, activation='relu')(mdo3)
mbn4 = tf.keras.layers.BatchNormalization()(mdense4)
mdo4 = tf.keras.layers.Dropout(0.125)(mbn4)
output = tf.keras.layers.Dense(LABELS_SIZE, activation='softmax')(mdo4)
model = tf.keras.Model(inputs=main_input, outputs=output)

adam = tf.keras.optimizers.Adam(lr=LR, decay=DECAY)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=validation_generator)

model.save('/data/fishid/fishid_20190929_11.h5')
