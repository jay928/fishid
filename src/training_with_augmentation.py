from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow_hub as hub

DIR = "/data/fishid/source/integration"
BATCH_SIZE, EPOCHS, LR, LABELS_SIZE, DECAY = 32, 30, 0.001, 50, 1e-6

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=10,
    shear_range=0.5,
    vertical_flip=True,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    subset='validation',
    color_mode='rgb',
    class_mode='categorical')

# module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3", trainable=True)
module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3", trainable=False)
model = tf.keras.Sequential([
    hub.KerasLayer(module),
    #     tf.keras.layers.Dense(1024, activation='relu'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dropout(0.125),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.125),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.125),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.125),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.125),
    tf.keras.layers.Dense(LABELS_SIZE, activation='softmax')
])

adam = tf.keras.optimizers.Adam(lr=LR, decay=DECAY)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=625,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=validation_generator,
    validation_steps=68)

model.save('/data/fishid/fishid_20190929_11.h5')
