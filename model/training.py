import tensorflow as tf
import tensorflow_hub as hub
# from tensorboardcolab import *

EPOCHS = 10

variables = [
    # {'batch_size': 32, 'lr': 0.001, 'dense_size': 32},
    # {'batch_size': 64, 'lr': 0.001, 'dense_size': 32},
    {'batch_size': 32, 'lr': 0.001, 'dense_size': 32},
    {'batch_size': 32, 'lr': 0.0005, 'dense_size': 32},
    #     {'batch_size': 32, 'lr': 0.0002, 'dense_size': 32},
    {'batch_size': 64, 'lr': 0.001, 'dense_size': 32},
    {'batch_size': 64, 'lr': 0.0005, 'dense_size': 32},
    #     {'batch_size': 64, 'lr': 0.0002, 'dense_size': 32},
    {'batch_size': 32, 'lr': 0.001, 'dense_size': 64},
    {'batch_size': 32, 'lr': 0.0005, 'dense_size': 64},
    #     {'batch_size': 32, 'lr': 0.0002, 'dense_size': 64},
    {'batch_size': 64, 'lr': 0.001, 'dense_size': 64},
    {'batch_size': 64, 'lr': 0.0005, 'dense_size': 64},
    #     {'batch_size': 64, 'lr': 0.0002, 'dense_size': 64},
]

# tf.disable_eager_execution()

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/3")
# module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3")

# tbc = TensorBoardColab()
# cb = TensorBoardColabCallback(tbc)

for variable in variables:
    print(variable)

    classifier = tf.keras.Sequential([
        hub.KerasLayer(module),
        tf.keras.layers.Dense(variable['dense_size'], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(variable['dense_size'], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(COUNT - len(NOISED), activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(lr=variable['lr'])
    classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #     log_dir = '/data/logs/' + str(variable['batch_size']) + str(variable['lr']) + str(variable['dense_size'])
    #     tlog = TensorBoard(log_dir=log_dir)

    classifier.fit(x_data, y_data, epochs=EPOCHS, batch_size=variable['batch_size'], verbose=1, shuffle=True, validation_split=0.05)

