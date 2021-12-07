from PIL import Image
import numpy as np

import tensorflow as tf
from keras.optimizers import adam_v2
import keras.losses
import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt


# build data
def build_dataset():
    individuals = ['yaleB0' + str(k) for k in range(1, 10)]
    individuals.extend(['yaleB' + str(k) for k in range(10, 14)])
    individuals.extend(['yaleB' + str(k) for k in range(15, 40)])

    # get data - images and labels
    dataset = []
    labels = []

    for id_ in individuals:
        filename = []
        for files in os.walk('./CroppedYale/' + id_):
            filename.append(files)
        img_name = [filename[0][2][k] for k in range(2, len(filename[0][2]) - 1)]
        labels.extend([id_ for k in range(len(img_name))])
        for k in img_name:
            if k.find('.pgm') == -1:
                continue
            img = Image.open('./CroppedYale/' + id_ + '/' + k)
            img = img.resize((224, 224), resample=2)
            img = np.array(img)
            # img = img.reshape(img.shape[0]*img.shape[1])
            dataset.append(img)
    # # convert labels(string) to labels(integer)
    labels_oneshot = []
    labels_int = []
    for i in range(len(labels)):
        temp = int(labels[i].replace('yaleB', ''))
        temp2 = np.zeros(39, dtype=int)
        temp2[temp-1] = 1
        labels_oneshot.append(temp2)
        labels_int.append(temp)
    labels = pd.DataFrame()
    labels['lb'] = labels_oneshot
    labels['gr'] = labels_int
    print(labels)
    # training set, validation set, test set
    train_y = labels.groupby('gr').sample(frac=0.8)
    train_x = []
    for idx in train_y.index:
        train_x.append(dataset[idx])
    # create validation set
    validation_y = train_y.sample(frac=0.2)
    validation_x = []
    for idx in validation_y.index:
        validation_x.append(dataset[idx])
    # create test set
    test_y = []
    test_x = []
    for idx in labels.index:
        if idx not in train_y.index:
            test_y.append(labels['lb'][idx])
            test_x.append(dataset[idx])
    print(train_y.index)
    temp = []
    for value in train_y['lb']:
        temp.append(value)
    train_y = tf.stack(temp)
    temp = []
    for value in validation_y['lb']:
        temp.append(value)
    validation_y = np.array(temp)
    # format unification
    train_x = np.array(train_x)
    train_x = np.expand_dims(train_x, -1)
    print(train_y.shape)
    validation_x = np.array(validation_x)
    validation_x = np.expand_dims(validation_x, -1)
    training_data = (train_x, train_y)  # size of 1690, divisible by 10
    validation_data = (validation_x, validation_y)
    tests_data = (test_x, test_y)
    return training_data, validation_data, tests_data


train_data, val_data, test_data = build_dataset()
# build model
model = Sequential()

model.add(Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(units=4096, activation="relu"))

model.add(Dense(units=4096, activation="relu"))

model.add(Dense(units=39, activation="softmax"))


opt = adam_v2.Adam(lr=0.001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

check_point = ModelCheckpoint("face_detector.h15", monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True)

hist = model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=10, callbacks=[check_point, early_stop])


