import tensorflow as tf
from tensorflow.keras import layers
from clr import cyclic_learning_rate
import numpy as np
import random
from keras.layers import Dense, Input, dot
from keras.models import Model

def get_all_samples(conjunction):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples



def BuildModel(train_gae,train_nmf,label):

    l1 = len(train_gae[1])
    l2=len(train_nmf[1])
    inputs_gae = Input(shape=(l1,))
    inputs_nmf = Input(shape=(l2,))

    x = Dense(128, activation='relu')(inputs_gae)
    x = Dense(64, activation='relu')(x)
    y = Dense(128, activation='relu')(inputs_nmf)
    y = Dense(64, activation='relu')(y)
    attention = layers.Attention()
    result = attention([x, y])
    predictions = Dense(1, activation='sigmoid')(result)

    model = Model(inputs=[inputs_gae,inputs_nmf], outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y)
    return model

