import time as tiimmee

import keras
import numpy as np
import tensorflow as tf
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          concatenate)
from keras.models import Model
from keras.utils import np_utils, plot_model

import loaddata


def importdata(dataset):
    # Ｔhis function import the data from the database(csv file)
    # And return the feature and the label of the problem
    # 'dataset' is the name of the file
    # 'train_feature, train_label, test_feature, test_label' 
    # is training data
    # 'inputnum' is number of input data
    # 'nb_classes' is the number of class

    # load the data
    questionpath = 'dataset/dataset-question1.txt'
    answerpath = 'dataset/dataset-answer1.txt'
    input_data, output_data = loaddata.load(questionpath, answerpath)

    input_data = np.array(input_data)
    output_data = np.array(output_data)
    features = input_data[:, 1:]
    labels = output_data
    nb_classes = int(max(labels))+1
    inputnum = features.shape[1]

    # devide the input data, devide it to train data, test_data and left data
    datashape = features.shape
    num_train = int(datashape[0]*0.8)
    num_test = int(datashape[0]*0.2)
    
    train_feature = features[:num_train, :]
    train_label = labels[:num_train]
    test_feature = features[num_train:num_test+num_train, :]
    test_label = labels[num_train:num_test+num_train]

    # reshape the data
    train_feature = train_feature.reshape(train_feature.shape[0], inputnum)
    test_feature = test_feature.reshape(test_feature.shape[0], inputnum)
    train_label = train_label.reshape(train_label.shape[0], 1)
    test_label = test_label.reshape(test_label.shape[0], 1)

    # translate the label data into onehot shape
    train_label = np_utils.to_categorical(train_label, nb_classes)
    test_label = np_utils.to_categorical(test_label, nb_classes)

    return train_feature, train_label, test_feature, test_label, inputnum, nb_classes


def creatmodel_ann(inputnum, numOfJobs, layer=15):
    # This function is used to create the ann model in keras
    # 'inputnum' is number of input data
    # 'numOfJobs' is the number of class
    # Layer is the number of the hidden ann layers
    # The output model is the created model used to train
    # See http://keras-cn.readthedocs.io/en/latest/models/model/

    sess = tf.InteractiveSession()
    input1 = Input(shape=(inputnum, ), name='i1')
    mid = Dense(200, activation='relu', use_bias=True)(input1)
    for i in range(layer):
        mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dense(80, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(50, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(20, activation='relu', use_bias=True)(mid)
    out = Dense(numOfJobs, activation='softmax',
                name='out', use_bias=True)(mid)

    model = Model(input1, out)
    sgd = keras.optimizers.SGD(
        lr=0.11, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    return model


def triannetwork(model, train_feature, train_label, test_feature, test_label,
                 batch_size, epochs):
    # network training function
    # see http://keras-cn.readthedocs.io/en/latest/models/model/

    model.fit(
        [train_feature], train_label, batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data=([test_feature], test_label)
    )
    return model


def savenetwork(model, name):
    # This function is used to save the the medol trained above

    time_now = int(tiimmee.time())
    time_local = tiimmee.localtime(time_now)
    time = tiimmee.strftime("%Y_%m_%d::%H_%M_%S", time_local)
    print('current time is :', time)
    savename = './model/' + 'ann_schedual_' + time + name+'.h5'
    model.save(savename)
    return savename


def main(dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'):

    # ann parmater
    layerofann = 15


    # import the the data
    # dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'
    train_feature, train_label, test_feature, test_label, inputnum, nb_classes = importdata(
        dataset)
    print('successfully import the dataset :', dataset)
    print('the shape of train_feature is :', train_feature.shape)
    print('the shape of train_label is :', train_label.shape)
    print('the shape of test_feature is :', test_feature.shape)
    print('the shape of test_label is :', test_label.shape)

    # create the nn model
    model = creatmodel_ann(inputnum, nb_classes, layer=layerofann)
    plot_model(model, to_file='model.png')  # draw the figure of this model

    # training parmater
    batch_size = 49
    epochs = 4000
    
    # train the network
    model = triannetwork(model, train_feature, train_label,
                         test_feature, test_label, batch_size, epochs)

    # save the model 
    savename = "ann_layer"+str(layerofann) + '_' + dataset
    savename = savenetwork(model, savename)
    return test_feature, test_label, savename

if __name__ == '__main__':
    test_feature, test_label, svaename = main()
