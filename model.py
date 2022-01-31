"""
Author :Yashish Maduwantha

Project : Model implementation of MatchboxNet (https://www.isca-speech.org/archive/pdfs/interspeech_2020/majumdar20_interspeech.pdf)

"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Conv1D, Flatten, BatchNormalization, MaxPool1D, ReLU, MaxPool2D, Conv2D, DepthwiseConv1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import NovoGrad
from pointwise_shuffle import pointwise_conv
from tensorflow.keras.utils import to_categorical
from warmup_lr_sheduler import WarmUpLearningRateScheduler

from matplotlib import pyplot
# pyplot.switch_backend('agg')

"""
    Uncomment the instruction below to enable Mixed precision training in the GPUs as used in the paper
"""
# tf.keras.mixed_precision.set_global_policy("mixed_float16")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def sub_match_block(input, k_size):
    """
    Compute output for the sub block
    :param input: input tensor
    :param k_size: size of the kernel
    :return: output from the sub block
    """
    # block layers
    depth_conv = DepthwiseConv1D(kernel_size=k_size, strides=1, padding='same', depth_multiplier=1)(input)
    point_conv = pointwise_conv(depth_conv, depth_conv.shape[2], 64)
    # depth_point_conv = SeparableConv1D(filters=64, kernel_size=k_size, strides=1, depth_multiplier=1)(input)

    bnorm_1 = BatchNormalization()(point_conv)
    act_1 = ReLU()(bnorm_1)
    drp_1 = Dropout(DROPOUT_PROB1)(act_1)

    return drp_1

def match_block(input, k_size, R=2):
    """
    Compute Main block output
    :param input: input tensor
    :param k_size: kernel size for all Conv layers
    :param R: Number of sub blocks
    :return: output from Main block
    """

    # skip connection
    pwise_conv = pointwise_conv(input, input.shape[2], 64)
    bnorm_skip = BatchNormalization()(pwise_conv)

    out = input
    for i in range(0,R):
        out = sub_match_block(out, k_size)

    depth_conv = DepthwiseConv1D(kernel_size = k_size, strides=1, padding='same', depth_multiplier=1)(out)
    point_conv = pointwise_conv(depth_conv, depth_conv.shape[2], 64)
    bnorm_1 = BatchNormalization()(point_conv)

    # add skip input
    out_skip = bnorm_1 + bnorm_skip

    act = ReLU()(out_skip)
    drp = Dropout(DROPOUT_PROB1)(act)

    return drp

def matchbox_model(sample_x):
    """
    Implmentation of the MatchboxNet model architecture
    :param sample_x: Input tensor
    :return: Model object
    """
    No_of_B = 3
    No_of_R = 2
    kernel_sizes = [13, 15, 17]
    print("-------------sample x shape---------------")
    print(sample_x.shape)
    inp = Input(shape=(sample_x.shape[0], sample_x.shape[1]))

    #prologue layer
    conv1 = Conv1D(NUM_FILTERS_CONV1, FILTER_SIZE1, strides=2, padding='same')(inp)
    bn1 = BatchNormalization()(conv1)
    pl1_out = ReLU()(bn1)

    #call block_layers
    out = pl1_out
    for i in range(0, No_of_B):
        out = match_block(out, kernel_sizes[i], No_of_R)

    #epilog layer_1
    conv2 = Conv1D(NUM_FILTERS_CONV2, FILTER_SIZE2, padding='same', dilation_rate=2)(out)
    bn2 = BatchNormalization()(conv2)
    pl2_out = ReLU()(bn2)

    #epilog layer_2
    conv3 = Conv1D(NUM_FILTERS_CONV3, FILTER_SIZE3, padding='same')(pl2_out)
    bn3 = BatchNormalization()(conv3)
    pl3_out = ReLU()(bn3)

    """
    final_layer: The original paper https://arxiv.org/pdf/1910.10261.pdf has a pointwise convolution defined 
    for final Convolution layer whereas the MatchboxNet paper mentions it as a 1x1 convolution. 
    """
    # conv3 = Conv1D(NUM_CLASSES, FILTER_SIZE4)(pl3_out)
    pwise_conv = pointwise_conv(pl3_out, pl3_out.shape[2], NUM_CLASSES)
    flaten = Flatten()(pwise_conv)
    out = Dense(NUM_CLASSES, activation='softmax')(flaten)
    return Model(inputs=inp, outputs=out)


def fine_tune(x_train, y_train, x_test, y_test, x_val, y_val):
    """
    Model training and evaluation
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param x_val:
    :param y_val:
    :return: Test accuracy evaluated on the test data
    """
    classifier = matchbox_model(x_train[0])
    opt = NovoGrad(lr=1e-3, beta_1=0.95, beta_2=0.5)
    # opt = Adam(lr = LR)
    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()

    """
    warm-up learning rate scheduler 
    (Not the exact learning rate scheduler implemented in the paper)
    """
    # Compute the number of warmup batches.
    warmup_epoch = EPOCHS*(5/100)  # 5% of the Epochs
    warmup_batches = warmup_epoch * x_train.shape[0] / BATCH_SIZE
    warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=0.05)

    # fit the model
    history = classifier.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              verbose=1, validation_data=(x_val, y_val), callbacks=[warm_up_lr])

    # compute train accuracy
    _, train_acc = classifier.evaluate(x_train, y_train, verbose=0)
    # test on data
    _, test_acc = classifier.evaluate(x_test, y_test, verbose=0)

    # plots for loss and accuracy change over epochs
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    return test_acc

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    NUM_CLASSES = 30  #depends of the version of the dataset

    # Define Model Parameters
    # test params defined below.
    params = {'FILTER_SIZE': (11, 29, 1, 1), 'NUM_FILTERS_CONV': (128, 128, 128), 'DROPOUT_PROB1': 0.5}
    print(params)

    NUM_FILTERS_CONV1 = params['NUM_FILTERS_CONV'][0]
    NUM_FILTERS_CONV2 = params['NUM_FILTERS_CONV'][1]
    NUM_FILTERS_CONV3 = params['NUM_FILTERS_CONV'][2]
    # LR = 1e-4
    DROPOUT_PROB1 = params['DROPOUT_PROB1']
    EPOCHS = 5
    BATCH_SIZE = 128
    FILTER_SIZE1 = params['FILTER_SIZE'][0]
    FILTER_SIZE2 = params['FILTER_SIZE'][1]
    FILTER_SIZE3 = params['FILTER_SIZE'][2]
    FILTER_SIZE4 = params['FILTER_SIZE'][3]

    """
    Define a random set of tensors for the sake of testing the model
    """
    X_tr  = tf.random.uniform(shape=[5000, 128, 64], seed=seed)
    X_val = tf.random.uniform(shape=[100, 128, 64], seed=seed)
    X_te = tf.random.uniform(shape=[200, 128, 64], seed=seed)

    Y_tr = tf.random.uniform(shape=[5000,1], minval=0, maxval= NUM_CLASSES-1, dtype= tf.int32, seed=seed)
    Y_val = tf.random.uniform(shape=[100,1], minval=0, maxval= NUM_CLASSES-1, dtype= tf.int32, seed=seed)
    Y_te = tf.random.uniform(shape=[200,1], minval=0, maxval= NUM_CLASSES-1, dtype= tf.int32, seed=seed)

    Y_tr = to_categorical(Y_tr, NUM_CLASSES)
    Y_val = to_categorical(Y_val, NUM_CLASSES)
    Y_te = to_categorical(Y_te, NUM_CLASSES)

    accs = fine_tune(X_tr, Y_tr, X_te, Y_te, X_val, Y_val)

    print('Test set accuracy:', accs)


