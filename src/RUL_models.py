from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras import initializers
from tensorflow.keras import Model

def dilated_cnn_v3(Model_input):
    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    x = Conv1D(8, 13, padding='same', kernel_initializer='he_uniform',
                bias_initializer=initializer)(Model_input)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, 13, padding='same', dilation_rate=2, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 13, padding='same', dilation_rate=4, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 13, padding='same', dilation_rate=8, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 13, padding='same', dilation_rate=16, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, 13, padding='same', dilation_rate=32, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(1, 1, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = Flatten()(x)

    x = Dense(50, use_bias=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def dilated_cnn_v5(Model_input):
    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    x = Conv1D(8, 13, padding='same', kernel_initializer='he_uniform',
                bias_initializer=initializer)(Model_input)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, 13, padding='same', dilation_rate=2, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 13, padding='same', dilation_rate=4, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 13, padding='same', dilation_rate=8, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 13, padding='same', dilation_rate=16, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, 13, padding='same', dilation_rate=32, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(512, 13, padding='same', dilation_rate=64, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(1, 1, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = Flatten()(x)

    x = Dense(50, use_bias=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def dilated_cnn_v7(Model_input):
    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    x = Conv1D(8, 29, padding='same', kernel_initializer='he_uniform',
                bias_initializer=initializer)(Model_input)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, 29, padding='same', dilation_rate=2, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 29, padding='same', dilation_rate=4, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 29, padding='same', dilation_rate=8, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(1, 1, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = Flatten()(x)

    x = Dense(50, use_bias=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def cnn_v4(Model_input):
    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    x = Conv1D(8, 13, padding='same', kernel_initializer='he_uniform',
                bias_initializer=initializer)(Model_input)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(1, 1, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = Flatten()(x)

    x = Dense(50, use_bias=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def cnn_v5(Model_input):
    initializer = initializers.RandomNormal(mean=0., stddev=1.)
    x = Conv1D(8, 13, padding='same', kernel_initializer='he_uniform',
                bias_initializer=initializer)(Model_input)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(16, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(32, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 13, padding='same',kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(1, 1, kernel_initializer='he_uniform',
                bias_initializer=initializer)(x)
    x = Flatten()(x)

    x = Dense(50, use_bias=True)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def dnn_v1(Model_input):
    x = Flatten()(Model_input)

    x = Dense(2048)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(128)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output

def dnn_v2(Model_input):
    x = Flatten()(Model_input)

    x = Dense(4096)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1024)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(256)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(16)(x)
    x = BatchNormalization(axis=-1, momentum=0.1, epsilon=0.00001)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    output = Model(Model_input, x)

    return output