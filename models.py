"""Model factories for Hnefatafl training."""

import sys

from keras import Model, Input
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, BatchNormalization, Add, Concatenate
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal

import hnefatafl as tafl


def initialize_residual_multiscale_cnn_model(num_channels=8, use_batchnorm=True, learning_rate=0.001):
    """Initialize CNN model optimized for board game learning."""
    print(f"Initializing CNN model v2 for board game learning ({num_channels} channels, batchnorm={use_batchnorm})")

    input_shape = (tafl.DIM, tafl.DIM, num_channels)
    std = 0.1

    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    res = x
    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation('relu')(x)

    branch1 = Conv2D(32, (5, 5), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(32, (3, 3), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    branch3 = Conv2D(32, (1, 1), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    print(f"Model initialized with learning_rate={learning_rate}")
    model.summary()
    return model


def initialize_cnn_model_claude(input_shape):
    """Initialize an alternate CNN model architecture."""
    std = 0.1
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)

    for _ in range(4):
        res = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Add()([x, res])
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

    branch1 = Conv2D(32, (5, 5), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch3 = Conv2D(32, (1, 1), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model


def initialize_compact_model_simple(num_channels=8, learning_rate=0.001):
    """Initialize compact model specifically for the 5x5 simple game mode."""
    print(f"Initializing minimal model for simple game mode ({num_channels} channels)")

    input_shape = (tafl.DIM, tafl.DIM, num_channels)
    std = 0.1

    inputs = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    x = Conv2D(4, (1, 1), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    print(f"Model initialized with learning_rate={learning_rate}")
    model.summary()
    return model


def validate_model_channels(model, expected_channels, model_name):
    """Ensure loaded model input channels match current encoding."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    model_channels = input_shape[-1]
    if model_channels != expected_channels:
        print(f"Error: {model_name} model expects {model_channels} channels, but current encoding uses {expected_channels}.")
        print("Hint: train/load compatible checkpoints that were built with the current 8-channel encoding.")
        sys.exit(1)
