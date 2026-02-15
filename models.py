"""Model factories for Hnefatafl training."""

import sys

from keras import Model, Input
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, BatchNormalization, Add, Concatenate
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal

import hnefatafl as tafl


MODEL_PRESETS = {
    "simple": {
        "kind": "compact",
    },
    "brandubh": {
        "kind": "residual_multiscale",
        "base_filters": 48,
        "num_res_blocks": 1,
        "branch_filters": 24,
        "trunk_filters": 24,
        "dense_units": (128, 64),
        "branch_dropout": 0.15,
        "dense_dropout": 0.20,
        "init_std": 0.1,
    },
    "hnefatafl": {
        "kind": "residual_multiscale",
        "base_filters": 64,
        "num_res_blocks": 1,
        "branch_filters": 32,
        "trunk_filters": 32,
        "dense_units": (256, 128),
        "branch_dropout": 0.20,
        "dense_dropout": 0.30,
        "init_std": 0.1,
    },
}


def resolve_model_preset(game_name, model_preset="auto"):
    """Resolve preset name based on game mode and optional explicit override."""
    if model_preset is None:
        model_preset = "auto"
    model_preset = model_preset.lower()
    game_key = game_name.lower()

    if model_preset != "auto":
        if model_preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset: {model_preset}")
        return model_preset

    if game_key in MODEL_PRESETS:
        return game_key
    return "hnefatafl"


def build_residual_multiscale_value_cnn(num_channels=8, use_batchnorm=True, learning_rate=0.001, config=None):
    """Build a residual + multiscale value CNN from a config dictionary."""
    if config is None:
        config = MODEL_PRESETS["hnefatafl"]

    base_filters = int(config["base_filters"])
    num_res_blocks = int(config["num_res_blocks"])
    branch_filters = int(config["branch_filters"])
    trunk_filters = int(config["trunk_filters"])
    dense_units = tuple(config["dense_units"])
    branch_dropout = float(config["branch_dropout"])
    dense_dropout = float(config["dense_dropout"])
    std = float(config.get("init_std", 0.1))

    input_shape = (tafl.DIM, tafl.DIM, num_channels)
    inputs = Input(shape=input_shape)

    x = Conv2D(base_filters, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(num_res_blocks):
        res = x
        x = Conv2D(base_filters, (3, 3), padding='same', use_bias=not use_batchnorm,
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(base_filters, (3, 3), padding='same', use_bias=not use_batchnorm,
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Add()([x, res])
        x = Activation('relu')(x)

    branch1 = Conv2D(branch_filters, (5, 5), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(branch_filters, (3, 3), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    branch3 = Conv2D(branch_filters, (1, 1), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(branch_dropout)(x)

    x = Conv2D(trunk_filters, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(dense_units[0], activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(dense_units[1], activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def initialize_compact_model_simple(num_channels=8, learning_rate=0.001):
    """Initialize compact model specifically for the 5x5 simple game mode."""
    print(f"Initializing compact model for simple game mode ({num_channels} channels)")

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


def initialize_model_for_game(game_name, num_channels=8, use_batchnorm=True, learning_rate=0.001, model_preset="auto"):
    """Initialize model using preset chosen from game mode or explicit override."""
    preset = resolve_model_preset(game_name, model_preset=model_preset)
    config = MODEL_PRESETS[preset]

    if config["kind"] == "compact":
        return initialize_compact_model_simple(num_channels=num_channels, learning_rate=learning_rate)

    print(
        f"Initializing {preset} preset (residual-multiscale CNN): "
        f"filters={config['base_filters']}, res_blocks={config['num_res_blocks']}, "
        f"dense={config['dense_units']}, batchnorm={use_batchnorm}"
    )
    model = build_residual_multiscale_value_cnn(
        num_channels=num_channels,
        use_batchnorm=use_batchnorm,
        learning_rate=learning_rate,
        config=config,
    )
    print(f"Model initialized with learning_rate={learning_rate}")
    model.summary()
    return model


def initialize_residual_multiscale_cnn_model(num_channels=8, use_batchnorm=True, learning_rate=0.001):
    """Compatibility wrapper for previous name."""
    return initialize_model_for_game(
        game_name="hnefatafl",
        num_channels=num_channels,
        use_batchnorm=use_batchnorm,
        learning_rate=learning_rate,
        model_preset="hnefatafl",
    )


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
