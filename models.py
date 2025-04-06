from keras import Model, Input
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, Add, Concatenate
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal


def simple_model(dim, learning_rate=0.01):
    """ Initialize a compact model for simple game mode. """
    print("Initializing simple model")
    
    input_shape = (dim, dim, 3)  # 3 channels: attacker, king, defender
    std = 0.1
    
    inputs = Input(shape=input_shape)
    
    # Single conv layer to detect piece positions and basic spatial patterns
    x = Conv2D(8, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    
    # Direct 1x1 convolution to focus on piece presence
    x = Conv2D(4, (1, 1), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Flatten and minimal dense layer
    x = Flatten()(x)
    x = Dense(16, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Output with tanh activation
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Use provided learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.summary()
    return model


def sonnet_model(dim, learning_rate=0.001):
    """ Initialize a CNN model optimized for 7x7 board game learning. """
    print("Initializing sonnet model")
    
    input_shape = (dim, dim, 3)  # 3 channels: attacker, king, defender
    std = 0.1  # Small std to prevent score explosion through deep network
    
    inputs = Input(shape=input_shape)
    
    # Initial convolution block
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    
    # First residual block
    res = x
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Add()([x, res])
    x = Activation('relu')(x)
    
    # Pattern recognition block - different kernel sizes
    # 5x5 for broader patterns like surrounding threats
    branch1 = Conv2D(32, (5, 5), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # 3x3 for local patterns
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # 1x1 for point-wise patterns
    branch3 = Conv2D(32, (1, 1), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)
    
    # Final convolution to reduce channels
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Final layer with tanh to bound outputs between -1 and 1
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Use provided learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.summary()
    return model


def claude_model(dim, learning_rate=0.01):
    """ Initialize a CNN model with residual blocks for complex board game learning. """
    print("Initializing claude model")
    
    input_shape = (dim, dim, 3)  # 3 channels: attacker, king, defender
    std = 0.1
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    
    # Residual blocks
    for _ in range(4):
        res = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Add()([x, res])
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
    
    # Pattern recognition block
    branch1 = Conv2D(32, (5, 5), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch3 = Conv2D(32, (1, 1), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)
    
    # Final convolution
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', 
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Output layer
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Use provided learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model 