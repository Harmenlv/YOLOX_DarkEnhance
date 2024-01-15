def non(x, y, out_dim):
    out_dam = out_dim / 2
    y1 = Conv2D(int(out_dam), 1, padding='same', data_format='channels_last')(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    y2 = Conv2D(int(out_dam), 1, padding='same', data_format='channels_last')(y)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    y3 = multiply([y1, y2])
    y3 = BatchNormalization()(y3)
    y3 = Activation('sigmoid')(y3)

    y4 = Conv2D(int(out_dam), 1, padding='same', data_format='channels_last')(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation('relu')(y4)

    non = multiply([y3, y4])
    y5 = Conv2D(int(out_dim), 1, padding='same', data_format='channels_last')(non)
    y5 = BatchNormalization()(y5)
    y5 = Activation('relu')(y5)
    non = add([y5, x])
    return non