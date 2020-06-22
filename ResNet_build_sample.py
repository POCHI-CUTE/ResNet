from functools import reduce
from keras import backend as K
from keras.layers import (Activation, Add, GlobalAveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Flatten, Input,
                          MaxPooling2D)
from keras.models import Model
from keras.regularizers import l2
from functions import (basic_block, bottleneck_block, compose, ResNetConv2D, residual_blocks)

class ResnetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions):
        if block_type == 'basic':
            block_fn = basic_block
        elif block_type == 'bottleneck':
            block_fn = bottleneck_block

        input = Input(shape=input_shape)

        conv1 = compose(ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),
                        BatchNormalization(),
                        Activation('relu'))(input)

        pool1 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(block_fn, filters=filters, repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        block = compose(BatchNormalization(),
                        Activation('relu'))(block)

        pool2 = GlobalAveragePooling2D()(block)

        fc1 = Dense(units=num_outputs,
                    kernel_initializer='he_normal',
                    activation='softmax')(pool2)

        return Model(inputs=input, outputs=fc1)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 8, 36, 3])