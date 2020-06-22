from keras.utils import np_utils
from keras.datasets import cifar10

NB_CLASSES = 10

def get_preprocessed_cifar10(nb_classes=NB_CLASSES, debug=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if debug:
        print("cifar-10_tarin_shape = ", x_train.shape)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return (x_train, y_train), (x_test, y_test)