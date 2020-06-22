from keras import losses
from keras.optimizers import SGD
from ResNet_build_sample import ResnetBuilder
from cifar_10_preprocess import get_preprocessed_cifar10
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os


NB_CLASSES = 10
NB_EPOCH = 100
BATCH_SIZE = 256
VERBOSE = 1

class ResNetCifar10:
    def __init__(self):
        self.cifar10_inputShape=(32, 32, 3)
        self.momentum = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
        self.label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.ResNetModel = ResnetBuilder.build_resnet_18(self.cifar10_inputShape, NB_CLASSES)
        #plot_model(self.ResNetModel, to_file='ResNetModel.png', show_shapes=True, show_layer_names=True)
        self.ResNetModel.compile(optimizer=self.momentum, loss=losses.binary_crossentropy, metrics=['acc'])
        

    def train_cifar10(self, output_graph=True, save_weight=True):
        (x_train, y_train), (x_test, y_test) = get_preprocessed_cifar10()
        self.ResNetModel.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=1, validation_data=(x_test, y_test))

        trainDataAccuracy_array = []
        testDataAccuracy_array = []
        epoch_array = range(1, NB_EPOCH + 1)

        start_time = time.time()
        for epoch in range(NB_EPOCH):
            perm = np.random.permutation(x_train.shape[0])

            for i in range(0, x_train.shape[0], BATCH_SIZE):
                x_batch = x_train[perm[i : i + BATCH_SIZE]]
                y_batch = y_train[perm[i : i + BATCH_SIZE]]

                self.ResNetModel.train_on_batch(x_batch, y_batch)

            train_score = self.ResNetModel.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSE)
            test_score = self.ResNetModel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
            trainDataAccuracy_array.append(train_score[1])
            testDataAccuracy_array.append(test_score[1])
            interval = int(time.time() - start_time)
            #print("epoch = {0:d} / {1:d} --- 実行時間 = {2:d}[sec] --- 1epochに掛かる平均時間 = {3:.2f[sec]}".format(epoch + 1, NB_EPOCH, interval, interval / (epoch + 1)))
            print("Test score : {0:f} --- Test accuracy : {1:f}".format(test_score[0], test_score[1]))
        end_time = int(time.time() - start_time)

        if output_graph:
            plt.plot(epoch_array, trainDataAccuracy_array, label="train")
            plt.plot(epoch_array, testDataAccuracy_array, linestyle="--",label="test")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.title("ResNet with Cifar-10 ({0:d}[sec])".format(end_time))
            plt.legend()
            plt.show()

        if save_weight:
            self.ResNetModel.save_weights('ResNetModel_cifar10_weights.h5')

    def ResNet_predict_from_oneFile(self, filePath, file_name, debug=True):
        img = Image.open(filePath)
        img_resize = img.resize((self.cifar10_inputShape[0], self.cifar10_inputShape[1]), Image.LANCZOS)
        img2np = np.asarray(img_resize)
        img2np.flags.writeable = False
        img2np.astype('float32')
        img2np_norm = np.true_divide(img2np, 255)[np.newaxis, :, :, :]

        print("\n")
        if debug:
            print(img_resize.size)
            print(img2np.shape)
            print(img2np_norm.shape)
            img_resize.show()

        result = self.ResNetModel.predict(img2np_norm, batch_size=1)

        if debug:
            print(result)

        print(file_name + 'は' + self.label[np.argmax(result)] + 'です')
        print("\n")

test=ResNetCifar10()
test.train_cifar10()
test.ResNet_predict_from_oneFile(os.path.abspath('image/ライチョウ.jpeg'), 'ライチョウ.jpeg')
test.ResNet_predict_from_oneFile(os.path.abspath('image/airplane.jpeg'), 'airplane.jpeg')
test.ResNet_predict_from_oneFile(os.path.abspath('image/4WD_car.jpeg'), '4WD_car.jpeg')